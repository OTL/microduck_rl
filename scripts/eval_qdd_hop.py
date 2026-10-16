"""Evaluate a hop (jumping skill) ONNX policy in mjlab: nominal and domain-randomised.

    python scripts/eval_qdd_hop.py <policy.onnx> [--envs 32] [--json out.json]
    python scripts/eval_qdd_hop.py zero ...      # zero action: a feed-forward skill's table alone

One skill execution per env from the standing pose (no RSI). Per env:
  - CoM rise: highest whole-robot CoM after take-off minus the CoM at take-off, and
    minus the standing CoM; take-off speed (max CoM vz); foot clearance: the largest
    height, over the flight, of the lowest point of both soles (corners of the 150 x 70 mm
    soles, every physics substep)
  - flight time (both soles off the floor, physics-substep resolution, 2.5 ms)
  - landing load per foot: peak 5 ms average vertical ground force / (m g)
  - joint peak speed / torque, and the peak use of the torque-speed envelope
    (|tau| / available torque at that speed while motoring)
  - motor bus current: electrical model of the robot repo's analysis/hop.py
    (GIM6010-8: tau w + 1.5 R (tau / Kt)^2, GIM3510-8: tau w / 0.7; / 22 V), peak of the
    5 ms average
  - falls (termination before the end), torso pitch change in flight, final pose error
    (max |q - standing| and torso tilt one step before the end)
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv

from mjlab_microduck.actuator.qdd_motor import KT_6010, R_PH_6010, motoring_limit
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.qdd_biped_skill_env_cfg import JUMP_SPEC, make_qdd_biped_skill_env_cfg
from mjlab_microduck.tasks.qdd_skill import FOOT_HALF_LENGTH_M, FOOT_HALF_WIDTH_M

V_BUS = 22.0
SETTLE_SKIP_S = 0.3  # ignore the first 0.3 s (spawn drop) for flights and peaks


def run(onnx_path: str, play: bool, n_envs: int, seed: int = 0) -> dict:
    import onnxruntime as ort

    torch.manual_seed(seed)
    cfg = make_qdd_biped_skill_env_cfg(JUMP_SPEC, play=play)
    cfg.scene.num_envs = n_envs
    cfg.commands["skill"].rsi_fraction = 0.0
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    sess = None if onnx_path == "zero" else ort.InferenceSession(onnx_path)
    robot = env.scene["robot"]
    sensor = env.scene.sensors["feet_ground_contact"]
    sk = env.command_manager.get_term("skill")
    mg = microduck_mdp._robot_mass(env, robot) * 9.81
    ankle = torch.tensor([n.endswith("ankle_pitch") for n in robot.joint_names])
    sites = [robot.find_sites(n)[0][0] for n in ("l_foot_site", "r_foot_site")]
    nj = len(robot.joint_names)
    B = n_envs

    sub = dict(w=[], tau=[], util=[], bus=[], sole=[])
    orig_step = env.sim.step

    def step_hook():
        orig_step()
        # speed the actuator used for its torque limit (before the step) and the applied torque
        w = robot.data.joint_vel.clone()
        for a in robot.actuators:
            if getattr(a, "_vel", None) is not None:
                w[:, a.target_ids] = a._vel
        tau = robot.data.qfrc_actuator.clone()
        sub["w"].append(w.abs())
        sub["tau"].append(tau.abs())
        lim = torch.where(ankle, motoring_limit("3510", w, torch.tensor(V_BUS)), motoring_limit("6010", w, torch.tensor(V_BUS)))
        lim = torch.minimum(lim, torch.where(ankle, torch.tensor(6.28), torch.tensor(11.0)))
        motoring = tau * w > 0
        util = torch.where(motoring, tau.abs() / lim.clamp(min=1e-3), torch.zeros_like(tau))
        sub["util"].append(util)
        p6010 = tau * w + 1.5 * R_PH_6010 * (tau / KT_6010) ** 2
        pank = torch.where(tau * w > 0, tau * w / 0.7, tau * w * 0.7)
        p = torch.where(ankle, pank, p6010).sum(dim=1)
        sub["bus"].append(p / V_BUS)
        sub["sole"].append(microduck_mdp.sole_lowest_z(robot, sites, FOOT_HALF_LENGTH_M, FOOT_HALF_WIDTH_M))

    env.sim.step = step_hook
    obs, _ = env.reset()
    n = sk.n
    com_stand = None
    rec = {k: [] for k in ("com_z", "com_vz", "air", "fz", "sole_z", "pitch")}
    alive = torch.ones(B, dtype=torch.bool)
    final = None
    for k in range(n - 1):  # the last step times out and respawns (not part of the hop)
        x = obs["actor"].numpy().astype(np.float32)
        if sess is None:
            act = np.zeros((B, env.action_manager.total_action_dim), dtype=np.float32)
        else:
            act = np.concatenate([sess.run(None, {"obs": x[i : i + 1]})[0] for i in range(B)])
        obs, _, term, trunc, _ = env.step(torch.from_numpy(act))
        com, vel = microduck_mdp.robot_com(env, robot)
        if com_stand is None:
            com_stand = com[:, 2].clone()
        fh = sensor.data.force_history[..., 2].abs()  # [B, 2, H] (H = substeps, 0 = newest)
        fh = torch.flip(fh, dims=[2])  # oldest first
        rec["com_z"].append(com[:, 2].clone())
        rec["com_vz"].append(vel[:, 2].clone())
        rec["fz"].append(fh.clone())
        rec["sole_z"].append(robot.data.site_pos_w[:, sites, 2].min(dim=1).values.clone())
        g = robot.data.projected_gravity_b
        rec["pitch"].append(torch.rad2deg(torch.atan2(g[:, 0], -g[:, 2])).clone())
        alive &= ~term
        if k == n - 2:
            q0 = robot.data.default_joint_pos
            err = torch.rad2deg((robot.data.joint_pos - q0).abs().max(dim=1).values)
            tilt = torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1, 1)))
            final = dict(pose_err_deg=err.clone(), tilt_deg=tilt.clone())
    env.sim.step = orig_step
    dt_sub = env.physics_dt
    H = env.cfg.decimation
    com_z = torch.stack(rec["com_z"], 1)  # [B, n]
    com_vz = torch.stack(rec["com_vz"], 1)
    fz = torch.cat(rec["fz"], dim=2) / mg  # [B, 2, n*H] substep forces in m g
    sole = torch.stack(rec["sole_z"], 1)
    pitch = torch.stack(rec["pitch"], 1)
    skip = round(SETTLE_SKIP_S / dt_sub)  # spawn transient (dropped from 5-10 mm, PD settling)
    bus = torch.stack(sub["bus"], 1)  # [B, n*H]
    sole_sub = torch.stack(sub["sole"], 1)  # [B, n*H] lowest sole point (both feet)
    bus[:, :skip] = 0.0
    for key in ("w", "tau", "util"):
        sub[key] = torch.stack(sub[key][skip:], 1).amax(dim=1)
    nav = max(1, round(0.005 / dt_sub))
    kern = torch.ones(1, 1, nav) / nav
    fz5 = torch.nn.functional.conv1d(fz.reshape(-1, 1, fz.shape[-1]), kern).reshape(B, 2, -1)
    bus5 = torch.nn.functional.conv1d(bus.reshape(B, 1, -1), kern).reshape(B, -1)
    out = []
    for b in range(B):
        air = (fz[b].sum(0) < 1e-4).numpy()  # substeps with both soles off
        air[:skip] = False
        # flights = runs of air of >= 4 substeps (10 ms)
        flights = []
        i = 0
        while i < len(air):
            if air[i]:
                j = i
                while j < len(air) and air[j]:
                    j += 1
                if j - i >= 4:
                    flights.append((i, j))
                i = j
            else:
                i += 1
        r = dict(fell=bool(not alive[b]), flights=len(flights))
        if flights:
            i0, i1 = max(flights, key=lambda f: f[1] - f[0])
            r["flight_s"] = (i1 - i0) * dt_sub
            k0, k1 = i0 // H, min(i1 // H + 1, com_z.shape[1] - 1)
            z_to = float(com_z[b, max(k0 - 1, 0)])
            r["com_rise_cm"] = 100 * (float(com_z[b, k0 : k1 + 1].max()) - z_to)
            r["com_rise_stand_cm"] = 100 * (float(com_z[b, k0 : k1 + 1].max()) - float(com_stand[b]))
            r["foot_clear_cm"] = 100 * float(sole_sub[b, i0:i1].max())
            r["foot_clear_site_cm"] = 100 * float(sole[b, k0 : k1 + 1].max())  # previous measure (sole centres)
            r["pitch_change_flight_deg"] = float(pitch[b, k0 : k1 + 1].max() - pitch[b, k0 : k1 + 1].min())
            r["t_takeoff"] = i0 * dt_sub
            ld = slice(min(i1, fz5.shape[-1] - 1), min(i1 + round(0.15 / dt_sub), fz5.shape[-1]))
            r["land_mg_l"], r["land_mg_r"] = (float(fz5[b, 0, ld].max()), float(fz5[b, 1, ld].max()))
            first = fz[b, :, min(i1, fz.shape[-1] - 1)]
            r["first_touch"] = "both" if bool((first > 0).all()) else ("l" if first[0] > 0 else "r")
        r["takeoff_vz"] = float(com_vz[b].max())
        r["peak_foot_mg"] = float(fz5[b].max())
        r["bus_a"] = float(bus5[b].max())
        r["t_bus_peak"] = float(bus5[b].argmax()) * dt_sub  # s since the skill started (5 ms window start)
        if flights:
            r["t_bus_peak_rel_takeoff"] = r["t_bus_peak"] - r["t_takeoff"]
        r["max_w"] = sub["w"][b].tolist()
        r["max_tau"] = sub["tau"][b].tolist()
        r["max_util"] = float(sub["util"][b].max())
        r["final_pose_err_deg"] = float(final["pose_err_deg"][b])
        r["final_tilt_deg"] = float(final["tilt_deg"][b])
        out.append(r)
    env.close()
    return dict(play=play, envs=out, joint_names=list(robot.joint_names))


def summarise(res: dict) -> dict:
    e = res["envs"]
    hop = [r for r in e if r["flights"] > 0]

    def stat(key, rows=hop):
        v = [r[key] for r in rows if key in r]
        return None if not v else dict(mean=round(float(np.mean(v)), 3), min=round(float(np.min(v)), 3),
                                       max=round(float(np.max(v)), 3))

    jn = res["joint_names"]
    w = np.max([r["max_w"] for r in e], axis=0)
    tau = np.max([r["max_tau"] for r in e], axis=0)
    return dict(
        mode="nominal" if res["play"] else "randomised",
        envs=len(e),
        falls=sum(r["fell"] for r in e),
        hopped=len(hop),
        two_or_more_flights=sum(r["flights"] > 1 for r in e),
        com_rise_cm=stat("com_rise_cm"),
        com_rise_stand_cm=stat("com_rise_stand_cm"),
        takeoff_vz=stat("takeoff_vz", e),
        flight_s=stat("flight_s"),
        foot_clear_cm=stat("foot_clear_cm"),
        pitch_change_flight_deg=stat("pitch_change_flight_deg"),
        land_mg_per_foot=stat("land_mg_l") and dict(
            mean=round(float(np.mean([max(r["land_mg_l"], r["land_mg_r"]) for r in hop])), 2),
            max=round(float(np.max([max(r["land_mg_l"], r["land_mg_r"]) for r in hop])), 2)),
        first_touch={k: sum(r.get("first_touch") == k for r in hop) for k in ("both", "l", "r")},
        bus_a=stat("bus_a", e),
        max_envelope_use=stat("max_util", e),
        joint_peak_speed={n: round(float(x), 1) for n, x in zip(jn, w)},
        joint_peak_torque={n: round(float(x), 2) for n, x in zip(jn, tau)},
        final_pose_err_deg=stat("final_pose_err_deg", e),
        final_tilt_deg=stat("final_tilt_deg", e),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--json")
    ap.add_argument("--nominal-only", action="store_true")
    a = ap.parse_args()
    out = {}
    for play, n in ((True, 2), (False, a.envs))[: 1 if a.nominal_only else 2]:
        res = run(a.onnx, play, n)
        out["nominal" if play else "randomised"] = dict(summary=summarise(res), envs=res["envs"])
        print(json.dumps(out["nominal" if play else "randomised"]["summary"], indent=1))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
