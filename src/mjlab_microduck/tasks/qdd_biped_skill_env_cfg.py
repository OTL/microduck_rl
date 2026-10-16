"""QDD biped skill (keyframe tracking) environment -- Mjlab-Skill-QddBiped.

Trains a policy for ONE skill described by a JSON spec (tasks/qdd_skill.py,
robot/qdd_biped/skills/*.json; default bow.json, override with
$QDD_SKILL_SPEC=<path or name> before the task package is imported).

Template = the walking task (qdd_biped_velocity_env_cfg.py) with the same robot,
actuators, domain randomisation, observation noise / delays, terminations and the
same 41-D actor observation, so the Rust controller runs a skill policy exactly
like the walking one:
  - command slot [36:39]  : zeros (ZeroTwistCommand),
  - gait_phase slot [39:41]: skill phase [sin(pi s), cos(pi s)], s = t / total in
    [0, 1] (t = time since the skill started, total = duration + settle). The
    policy_io.json for a skill (build_skill_policy_io) describes it as a gait
    clock with period 2 * total that always runs, so an unchanged controller
    feeds it with any command (see the "skill" section there).
Episode = one skill: start from the standing pose (or, for 30 % of the training
resets, from a random point of the reference = reference-state initialisation),
track the keyframe reference, end with the return-to-stand phase; the episode
times out at t = total.

Rewards (weights from the spec, bounded by qdd_skill.REWARD_WEIGHT_BOUNDS):
joint tracking, torso pitch / roll and height tracking (reference from flat-sole
kinematics), balance (no drift / spin), both soles on the floor, no foot slip,
final standing pose; costs: action rate, torques. Fixed by the template (not
settable from a spec): joint-limit / knee-limit / self-collision / flat-feet
costs, termination penalty, terminations (tilt > 45 deg, torso < 0.20 m, body
contact, knee limit, NaN).

Play mode (export / video): nominal robot as in the walking play mode, always
from the standing pose, episodes repeat the skill.

Jumps (spec schema 2 with a flight phase, e.g. skills/hop.json; task
Mjlab-Skill-Hop-QddBiped): physics at 2.5 ms, actuators = explicit PD with the
motors' torque-speed envelope (actuator/qdd_motor.py, robot repo analysis/hop.py
tmax), the contact schedule replaces "both soles down", and: CoM height tracking
(push-off + ballistic flight to the spec's CoM rise), take-off speed limit, no
torso rotation in flight, landing load per foot (peak over the physics substeps,
strong above the spec's limit), toe-first touchdown, motoring power; torso
orientation / height references and the flat-feet cost are off in the flight /
landing window, joint tracking is looser and the action-rate cost smaller from the
push-off to the landing; RSI only outside that window. Extra DR: sole-floor contact
stiffness / damping (floor, sole pad), motor bus voltage and strength.
"""

from __future__ import annotations

import math
import os
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import rewards as mj_rewards
from mjlab.managers import ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg

from mjlab_microduck.robot.qdd_biped_constants import FOOT_SITES
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import qdd_skill
from mjlab_microduck.tasks.qdd_biped_velocity_env_cfg import (
    QddBipedRlCfg,
    make_qdd_biped_velocity_env_cfg,
)

TASK_ID = "Mjlab-Skill-QddBiped"

# Kernels (fixed by the template)
JOINT_TRACKING_STD = 0.15  # rad, rms over the 10 joints
TORSO_ORIENTATION_STD = 0.10  # rad (~6 deg)
TORSO_HEIGHT_STD = 0.02  # m
BALANCE_LIN_STD = 0.10  # m/s CoM drift
BALANCE_YAW_STD = 0.5  # rad/s
FINAL_STAND_STD = 0.15  # rad
FINAL_STAND_VEL_STD = 1.0  # rad/s

# Fixed safety / quality costs
KNEE_LIMIT_WEIGHT = -10.0
SELF_COLLISION_WEIGHT = -1.0
JOINT_LIMIT_WEIGHT = -1.0
FEET_FLAT_WEIGHT = -2.0
TERMINATION_WEIGHT = -100.0  # x dt = -2 per fall (a whole episode pays ~+30)
FALL_TILT_DEG = 45.0  # termination (walking: 60); controller hard limit 50, skills <= 30

RSI_FRACTION = 0.3

# --- Jumps (spec with a flight phase) -------------------------------------------------
JUMP_TASK_ID = "Mjlab-Skill-Hop-QddBiped"
JUMP_SPEC = "hop"  # skills/hop.json (yatai cannot pass $QDD_SKILL_SPEC: own task id)
JUMP_PHYSICS_DT = 0.0025  # s (explicit PD of the ankle is not stable at 5 ms)
JUMP_ACTUATOR_MAX_LAG_S = 0.01  # command latency 0-10 ms (as walking)
JUMP_PLAY_LAG_S = 0.005
FLIGHT_HEIGHT_STD = 0.015  # m, CoM height tracking in the push-off + flight window
FLIGHT_VZ_STD = 0.3  # m/s, CoM vertical speed tracking in the same window
CONTACT_AIR_GAIN = 3.0  # airborne steps of the contact schedule pay this much
RSI_JUMP_FRACTION = 0.5  # of the RSI starts (30 %): inside the push-off / flight
FAST_TRACKING_STD_SCALE = 2.0  # joint tracking std x this from push-off to landing
FAST_ACTION_RATE_SCALE = 0.1  # action-rate cost x this from push-off to landing
LANDING_SOFT_FRACTION = 0.6  # landing cost starts at 0.6 x the spec limit (quadratic)
LANDING_HARD_GAIN = 10.0  # ... and is 10x steeper above the limit
TOE_ANGLE_SCALE_RAD = 0.17  # 10 deg toe-down at touchdown = full toe-first reward
FOOT_SOLREF_TIMECONST_RANGE = (0.006, 0.03)  # s: hard floor / thin rubber .. soft pad
FOOT_SOLREF_DAMPRATIO_RANGE = (0.7, 1.2)
MOTOR_V_BUS_RANGE = (19.5, 24.0)  # V: low battery under load .. full pack
MOTOR_STRENGTH_RANGE = (0.9, 1.1)  # torque constant / driver current calibration
PUSH_RANGE = (-0.15, 0.15)  # m/s (walking: 0.3; a planted-feet skill should not need a step)
SETTLE_EXTRA_S = 1.0  # episode_length_s = total + this (skill_done ends it at total)

# Warm start: initialise the actor (MLP + obs normaliser) from a walking ONNX with
# the same 41-D obs / 10-D action layout. $QDD_SKILL_INIT_ONNX = path, or "none".
WARM_START_ONNX_ENV = "QDD_SKILL_INIT_ONNX"
DEFAULT_WARM_START_ONNX = qdd_skill.SKILLS_DIR / "warm_start" / "walk_gait2.onnx"
WARM_START_INIT_STD = 0.4
SCRATCH_INIT_STD = 0.6


def warm_start_onnx_path():
    v = os.environ.get(WARM_START_ONNX_ENV, "")
    if v.lower() in ("none", "0", "off"):
        return None
    from pathlib import Path

    p = Path(v) if v else DEFAULT_WARM_START_ONNX
    return p if p.exists() else None


def make_qdd_biped_skill_env_cfg(spec_path=None, play: bool = False) -> ManagerBasedRlEnvCfg:
    spec = qdd_skill.load_spec(spec_path)
    w = spec["reward_weights"]
    total = qdd_skill.total_s(spec)
    jump = spec.get("flight") is not None

    cfg = make_qdd_biped_velocity_env_cfg(play=play)
    cfg.episode_length_s = total + SETTLE_EXTRA_S
    if jump:
        _make_jump_physics(cfg, spec, play)
    elif abs(spec["policy_dt_s"] - cfg.sim.mujoco.timestep * cfg.decimation) > 1e-9:
        cfg.decimation = int(round(spec["policy_dt_s"] / cfg.sim.mujoco.timestep))

    # --- Commands: zero twist (obs slot kept) + skill clock / reference ----------
    cfg.commands = {
        "twist": microduck_mdp.ZeroTwistCommandCfg(),
        "skill": microduck_mdp.SkillClockCommandCfg(
            spec_text=spec["_text"],
            spec_source=spec["_source"],
            rsi_fraction=0.0 if play else RSI_FRACTION,
        ),
    }

    # --- Observations: same 41-D layout; gait_phase slot = skill phase -------------
    for group in ("actor", "critic"):
        cfg.observations[group].terms["gait_phase"] = ObservationTermCfg(
            func=microduck_mdp.skill_phase_obs, params={"command_name": "skill"}
        )

    # --- Rewards ---------------------------------------------------------------------
    old = cfg.rewards
    feet = SceneEntityCfg("robot", site_names=FOOT_SITES)
    cfg.rewards = {
        "joint_tracking": RewardTermCfg(
            func=microduck_mdp.skill_joint_tracking,
            weight=w["joint_tracking"],
            params={"std": JOINT_TRACKING_STD, "command_name": "skill"},
        ),
        "torso_orientation": RewardTermCfg(
            func=microduck_mdp.skill_torso_orientation,
            weight=w["torso_orientation"],
            params={"std": TORSO_ORIENTATION_STD, "command_name": "skill"},
        ),
        "torso_height": RewardTermCfg(
            func=microduck_mdp.skill_torso_height,
            weight=w["torso_height"],
            params={"std": TORSO_HEIGHT_STD, "command_name": "skill"},
        ),
        "balance": RewardTermCfg(
            func=microduck_mdp.skill_balance,
            weight=w["balance"],
            params={"lin_std": BALANCE_LIN_STD, "yaw_std": BALANCE_YAW_STD},
        ),
        "feet_contact": RewardTermCfg(
            func=microduck_mdp.skill_feet_contact,
            weight=w["feet_contact"],
            params={"sensor_name": "feet_ground_contact"},
        ),
        "final_stand": RewardTermCfg(
            func=microduck_mdp.skill_final_stand,
            weight=w["final_stand"],
            params={"std": FINAL_STAND_STD, "vel_std": FINAL_STAND_VEL_STD, "command_name": "skill"},
        ),
        "foot_slip": RewardTermCfg(
            func=microduck_mdp.skill_foot_slip,
            weight=-w["foot_slip"],
            params={"sensor_name": "feet_ground_contact", "asset_cfg": feet},
        ),
        "action_rate_l2": RewardTermCfg(func=mj_rewards.action_rate_l2, weight=-w["action_rate"]),
        "joint_torques": RewardTermCfg(func=mj_rewards.joint_torques_l2, weight=-w["joint_torques"]),
        # --- fixed by the template
        "dof_pos_limits": replace(old["dof_pos_limits"], weight=JOINT_LIMIT_WEIGHT),
        "knee_limit": replace(old["knee_limit"], weight=KNEE_LIMIT_WEIGHT),
        "self_collisions": replace(old["self_collisions"], weight=SELF_COLLISION_WEIGHT),
        "feet_flat": replace(old["feet_flat"], weight=FEET_FLAT_WEIGHT),
        "termination": RewardTermCfg(func=mj_rewards.is_terminated, weight=TERMINATION_WEIGHT),
    }

    if jump:
        _add_jump_rewards(cfg, spec)
        if not play:
            cfg.commands["skill"].rsi_jump_fraction = RSI_JUMP_FRACTION

    # --- Terminations ------------------------------------------------------------------
    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(FALL_TILT_DEG)
    cfg.terminations["skill_done"] = TerminationTermCfg(
        func=microduck_mdp.skill_done, time_out=True, params={"command_name": "skill"}
    )

    # --- Events / curricula -------------------------------------------------------------
    if "push_robot" in cfg.events:
        cfg.events["push_robot"].params["velocity_range"] = {"x": PUSH_RANGE, "y": PUSH_RANGE}
    if jump and not play:
        _add_jump_events(cfg)
    cfg.curriculum = {}
    return cfg


def _make_jump_physics(cfg: ManagerBasedRlEnvCfg, spec: dict, play: bool) -> None:
    """2.5 ms physics, envelope actuators, per-substep foot forces."""
    from dataclasses import replace as dc_replace

    from mjlab_microduck.robot.qdd_biped_constants import qdd_envelope_actuators

    dt = JUMP_PHYSICS_DT
    cfg.sim.mujoco.timestep = dt
    cfg.decimation = int(round(spec["policy_dt_s"] / dt))
    assert abs(cfg.decimation * dt - spec["policy_dt_s"]) < 1e-9
    robot = cfg.scene.entities["robot"]
    lag = int(round(JUMP_ACTUATOR_MAX_LAG_S / dt))
    acts = qdd_envelope_actuators(delay_max_lag=lag)
    if play:
        pl = int(round(JUMP_PLAY_LAG_S / dt))
        acts = tuple(dc_replace(a, delay_min_lag=pl, delay_max_lag=pl) for a in acts)
    robot.articulation = dc_replace(robot.articulation, actuators=acts)
    cfg.scene.sensors = tuple(
        dc_replace(sn, history_length=cfg.decimation) if sn.name == "feet_ground_contact" else sn
        for sn in cfg.scene.sensors
    )


def _add_jump_rewards(cfg: ManagerBasedRlEnvCfg, spec: dict) -> None:
    w = spec["reward_weights"]
    fl = spec["flight"]
    feet = SceneEntityCfg("robot", site_names=FOOT_SITES)
    r = cfg.rewards
    r["joint_tracking"].params["fast_std_scale"] = FAST_TRACKING_STD_SCALE
    r["feet_contact"] = RewardTermCfg(
        func=microduck_mdp.skill_contact_schedule,
        weight=w["feet_contact"],
        params={"sensor_name": "feet_ground_contact", "command_name": "skill", "air_gain": CONTACT_AIR_GAIN},
    )
    r["flight_height"] = RewardTermCfg(
        func=microduck_mdp.skill_flight_height,
        weight=w["flight_height"],
        params={"std": FLIGHT_HEIGHT_STD, "vel_std": FLIGHT_VZ_STD, "command_name": "skill"},
    )
    r["takeoff_speed"] = RewardTermCfg(
        func=microduck_mdp.skill_takeoff_speed_cost,
        weight=-w["takeoff_speed"],
        params={"max_speed": fl["max_takeoff_speed_mps"]},
    )
    r["flight_rotation"] = RewardTermCfg(
        func=microduck_mdp.skill_flight_rotation_cost,
        weight=-w["flight_rotation"],
        params={"sensor_name": "feet_ground_contact", "command_name": "skill"},
    )
    r["landing_load"] = RewardTermCfg(
        func=microduck_mdp.skill_landing_load_cost,
        weight=-w["landing_load"],
        params={
            "sensor_name": "feet_ground_contact",
            "limit_mg": fl["max_landing_load_mg"],
            "soft_fraction": LANDING_SOFT_FRACTION,
            "hard_gain": LANDING_HARD_GAIN,
        },
    )
    if fl.get("toe_first", True) and w["toe_first"] > 0:
        r["toe_first"] = RewardTermCfg(
            func=microduck_mdp.skill_toe_first,
            weight=w["toe_first"],
            params={
                "sensor_name": "feet_ground_contact",
                "angle_scale_rad": TOE_ANGLE_SCALE_RAD,
                "command_name": "skill",
                "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES),
            },
        )
    if w["joint_power"] > 0:
        r["joint_power"] = RewardTermCfg(func=microduck_mdp.skill_joint_power_cost, weight=-w["joint_power"])
    r["action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.skill_action_rate,
        weight=-w["action_rate"],
        params={"fast_scale": FAST_ACTION_RATE_SCALE, "command_name": "skill"},
    )
    r["feet_flat"] = RewardTermCfg(
        func=microduck_mdp.skill_feet_flat,
        weight=FEET_FLAT_WEIGHT,
        params={"sensor_name": "feet_ground_contact", "height_threshold": 0.02, "asset_cfg": feet},
    )


def _add_jump_events(cfg: ManagerBasedRlEnvCfg) -> None:
    from mjlab.managers import EventTermCfg

    from mjlab_microduck.robot.qdd_biped_constants import FOOT_GEOMS

    cfg.events.pop("randomize_effort_limits", None)  # the driver clamp is fixed; the envelope varies
    cfg.events["randomize_motor_envelope"] = EventTermCfg(
        func=microduck_mdp.randomize_qdd_motor_envelope,
        mode="reset",
        params={"v_bus_range": MOTOR_V_BUS_RANGE, "strength_range": MOTOR_STRENGTH_RANGE},
    )
    cfg.events["foot_contact_solref"] = EventTermCfg(
        func=microduck_mdp.randomize_geom_solref,
        mode="reset",
        params={
            "timeconst_range": FOOT_SOLREF_TIMECONST_RANGE,
            "dampratio_range": FOOT_SOLREF_DAMPRATIO_RANGE,
            "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS),
        },
    )


def _skill_rl_cfg():
    from dataclasses import replace as dc_replace

    warm = warm_start_onnx_path() is not None
    actor = dc_replace(
        QddBipedRlCfg.actor,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": WARM_START_INIT_STD if warm else SCRATCH_INIT_STD,
            "std_type": "scalar",
        },
    )
    return dc_replace(
        QddBipedRlCfg,
        actor=actor,
        critic=RslRlModelCfg(**vars(QddBipedRlCfg.critic)),
        # A skill is not left-right symmetric in general and its phase is not a
        # per-leg clock: no mirror loss.
        algorithm=dc_replace(QddBipedRlCfg.algorithm, symmetry_cfg=None),
        experiment_name="qdd_biped_skill",
        run_name="qdd_biped_skill",
        save_interval=100,
        max_iterations=600,
    )


QddSkillRlCfg = _skill_rl_cfg()


def _jump_rl_cfg():
    from dataclasses import replace as dc_replace

    return dc_replace(
        QddSkillRlCfg,
        experiment_name="qdd_biped_skill_hop",
        run_name="qdd_biped_skill_hop",
        max_iterations=800,
    )


QddJumpRlCfg = _jump_rl_cfg()


# =============================================================================
# Policy I/O contract for a skill policy (embedded in its ONNX)
# =============================================================================
def build_skill_policy_io(spec: dict) -> dict:
    """The walking contract (same obs / action / gains), with the command slot fixed
    to zeros and the gait clock replaced by the skill clock (as a gait clock with
    period 2 * total that always runs, so the existing controller can run it)."""
    from mjlab_microduck.tasks.qdd_biped_velocity_env_cfg import build_policy_io

    io = build_policy_io()
    total = qdd_skill.total_s(spec)
    dt = spec.get("policy_dt_s", io["policy_dt_s"])
    io["policy_dt_s"] = dt
    io["control_rate_hz"] = round(1.0 / dt, 6)
    flight = spec.get("flight")
    io["task"] = JUMP_TASK_ID if flight is not None else TASK_ID
    io["kind"] = "skill"
    if flight is not None:
        io["actuators"]["torque_speed_envelope"] = (
            "trained with the motors' torque-speed envelope (GIM6010-8: FOC voltage limit at 19.5-24 V, "
            "no field weakening; GIM3510-8: linear to 0 at 42.4 rad/s x V / 24) inside +/- effort_limit; "
            "the drivers' MIT torque range must be at least effort_limit (GDK468: 11 Nm)"
        )
    for t in io["observation"]["terms"]:
        if t["name"] == "command":
            t["description"] = "always [0, 0, 0] for a skill policy (slot kept from the walking layout)"
        elif t["name"] == "gait_phase":
            t["description"] = (
                "skill phase: [sin(pi s), cos(pi s)], s = clamp(k / n, 0, 1), k = policy steps since the "
                "skill started (0 at the first observation), n = round(skill.total_s / policy_dt_s); "
                "= the gait_clock below"
            )
    io["skill"] = {
        "name": spec["name"],
        "description": spec.get("description", ""),
        "duration_s": spec["duration_s"],
        "settle_s": spec["settle_s"],
        "total_s": total,
        "n_steps": int(round(total / dt)),
        "support": spec["support"],
        "spec_sha256": spec["_sha256"],
        "start": "standing pose (action.default_joint_pos), torso upright, at rest; switch from the "
        "walking policy only while standing still (command 0, gait clock paused)",
        "end": "after total_s the robot is back in the standing pose: switch back to the walking "
        "policy (reset its last_action to 0 and its gait clock to initial_phase)",
        "phase": "s = k / n_steps, k = 0 at the first observation of the skill; the policy was "
        "trained for s in [0, 1] only: stop the skill at k = n_steps",
        "last_action": "zeros at the start of the skill (like any policy start)",
        "max_torso_pitch_deg": (
            qdd_skill.FLIGHT_TORSO_PITCH_RANGE_DEG if flight is not None else qdd_skill.TORSO_PITCH_RANGE_DEG
        )[1],
        "fall_termination_tilt_deg": FALL_TILT_DEG,
        "policy_dt_s": dt,
    }
    if flight is not None:
        io["skill"]["contact_schedule"] = [dict(c) for c in spec["contact"]]
        io["skill"]["flight"] = dict(flight)
        io["skill"]["max_flight_s"] = qdd_skill.FLIGHT_TIME_RANGE_S[1]
        io["skill"]["contact_note"] = (
            "contact_schedule: t [s] since the skill started -> feet 'both' (both soles on the floor) / "
            "'none' (flight); the policy may take off / land up to "
            f"{qdd_skill.CONTACT_TOL_S} s early or late. The controller allows flight only inside "
            "a scheduled flight (+ tolerance) and faults after max_flight_s airborne."
        )
    io["gait_clock"] = {
        "period_s": 2.0 * total,
        "swing_fraction": 0.0,
        "active_threshold": -1.0,
        "active_rule": "always active (threshold < 0): the skill clock runs whatever the command",
        "update_rule": io["gait_clock"]["update_rule"],
        "initial_phase": 1.0 - dt / (2.0 * total),
        "observation": "gait_phase = [sin(2 pi phi), cos(2 pi phi)] = [sin(pi s), cos(pi s)] with "
        "phi = s / 2; initial_phase makes the first observation phi = 0",
        "schedule": "none (skill)",
    }
    io["commands"] = {
        "lin_vel_x_range": [0.0, 0.0],
        "lin_vel_y_range": [0.0, 0.0],
        "ang_vel_z_range": [0.0, 0.0],
        "standing": "skill policy: the command is ignored (always [0, 0, 0])",
    }
    return io


def skill_policy_io_json(spec: dict) -> str:
    import json

    return json.dumps(build_skill_policy_io(spec), indent=2, ensure_ascii=False) + "\n"


def skill_onnx_metadata(spec: dict | None = None) -> dict[str, str]:
    """ONNX metadata_props for a skill policy: its policy_io (+ sha256) and the
    skill spec it was trained on."""
    import hashlib

    if spec is None:
        spec = qdd_skill.load_spec()
    text = skill_policy_io_json(spec)
    return {
        "policy_io": text,
        "policy_io_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "skill_spec": spec["_text"],
    }


if __name__ == "__main__":
    import sys

    print(skill_policy_io_json(qdd_skill.load_spec(sys.argv[1] if len(sys.argv) > 1 else None)))
