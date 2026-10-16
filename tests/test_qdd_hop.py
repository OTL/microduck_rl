"""Mjlab-Skill-Hop-QddBiped: skill spec schema 2 (contact schedule, flight), the hop
spec, the torque-speed envelope actuator, the jump env and its policy_io (CPU).

Two hop specs: tests/fixtures/hop_keyframes.json (the first, keyframe-only hop, 50 Hz:
gate / interpolation tests) and skills/hop.json (the trained one: 100 Hz with the
feed-forward table skills/hop_ref.json, residual action)."""

import copy
import json
import math

import numpy as np
import pytest
import torch

from mjlab_microduck.tasks import mdp as microduck_mdp  # noqa: F401  (tasks package first)
from mjlab_microduck.tasks import qdd_skill
from mjlab_microduck.tasks.qdd_biped_skill_env_cfg import (
    JUMP_PHYSICS_DT,
    JUMP_TASK_ID,
    QddJumpRlCfg,
    build_skill_policy_io,
    make_qdd_biped_skill_env_cfg,
    skill_onnx_metadata,
)
from mjlab_microduck.robot.qdd_biped_constants import QDD_JOINT_NAMES
from pathlib import Path

KEYFRAME_HOP = Path(__file__).parent / "fixtures" / "hop_keyframes.json"


def _hop_raw() -> dict:
    """The keyframe-only hop (no table)."""
    return json.loads(KEYFRAME_HOP.read_text(encoding="utf-8"))


# --- spec ---------------------------------------------------------------------------


def test_hop_spec_is_valid():
    spec = qdd_skill.validate_spec(_hop_raw())
    assert spec["schema_version"] == 2 and spec["interpolation"] == "pchip"
    assert spec["policy_rate_hz"] == 50 and spec["policy_dt_s"] == pytest.approx(0.02)
    assert qdd_skill.flight_intervals(spec["contact"]) == [(1.39, 1.55)]
    fl = spec["flight"]
    assert fl["com_rise_m"] == 0.03 and fl["max_landing_load_mg"] == 4.0 and fl["toe_first"] is True
    assert "_table" not in spec
    assert qdd_skill.main(["hop"]) == 0 and qdd_skill.main([str(KEYFRAME_HOP)]) == 0


def test_feedforward_hop_spec_and_table():
    spec = qdd_skill.load_spec("hop")
    assert spec["policy_rate_hz"] == 100 and spec["reference_table"] == "hop_ref.json"
    tb = spec["_table"]
    assert tb["dt"] == pytest.approx(0.005) and tb["q"].shape[1] == 10
    (t0, t1), = qdd_skill.flight_intervals(spec["contact"])
    assert abs(tb["meta"]["t_takeoff"] - t0) < 0.01 and abs(tb["meta"]["t_touchdown"] - t1) < 0.01
    # residual ramp-in 0.1 -> 1 over the first 0.2 s (hand-over from the walking policy), 0.3 in the push
    rs = tb["residual_scale"]
    assert rs[0] == pytest.approx(0.1) and rs[int(round(0.1 / tb["dt"]))] == pytest.approx(0.55, abs=0.01)
    assert rs.max() == 1.0 and rs[int(round(0.25 / tb["dt"]))] == 1.0
    assert 0.25 <= rs[int(round(0.25 / tb["dt"])):].min() < 1.0
    # foot clearance plan and the hip-pitch residual factor (smaller near its -100 deg limit)
    assert 0.05 < tb["meta"]["foot_clear_m"] < 0.12 and t0 < tb["meta"]["t_foot_clear"] < t1
    rsj = tb["residual_scale_joints"]
    hip = [QDD_JOINT_NAMES.index(n) for n in ("l_hip_pitch", "r_hip_pitch")]
    other = [j for j in range(10) if j not in hip]
    assert np.all(rsj[:, other] == 1.0) and 0.1 <= rsj[:, hip].min() < 0.5
    near = tb["q"][:, hip[0]] < math.radians(-90)
    assert near.any() and np.all(rsj[near][:, hip] < 1.0)
    # the dense joint reference is the table's q (not the keyframe interpolation)
    ref = qdd_skill.build_reference(spec)
    k = int(round(1.25 / 0.01))
    assert np.allclose(ref["q"][k], qdd_skill.sample_table(tb, np.array([1.25]))[0])
    # push-off up to ~15-18 rad/s, crouch calm
    dq = np.abs(np.asarray(tb["dq"]))
    assert 12.0 < dq.max() <= qdd_skill.FAST_JOINT_SPEED_RAD_S + 0.01


def _table_raw() -> dict:
    return json.loads((qdd_skill.SKILLS_DIR / "hop_ref.json").read_text(encoding="utf-8"))


def _bad_table(fn):
    raw = _table_raw()
    fn(raw)
    return raw


BAD_TABLES = {
    "format": (_bad_table(lambda r: r.update(format="x")), "format must be"),
    "joint_order": (_bad_table(lambda r: r["joint_order"].reverse()), "joint_order must be"),
    "dt": (_bad_table(lambda r: r.update(dt=0.05)), "reference_table: dt"),
    "ragged": (_bad_table(lambda r: r["dq"].pop()), "different lengths"),
    "nan": (_bad_table(lambda r: r["q"][5].__setitem__(2, float("nan"))), "must be finite"),
    "end_tau": (_bad_table(lambda r: r["tau_ff"][-1].__setitem__(3, 1.0)), "last row must be the standing pose"),
    "start_pose": (_bad_table(lambda r: r["q"][0].__setitem__(3, 1.2)), "first row must be the standing pose"),
    "tau_big": (_bad_table(lambda r: r["tau_ff"][200].__setitem__(3, 9.5)), "exceeds 0.8 x the motor peak"),
    "residual": (_bad_table(lambda r: r["residual_scale"].__setitem__(10, 0.0)), "residual_scale must be in"),
    "takeoff": (_bad_table(lambda r: r.update(t_takeoff=r["t_takeoff"] + 0.1)), "reference_table.t_takeoff"),
    "short": (_bad_table(lambda r: [r[k].__delitem__(slice(300, None)) for k in ("q", "dq", "tau_ff", "residual_scale")]),
              "less than duration_s"),
    "rsj_range": (_bad_table(lambda r: r["residual_scale_joints"][10].__setitem__(2, 0.0)),
                  "residual_scale_joints: must be in"),
    "rsj_rows": (_bad_table(lambda r: r["residual_scale_joints"].pop()), "residual_scale_joints: must be finite"),
    "foot_clear": (_bad_table(lambda r: r.update(foot_clear_m=0.5)), "foot_clear_m"),
}


@pytest.mark.parametrize("case", sorted(BAD_TABLES))
def test_bad_reference_tables_are_rejected(case, monkeypatch):
    raw, needle = BAD_TABLES[case]
    monkeypatch.setattr(qdd_skill, "load_table_text", lambda name: json.dumps(raw))
    spec = json.loads((qdd_skill.SKILLS_DIR / "hop.json").read_text(encoding="utf-8"))
    with pytest.raises(qdd_skill.SkillSpecError) as e:
        qdd_skill.validate_spec(spec)
    assert any(needle in msg for msg in e.value.errors), (case, e.value.errors)


def test_reference_table_name_and_version():
    spec = json.loads((qdd_skill.SKILLS_DIR / "hop.json").read_text(encoding="utf-8"))
    for name in ("../hop_ref.json", "hop_ref.txt", "missing.json"):
        with pytest.raises(qdd_skill.SkillSpecError):
            qdd_skill.validate_spec({**spec, "reference_table": name})
    v1 = json.loads((qdd_skill.SKILLS_DIR / "bow.json").read_text(encoding="utf-8"))
    with pytest.raises(qdd_skill.SkillSpecError) as e:
        qdd_skill.validate_spec({**v1, "reference_table": "hop_ref.json"})
    assert any("needs schema_version 2" in m for m in e.value.errors)


def test_v1_spec_normalised_without_flight():
    spec = qdd_skill.load_spec("bow")
    assert spec["schema_version"] == 1
    assert spec["contact"] == [{"t": 0.0, "feet": "both"}] and spec["flight"] is None
    assert spec["policy_rate_hz"] == 50


def _mutate(fn):
    raw = _hop_raw()
    fn(raw)
    return raw


def _flight(**kw):
    return lambda r: r["flight"].update(kw)


BAD_HOP_SPECS = {
    "v1_with_contact": (_mutate(lambda r: r.update(schema_version=1)), "needs schema_version 2"),
    "no_flight_section": (_mutate(lambda r: r.pop("flight")), "flight: required"),
    "flight_without_none": (
        _mutate(lambda r: r.update(contact=[{"t": 0.0, "feet": "both"}])),
        "no 'none' (flight) phase",
    ),
    "single_foot": (_mutate(lambda r: r["contact"][1].update(feet="left")), "single-foot"),
    "ends_in_air": (_mutate(lambda r: r["contact"].pop()), "must end with both feet down"),
    "not_alternating": (
        _mutate(lambda r: r["contact"].insert(2, {"t": 1.45, "feet": "none"})),
        "must alternate",
    ),
    "too_long_flight": (
        _mutate(lambda r: r["contact"][2].update(t=1.75)),
        "controller faults after 0.3 s",
    ),
    "flight_vs_ballistic": (
        _mutate(lambda r: r["contact"][2].update(t=1.65)),
        "ballistic flight time",
    ),
    "too_high": (_mutate(_flight(com_rise_m=0.08)), "flight.com_rise_m"),
    "takeoff_speed_conflict": (
        _mutate(_flight(com_rise_m=0.04, max_takeoff_speed_mps=0.8)),
        "needs a take-off speed",
    ),
    "landing_limit": (_mutate(_flight(max_landing_load_mg=6.0)), "flight.max_landing_load_mg"),
    "unknown_flight_key": (_mutate(_flight(spin=True)), "flight.spin"),
    "policy_rate": (_mutate(lambda r: r.update(policy_rate_hz=200)), "policy_rate_hz"),
    "landing_too_late": (
        _mutate(lambda r: r.update(duration_s=1.9) or r["keyframes"].__setitem__(-1, {"t": 1.9})),
        "to return to the standing pose",
    ),
    "torso_check_in_flight": (
        _mutate(lambda r: r["keyframes"][7].update(torso={"pitch_deg": 20.0})),
        "no torso cross-check in a flight",
    ),
    "push_too_fast": (
        # the whole push in one 0.02 s step: knee 33 deg -> 29 rad/s
        _mutate(lambda r: r["keyframes"][6].update(t=1.322) or [r["keyframes"].pop(i) for i in (5, 4, 3)]),
        "rad/s",
    ),
    "slow_part_gap": (
        _mutate(lambda r: r["keyframes"][1].update(t=0.05)),
        "at least 0.2 s after",
    ),
    "crouch_too_deep": (
        _mutate(
            lambda r: [
                kf["joints_deg"].update(hip_pitch=-97.0, ankle_pitch=-41.0) or kf.pop("torso", None)
                for kf in r["keyframes"][1:3]
            ]
        ),
        "torso pitch reaches",
    ),
}


@pytest.mark.parametrize("case", sorted(BAD_HOP_SPECS))
def test_bad_hop_specs_are_rejected(case):
    raw, needle = BAD_HOP_SPECS[case]
    with pytest.raises(qdd_skill.SkillSpecError) as e:
        qdd_skill.validate_spec(copy.deepcopy(raw))
    assert any(needle in msg for msg in e.value.errors), (case, e.value.errors)


def test_pchip_is_monotone_and_continues_through_the_push():
    kf_t = np.array([0.0, 1.0, 1.1, 1.2, 2.0])
    kf_q = np.array([[0.0], [0.0], [0.5], [1.0], [1.0]])
    t = np.linspace(0, 2, 401)
    q = qdd_skill.sample_joint_reference(t, kf_t, kf_q, "pchip")[:, 0]
    assert np.all(np.diff(q) >= -1e-12) and q.min() >= 0 and q.max() <= 1.0 + 1e-12
    # velocity at the middle push keyframe is not zero (unlike min-jerk)
    k = int(np.argmin(np.abs(t - 1.1)))
    assert (q[k + 1] - q[k - 1]) / (t[k + 1] - t[k - 1]) > 4.0
    for kt, kq in zip(kf_t, kf_q[:, 0]):
        assert qdd_skill.sample_joint_reference(np.array([kt]), kf_t, kf_q, "pchip")[0, 0] == pytest.approx(kq)


def test_hop_reference_windows_and_ballistic_com():
    spec = qdd_skill.validate_spec(_hop_raw())
    ref = qdd_skill.build_reference(spec)
    t = ref["t"]
    t0, t1 = 1.39, 1.55
    assert np.array_equal(ref["in_air"], (t >= t0 - 1e-9) & (t < t1 - 1e-9))
    assert np.all(ref["air_window"][ref["in_air"]]) and ref["air_window"].sum() > ref["in_air"].sum()
    assert np.all(ref["fast_window"][ref["air_window"]])
    assert not np.any(ref["rsi_ok"][ref["fast_window"]])
    # apex of the CoM reference = take-off CoM + com_rise
    k_to = int(math.ceil(t0 / 0.02 - 1e-6)) - 1
    rise = ref["com_z_ref"][ref["in_air"]].max() - ref["com_z"][k_to]
    assert rise == pytest.approx(0.03, abs=0.002)
    # crouch: 35 deg forward, push-off joint speeds up to ~15 rad/s, calm elsewhere
    assert math.degrees(ref["pitch"].max()) == pytest.approx(35.0, abs=0.5)
    dq = np.abs(np.diff(ref["q"], axis=0)) / 0.02
    fast = ref["fast_window"][:-1] | ref["fast_window"][1:]
    assert 10.0 < dq[fast].max() <= qdd_skill.FAST_JOINT_SPEED_RAD_S
    assert dq[~fast].max() <= qdd_skill.MAX_JOINT_SPEED_RAD_S
    # toes down 20 deg at touchdown (ankle -10 vs the flat -30 at knee 60)
    k_td = int(round(t1 / 0.02))
    assert math.degrees(ref["q"][k_td, QDD_JOINT_NAMES.index("l_ankle_pitch")]) == pytest.approx(-10.0, abs=1.0)


# --- actuator envelope ----------------------------------------------------------------


def test_gim6010_envelope_matches_hop_py():
    from mjlab_microduck.actuator.qdd_motor import gim3510_motoring_limit, gim6010_motoring_limit

    v = torch.tensor(22.0)
    # docs/jump-feasibility.md: 22 V -> 7.6 / 5.6 / 4.1 / 2.0 Nm at 10 / 15 / 20 / 30 rad/s
    for w, tau in ((10, 7.6), (15, 5.6), (20, 4.1), (30, 2.0)):
        assert float(gim6010_motoring_limit(torch.tensor(float(w)), v)) == pytest.approx(tau, abs=0.06)
    assert float(gim6010_motoring_limit(torch.tensor(60.0), v)) == 0.0
    assert float(gim3510_motoring_limit(torch.tensor(0.0), torch.tensor(24.0))) == pytest.approx(6.28)
    assert float(gim3510_motoring_limit(torch.tensor(21.2), torch.tensor(24.0))) == pytest.approx(3.14, abs=0.01)


# --- env ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hop_env():
    from mjlab.envs import ManagerBasedRlEnv

    cfg = make_qdd_biped_skill_env_cfg("hop")
    cfg.scene.num_envs = 4
    cfg.commands["skill"].rsi_fraction = 0.0
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    yield env
    env.close()


def test_hop_env_physics_actuators_and_terms(hop_env):
    from mjlab_microduck.actuator.qdd_motor import QddMotorActuator

    env = hop_env
    assert env.cfg.sim.mujoco.timestep == JUMP_PHYSICS_DT and env.cfg.decimation == 4
    assert env.step_dt == pytest.approx(0.01)
    acts = env.scene["robot"].actuators
    assert len(acts) == 3 and all(isinstance(a, QddMotorActuator) for a in acts)
    terms = set(env.reward_manager.active_terms)
    assert {"flight_height", "takeoff_speed", "flight_rotation", "landing_load", "toe_first"} <= terms
    w = {n: env.reward_manager.get_term_cfg(n).weight for n in terms}
    for cost in ("takeoff_speed", "flight_rotation", "landing_load", "joint_power", "action_rate_l2", "feet_flat"):
        assert w[cost] < 0, cost
    for rew in ("flight_height", "feet_contact", "toe_first", "joint_tracking"):
        assert w[rew] > 0, rew
    fh = env.scene.sensors["feet_ground_contact"].data.force_history
    assert fh is not None and fh.shape[2] == 4
    assert w["residual_l2"] < 0 and w["bus_current"] < 0 and w["termination"] == pytest.approx(-50.0)
    # the soles are the height objective (foot clearance), the CoM height tracking stays smaller
    assert w["foot_clearance"] == pytest.approx(5.0) and w["flight_height"] == pytest.approx(2.0)
    assert env.reward_manager.get_term_cfg("foot_clearance").params["target_m"] == pytest.approx(
        env.command_manager.get_term("skill").spec["_table"]["meta"]["foot_clear_m"])


def test_envelope_clips_motoring_torque(hop_env):
    env = hop_env
    knee = env.scene["robot"].actuators[1]
    vel = torch.full((4, 4), 15.0)
    eff = torch.full((4, 4), 11.0)
    lo, hi = knee.available(eff, vel)
    assert torch.allclose(hi, torch.tensor(5.56), atol=0.05)  # motoring at 15 rad/s, 22 V
    assert torch.allclose(lo, torch.tensor(-11.0))  # braking: driver clamp
    knee.set_envelope(torch.tensor([0]), v_bus=torch.tensor([19.5]), strength=torch.tensor([0.9]))
    lo, hi = knee.available(eff, vel)
    assert hi[0, 0] < 0.9 * 5.56 and hi[1, 0] == pytest.approx(5.56, abs=0.05)
    knee.set_envelope(torch.tensor([0]), v_bus=torch.tensor([22.0]), strength=torch.tensor([1.0]))


def test_hop_env_steps_and_rewards_are_finite(hop_env):
    env = hop_env
    env.reset()
    act = torch.zeros(4, 10)
    for _ in range(10):
        _, rew, term, _, _ = env.step(act)
        assert torch.all(torch.isfinite(rew)) and not torch.any(term)
    # standing still: per-foot load ~0.5 m g, no landing cost, no take-off cost
    assert float(microduck_mdp.skill_landing_load_cost(env, "feet_ground_contact", 4.0).max()) == 0.0
    assert float(microduck_mdp.skill_takeoff_speed_cost(env, 0.85).max()) == 0.0
    assert float(microduck_mdp.skill_bus_current_cost(env, 20.0).max()) == 0.0
    # soles flat on the floor: lowest sole point ~0, no foot-clearance reward on the ground
    robot = env.scene["robot"]
    sites = robot.find_sites(["l_foot_site", "r_foot_site"], preserve_order=True)[0]
    z = microduck_mdp.sole_lowest_z(robot, sites, qdd_skill.FOOT_HALF_LENGTH_M, qdd_skill.FOOT_HALF_WIDTH_M)
    assert torch.all(z.abs() < 0.006), z
    assert float(microduck_mdp.skill_foot_clearance(env, "feet_ground_contact", 0.07).abs().max()) == 0.0
    # schedule: both feet down before the take-off pays 1
    assert torch.allclose(microduck_mdp.skill_contact_schedule(env, "feet_ground_contact"), torch.ones(4))


def test_hop_rsi_centred_on_the_crouch(hop_env):
    env = hop_env
    sk = env.command_manager.get_term("skill")
    # training cfg of the feed-forward hop: no RSI inside the jump, most in the crouch before the push
    assert env.cfg.commands["skill"].rsi_jump_fraction == 0.0
    lo, hi = env.cfg.commands["skill"].rsi_focus_window
    assert 0.5 < lo < hi < qdd_skill.flight_intervals(sk.spec["contact"])[0][0] - 0.05
    fast = sk.ref_fast_window.cpu().numpy()
    sk.cfg.rsi_fraction = 1.0
    try:
        for foc in (0.0, 1.0):
            sk.cfg.rsi_focus_fraction = foc
            env.reset()
            k0 = sk.k0.cpu().numpy()
            assert np.all(k0 > 0) and not np.any(fast[k0])
            if foc:
                t = k0 * env.step_dt
                assert np.all((t >= lo - 1e-6) & (t <= hi + 1e-6)), t
    finally:
        sk.cfg.rsi_fraction = 0.0
        sk.cfg.rsi_focus_fraction = env.cfg.commands["skill"].rsi_focus_fraction


def test_reference_action_plays_the_table(hop_env):
    env = hop_env
    robot = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    assert isinstance(term, microduck_mdp.SkillReferenceJointPositionAction)
    sk = env.command_manager.get_term("skill")
    tb = sk.spec["_table"]
    ids = robot.find_joints(QDD_JOINT_NAMES, preserve_order=True)[0]
    env.reset()
    gain = term.residual_gain
    for a in (0.0, 1.0):
        env.step(torch.full((4, 10), a))
        # the last substep of step k = 0 / 1 was at t = k * dt + 3 * physics_dt
        k = int(sk.step_index[0]) - 1
        t = k * env.step_dt + (env.cfg.decimation - 1) * env.physics_dt
        bias = robot.data.encoder_bias[0, ids]
        want_q = qdd_skill.sample_table(tb, np.array([t]))[0]
        res = qdd_skill.sample_table(tb, np.array([t]), "residual_scale")[0]
        rsj = qdd_skill.sample_table({"dt": tb["dt"], "x": tb["residual_scale_joints"]}, np.array([t]), "x")[0]
        want_q = want_q + a * 0.25 * res * rsj * gain
        got = (robot.data.joint_pos_target[0, ids] + bias).numpy()
        assert np.allclose(got, want_q, atol=1e-5), (a, got - want_q)
        assert np.allclose(robot.data.joint_vel_target[0, ids].numpy(), qdd_skill.sample_table(tb, np.array([t]), "dq")[0], atol=1e-5)
        assert np.allclose(robot.data.joint_effort_target[0, ids].numpy(), qdd_skill.sample_table(tb, np.array([t]), "tau_ff")[0], atol=1e-5)


def test_bus_current_is_the_largest_over_the_substeps(hop_env):
    env = hop_env
    term = env.action_manager.get_term("joint_pos")
    robot = env.scene["robot"]
    env.reset()
    seen = []
    orig = env.sim.step

    def hook():
        orig()
        seen.append(microduck_mdp.qdd_bus_current(env, robot))

    env.sim.step = hook
    try:
        env.step(torch.full((4, 10), 0.5))
    finally:
        env.sim.step = orig
    assert len(seen) == env.cfg.decimation
    # the action term saw substeps 1..3 (it samples before each physics step); the cost adds the last
    want = torch.stack(seen[:-1], 1).amax(1)
    assert torch.allclose(term.bus_max, want, atol=1e-4)
    cost = microduck_mdp.skill_bus_current_cost(env, 0.0)
    assert torch.allclose(cost, torch.stack(seen, 1).amax(1).clamp(min=0.0) ** 2, rtol=1e-4, atol=1e-4)


# --- policy_io / cfg ----------------------------------------------------------------------


def test_hop_policy_io():
    spec = qdd_skill.load_spec("hop")
    io = build_skill_policy_io(spec)
    assert io["task"] == JUMP_TASK_ID and io["kind"] == "skill"
    sk = io["skill"]
    total = spec["duration_s"] + spec["settle_s"]
    assert sk["n_steps"] == round(total / 0.01) and sk["policy_dt_s"] == pytest.approx(0.01)
    assert sk["contact_schedule"] == spec["contact"] and sk["contact_schedule"][1]["feet"] == "none"
    assert sk["flight"]["com_rise_m"] == spec["flight"]["com_rise_m"] and sk["max_flight_s"] == 0.3
    assert sk["max_torso_pitch_deg"] == 40.0
    assert "torque_speed_envelope" in io["actuators"]
    # the feed-forward table for the controller
    r = sk["reference"]
    assert r["format"] == qdd_skill.TABLE_FORMAT and r["joint_order"] == list(QDD_JOINT_NAMES)
    n = len(r["q"])
    assert len(r["dq"]) == len(r["tau_ff"]) == len(r["residual_scale"]) == n and (n - 1) * r["dt"] >= total - 1e-6
    stand = qdd_skill.stand_pose()
    assert np.allclose(r["q"][0], stand, atol=1e-3) and np.allclose(r["q"][-1], stand, atol=1e-3)
    assert "reference" in io["action"]["target"] and "residual_scale_joints" in io["action"]["target"]
    assert len(r["residual_scale_joints"]) == n and r["foot_clear_m"] == spec["_table"]["meta"]["foot_clear_m"]
    md = skill_onnx_metadata(spec)
    assert json.loads(md["policy_io"])["skill"]["name"] == "hop"
    # the keyframe-only hop has no reference
    kf = qdd_skill.validate_spec(_hop_raw())
    assert "reference" not in build_skill_policy_io({**kf, "_sha256": "x", "_text": "{}"})["skill"]


def test_policy_rate_100hz_cfg():
    raw = _hop_raw()
    raw["policy_rate_hz"] = 100
    spec = qdd_skill.validate_spec(raw)
    assert spec["policy_dt_s"] == pytest.approx(0.01)
    ref = qdd_skill.build_reference(spec)
    assert len(ref["t"]) == 401
    io = build_skill_policy_io({**spec, "_sha256": "x", "_text": "{}"})
    assert io["policy_dt_s"] == pytest.approx(0.01) and io["skill"]["n_steps"] == 400
    assert io["gait_clock"]["initial_phase"] == pytest.approx(1.0 - 0.01 / 8.0)


def test_jump_rl_cfg():
    assert QddJumpRlCfg.experiment_name == "qdd_biped_skill_hop"
    assert QddJumpRlCfg.algorithm.symmetry_cfg is None
