"""Mjlab-Skill-QddBiped: skill spec validation, reference, env, policy_io, warm start (CPU)."""

import copy
import json
import math

import numpy as np
import pytest
import torch

# Import the tasks package first (as mjlab's plugin loader does).
from mjlab_microduck.tasks import mdp as microduck_mdp  # noqa: F401
from mjlab_microduck.tasks import qdd_skill
from mjlab_microduck.tasks.qdd_biped_skill_env_cfg import (
    DEFAULT_WARM_START_ONNX,
    QddSkillRlCfg,
    build_skill_policy_io,
    make_qdd_biped_skill_env_cfg,
    skill_onnx_metadata,
)
from mjlab_microduck.tasks.qdd_biped_velocity_env_cfg import build_policy_io
from mjlab_microduck.robot.qdd_biped_constants import QDD_JOINT_NAMES, STAND_BASE_Z


def _bow_raw() -> dict:
    return json.loads(qdd_skill.DEFAULT_SKILL_JSON.read_text(encoding="utf-8"))


# --- spec validation ---------------------------------------------------------------


def test_bow_spec_is_valid():
    spec = qdd_skill.load_spec("bow")
    assert spec["name"] == "bow"
    assert spec["duration_s"] == 3.0 and spec["settle_s"] == 1.0
    assert set(spec["reward_weights"]) == set(qdd_skill.REWARD_WEIGHT_BOUNDS)
    assert spec["keyframes"][1]["_q_rad"][QDD_JOINT_NAMES.index("r_hip_pitch")] == pytest.approx(
        math.radians(-55.0)
    )
    assert qdd_skill.main(["bow"]) == 0


def test_schema_file_matches_validator():
    schema = json.loads(qdd_skill.SKILL_SCHEMA_JSON.read_text(encoding="utf-8"))
    assert set(schema["properties"]) == qdd_skill.TOP_KEYS
    assert tuple(schema["required"]) == qdd_skill.REQUIRED_TOP_KEYS
    assert set(schema["$defs"]["keyframe"]["properties"]) == qdd_skill.KEYFRAME_KEYS
    assert set(schema["$defs"]["keyframe"]["properties"]["torso"]["properties"]) == qdd_skill.TORSO_KEYS
    rw = schema["properties"]["reward_weights"]["properties"]
    assert set(rw) == set(qdd_skill.REWARD_WEIGHT_BOUNDS)
    for k, (lo, hi, d) in qdd_skill.REWARD_WEIGHT_BOUNDS.items():
        assert (rw[k]["minimum"], rw[k]["maximum"], rw[k]["default"]) == (lo, hi, d), k
    assert schema["properties"]["duration_s"]["minimum"] == qdd_skill.DURATION_RANGE_S[0]
    assert schema["properties"]["duration_s"]["maximum"] == qdd_skill.DURATION_RANGE_S[1]
    # joint ranges = joint range - 3 deg (knee may straighten fully)
    lim = np.degrees(qdd_skill._joint_limits_rad())
    jp = schema["$defs"]["joints_deg"]["properties"]
    for j, n in enumerate(QDD_JOINT_NAMES):
        lo = lim[j, 0] + (0 if n.endswith("knee") else qdd_skill.JOINT_LIMIT_MARGIN_DEG)
        hi = lim[j, 1] - qdd_skill.JOINT_LIMIT_MARGIN_DEG
        assert jp[n]["minimum"] == pytest.approx(lo, abs=0.05), n
        assert jp[n]["maximum"] == pytest.approx(hi, abs=0.05), n
        if n.startswith("l_"):
            assert jp[n[2:]] == jp[n]


def _mutate(fn):
    raw = _bow_raw()
    fn(raw)
    return raw


def _set_kf(i, **kw):
    def f(raw):
        raw["keyframes"][i].update(kw)

    return f


BAD_SPECS = {
    "not_object": ([1, 2], "JSON object"),
    "missing_keyframes": (_mutate(lambda r: r.pop("keyframes")), "'keyframes' is required"),
    "terminations": (_mutate(lambda r: r.update(terminations={"fell_over": 90})), "fixed by the template"),
    "schema_version": (_mutate(lambda r: r.update(schema_version=2)), "schema_version"),
    "bad_name": (_mutate(lambda r: r.update(name="Bow!")), "name:"),
    "long_duration": (_mutate(lambda r: r.update(duration_s=20.0)), "duration_s: 20.0 is outside"),
    "weight_high": (
        _mutate(lambda r: r["reward_weights"].update(joint_tracking=50)),
        "reward_weights.joint_tracking",
    ),
    "balance_off": (_mutate(lambda r: r["reward_weights"].update(balance=0.0)), "reward_weights.balance"),
    "unknown_weight": (_mutate(lambda r: r["reward_weights"].update(alive=1.0)), "unknown term"),
    "start_not_stand": (
        _mutate(_set_kf(0, joints_deg={"knee": 60})),
        "must start in the standing pose",
    ),
    "end_time": (_mutate(_set_kf(3, t=2.5)), "last keyframe must be at t = duration_s"),
    "keyframe_gap": (_mutate(_set_kf(2, t=1.3)), "at least 0.2 s after"),
    "unknown_joint": (_mutate(_set_kf(1, joints_deg={"elbow": 10})), "unknown joint"),
    "ambiguous_joint": (_mutate(_set_kf(1, joints_deg={"knee": 60, "l_knee": 50})), "ambiguous"),
    "joint_range": (_mutate(_set_kf(1, joints_deg={"ankle_pitch": -60})), "outside the allowed range"),
    "knee_max": (
        _mutate(_set_kf(1, joints_deg={"hip_pitch": 5, "knee": 85, "ankle_pitch": -30})),
        "pose-dependent knee limit",
    ),
    "torso_mismatch": (_mutate(_set_kf(1, torso={"pitch_deg": 10})), "the joint targets give"),
    "too_fast": (
        _mutate(
            lambda r: r.update(
                keyframes=[
                    {"t": 0.0},
                    {"t": 0.2, "joints_deg": {"hip_pitch": -50, "knee": 75, "ankle_pitch": -25}},
                    {"t": 3.0},
                ]
            )
        ),
        "rad/s",
    ),
    "too_deep": (
        _mutate(_set_kf(1, joints_deg={"hip_pitch": -75, "knee": 60, "ankle_pitch": -30}, torso={})),
        "torso pitch reaches",
    ),
    "one_foot_tilted": (
        _mutate(_set_kf(1, joints_deg={"hip_pitch": -55, "knee": 60, "l_ankle_pitch": -30, "r_ankle_pitch": -15})),
        "soles are not parallel",
    ),
    "unbalanced": (
        # lean the whole body forward from the ankles: CoM moves past the toes
        _mutate(_set_kf(1, joints_deg={"hip_pitch": 5, "knee": 40, "ankle_pitch": -45}, torso={})),
        "statically unbalanced",
    ),
}


@pytest.mark.parametrize("case", sorted(BAD_SPECS))
def test_bad_specs_are_rejected_with_a_clear_error(case):
    raw, needle = BAD_SPECS[case]
    with pytest.raises(qdd_skill.SkillSpecError) as e:
        qdd_skill.validate_spec(copy.deepcopy(raw))
    assert any(needle in msg for msg in e.value.errors), (case, e.value.errors)


def test_bilateral_names_mirror_yaw():
    """Side-less names mirror (yaw / roll negate on the right). Toe-out with the
    standing crouch keeps the soles nearly parallel; with a deep hip flexion the
    yawed pitch axis would tilt the soles (the validator catches that)."""
    raw = {
        "schema_version": 1,
        "name": "toe_out",
        "duration_s": 2.0,
        "keyframes": [{"t": 0.0}, {"t": 1.0, "joints_deg": {"hip_yaw": 3.0}}, {"t": 2.0}],
    }
    spec = qdd_skill.validate_spec(raw)
    q = spec["keyframes"][1]["_q_rad"]
    assert q[QDD_JOINT_NAMES.index("l_hip_yaw")] == pytest.approx(math.radians(3.0))
    assert q[QDD_JOINT_NAMES.index("r_hip_yaw")] == pytest.approx(math.radians(-3.0))
    raw["keyframes"][1]["joints_deg"] = {"hip_yaw": 5.0, "hip_pitch": -55.0, "knee": 60.0, "ankle_pitch": -30.0}
    with pytest.raises(qdd_skill.SkillSpecError, match="soles are not parallel"):
        qdd_skill.validate_spec(raw)


def test_invalid_json_text():
    with pytest.raises(qdd_skill.SkillSpecError, match="not valid JSON"):
        qdd_skill.load_spec_text("{nope", "x.json")


# --- reference ---------------------------------------------------------------------


def test_reference_from_keyframes():
    spec = qdd_skill.load_spec("bow")
    ref = qdd_skill.build_reference(spec)
    stand = qdd_skill.stand_pose()
    n = len(ref["t"]) - 1
    assert n == round(qdd_skill.total_s(spec) / qdd_skill.POLICY_DT_S) == 200
    assert np.allclose(ref["q"][0], stand) and np.allclose(ref["q"][-1], stand)
    # standing: upright, soles on the floor at the stand height
    assert abs(ref["pitch"][0]) < 1e-6 and abs(ref["roll"][0]) < 1e-6
    assert ref["height"][0] == pytest.approx(STAND_BASE_Z, abs=1e-3)
    # keyframes are hit exactly; min-jerk = zero velocity at keyframes
    for kf in spec["keyframes"]:
        k = round(kf["t"] / qdd_skill.POLICY_DT_S)
        assert np.allclose(ref["q"][k], kf["_q_rad"], atol=1e-9)
    k1 = round(1.2 / qdd_skill.POLICY_DT_S)
    assert math.degrees(ref["pitch"][k1]) == pytest.approx(25.0, abs=0.5)
    v = np.abs(np.diff(ref["q"], axis=0)) / qdd_skill.POLICY_DT_S
    assert v[k1].max() < 0.01 and v[k1 - 1].max() < 0.01
    # monotone bow down / up, settle = standing
    hp = ref["q"][:, QDD_JOINT_NAMES.index("l_hip_pitch")]
    assert np.all(np.diff(hp[: k1 + 1]) <= 1e-12)
    assert np.all(ref["in_skill"][:150]) and not np.any(ref["in_skill"][150:])
    assert np.allclose(ref["q"][150:], stand)
    assert ref["support_margin"].min() > qdd_skill.COM_SUPPORT_MARGIN_M


def test_linear_interpolation():
    kf_t = np.array([0.0, 1.0])
    kf_q = np.array([[0.0], [1.0]])
    q = qdd_skill.sample_joint_reference(np.array([0.0, 0.25, 0.5, 1.0, 2.0]), kf_t, kf_q, "linear")
    assert np.allclose(q[:, 0], [0.0, 0.25, 0.5, 1.0, 1.0])
    qm = qdd_skill.sample_joint_reference(np.array([0.5]), kf_t, kf_q, "minjerk")
    assert qm[0, 0] == pytest.approx(0.5)


# --- env ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def skill_env():
    from mjlab.envs import ManagerBasedRlEnv

    cfg = make_qdd_biped_skill_env_cfg()  # training cfg (DR, RSI)
    cfg.scene.num_envs = 4
    cfg.commands["skill"].rsi_fraction = 0.0
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    yield env
    env.close()


def test_env_obs_layout_matches_walking(skill_env):
    env = skill_env
    walk = build_policy_io()
    om = env.observation_manager
    names = om.active_terms["actor"]
    dims = [int(np.prod(d)) for d in om.group_obs_term_dim["actor"]]
    assert names == [t["name"] for t in walk["observation"]["terms"]]
    assert dims == [t["size"] for t in walk["observation"]["terms"]]
    assert env.step_dt == pytest.approx(walk["policy_dt_s"])


def test_env_steps_phase_and_timeout(skill_env):
    env = skill_env
    obs, _ = env.reset()
    sk = env.command_manager.get_term("skill")
    assert torch.all(sk.k0 == 0)
    assert torch.allclose(obs["actor"][:, 39:41], torch.tensor([0.0, 1.0]).expand(4, 2), atol=1e-6)
    act = torch.zeros(4, 10)
    for k in range(1, 6):
        obs, rew, term, trunc, _ = env.step(act)
        assert torch.all(obs["actor"][:, 36:39] == 0.0)  # command slot
        s = k / sk.n
        want = torch.tensor([math.sin(math.pi * s), math.cos(math.pi * s)])
        assert torch.allclose(obs["actor"][:, 39:41], want.expand(4, 2), atol=1e-5)
        assert torch.all(torch.isfinite(rew))
    # time-out exactly at the end of the skill
    env.episode_length_buf[:] = sk.n - 1
    _, _, term, trunc, _ = env.step(act)
    assert torch.all(trunc) and not torch.any(term)
    assert torch.all(env.episode_length_buf == 0)


def test_reference_state_init_puts_robot_on_the_reference(skill_env):
    env = skill_env
    sk = env.command_manager.get_term("skill")
    sk.cfg.rsi_fraction = 1.0
    try:
        env.reset()
    finally:
        sk.cfg.rsi_fraction = 0.0
    assert torch.all(sk.k0 > 0)
    env.sim.forward()
    asset = sk.asset
    k = sk.step_index
    assert torch.allclose(sk.joint_pos(), sk.ref_q[k], atol=1e-5)
    g = asset.data.projected_gravity_b
    pitch = torch.atan2(g[:, 0], -g[:, 2])
    assert torch.allclose(pitch, sk.ref_pitch[k], atol=1e-3)
    sole_z = asset.data.site_pos_w[:, [asset.find_sites("l_foot_site")[0][0], asset.find_sites("r_foot_site")[0][0]], 2]
    assert torch.all(sole_z > 0.0) and torch.all(sole_z < 0.01)


def test_rewards_peak_on_the_reference():
    """Standing still in play mode at t = 0 (= reference) gets ~full tracking reward."""
    from mjlab.envs import ManagerBasedRlEnv

    cfg = make_qdd_biped_skill_env_cfg(play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        env.reset()
        env.sim.forward()
        assert microduck_mdp.skill_joint_tracking(env, 0.15).item() > 0.99
        assert microduck_mdp.skill_torso_orientation(env, 0.1).item() > 0.99
        assert microduck_mdp.skill_torso_height(env, 0.02).item() > 0.8  # spawned up to 5 mm high
        assert microduck_mdp.skill_final_stand(env, 0.15, 1.0).item() == 0.0  # not settling yet
    finally:
        env.close()


# --- policy_io for skills ------------------------------------------------------------


def test_skill_policy_io_clock_reproduces_training_phase():
    spec = qdd_skill.load_spec("bow")
    io = build_skill_policy_io(spec)
    walk = build_policy_io()
    assert io["kind"] == "skill" and io["task"] == "Mjlab-Skill-QddBiped"
    assert io["schema_version"] == walk["schema_version"] == 1
    assert io["action"] == walk["action"] and io["actuators"] == walk["actuators"]
    strip = lambda terms: [{k: v for k, v in t.items() if k != "description"} for t in terms]  # noqa: E731
    assert strip(io["observation"]["terms"]) == strip(walk["observation"]["terms"])
    sk = io["skill"]
    assert sk["n_steps"] == 200 and sk["total_s"] == 4.0 and sk["spec_sha256"] == spec["_sha256"]
    # the controller's gait clock (policy_io.rs GaitClock::update, before each obs)
    g = io["gait_clock"]
    assert g["active_threshold"] < 0  # runs with a zero command
    phi = np.float32(g["initial_phase"])
    dt = np.float32(io["policy_dt_s"])
    for k in range(sk["n_steps"] + 1):
        phi = np.float32(np.remainder(phi + dt / np.float32(g["period_s"]), np.float32(1.0)))
        got = np.array([np.sin(2 * np.pi * phi), np.cos(2 * np.pi * phi)])
        s = k / sk["n_steps"]
        assert np.allclose(got, [math.sin(math.pi * s), math.cos(math.pi * s)], atol=1e-4), k
    assert io["commands"]["lin_vel_x_range"] == [0.0, 0.0]


def test_skill_onnx_metadata():
    spec = qdd_skill.load_spec("bow")
    md = skill_onnx_metadata(spec)
    import hashlib

    assert hashlib.sha256(md["policy_io"].encode("utf-8")).hexdigest() == md["policy_io_sha256"]
    assert json.loads(md["skill_spec"])["name"] == "bow"
    assert json.loads(md["policy_io"])["skill"]["name"] == "bow"


# --- warm start / rl cfg ---------------------------------------------------------------


def test_rl_cfg():
    assert QddSkillRlCfg.algorithm.symmetry_cfg is None
    assert QddSkillRlCfg.experiment_name == "qdd_biped_skill"
    assert QddSkillRlCfg.actor.obs_normalization is True


def test_warm_start_actor_matches_walking_onnx(skill_env):
    ort = pytest.importorskip("onnxruntime")
    from dataclasses import asdict

    from mjlab.rl import RslRlVecEnvWrapper

    from mjlab_microduck.tasks import QddSkillOnPolicyRunner

    assert DEFAULT_WARM_START_ONNX.exists()
    env = RslRlVecEnvWrapper(skill_env)
    runner = QddSkillOnPolicyRunner(env, asdict(QddSkillRlCfg), device="cpu")
    obs = env.get_observations()
    runner.alg.actor.eval()
    with torch.no_grad():
        mine = runner.alg.actor(obs).numpy()
    sess = ort.InferenceSession(str(DEFAULT_WARM_START_ONNX))
    x = obs["actor"].numpy().astype(np.float32)
    ref = np.concatenate([sess.run(None, {"obs": x[i : i + 1]})[0] for i in range(len(x))])
    assert np.allclose(mine, ref, atol=1e-4)
