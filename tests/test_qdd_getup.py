"""Mjlab-Getup-QddBiped and Mjlab-Velocity-Rough-QddBiped (CPU)."""

import json
import math

import mujoco
import numpy as np
import pytest
import torch

# tasks first (circular import through mjlab -> mjlab_microduck.tasks)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import symmetry as sym
from mjlab_microduck.tasks.qdd_biped_getup_env_cfg import (
    GETUP_KIND_PROBS,
    GETUP_ACTION_SCALE,
    build_getup_policy_io,
    getup_policy_io_json,
    make_qdd_biped_getup_env_cfg,
)
from mjlab_microduck.tasks.qdd_biped_rough_env_cfg import make_qdd_biped_rough_env_cfg
from mjlab_microduck.tasks.qdd_biped_velocity_env_cfg import build_policy_io, make_qdd_biped_velocity_env_cfg
from mjlab_microduck.robot import qdd_biped_constants as rc


# ---------------------------------------------------------------- collision model


def test_getup_collision_geoms_touch_the_floor_only():
    m = rc.get_qdd_biped_getup_spec().compile()
    added = [m.geom(i).name for i in range(m.ngeom) if m.geom(i).name.endswith("_collision")]
    motors = [n for n in added if "_motor" in n and n.endswith("_collision") and "knee_motor_collision" not in n]
    # 2 hip yaw (torso) + per leg: roll, pitch, knee, ankle
    assert len(motors) == 2 + 2 * 4
    assert "torso_top_collision" in added
    for n in motors + ["torso_top_collision"]:
        g = m.geom(n)
        assert g.contype[0] == rc.GETUP_FLOOR_BIT and g.conaffinity[0] == 0
        assert g.condim[0] == 3 and g.friction[0] > 0.0
    # walking model untouched
    m0 = rc.get_qdd_biped_spec().compile()
    assert not any("_motor0_collision" in m0.geom(i).name for i in range(m0.ngeom))
    # same masses (added shapes are massless)
    assert m.body_subtreemass[1] == pytest.approx(m0.body_subtreemass[1])


# ---------------------------------------------------------------- fallen-state library


def test_getup_state_library():
    d = np.load(rc.QDD_GETUP_STATES_NPZ)
    assert tuple(str(n) for n in d["joint_names"]) == rc.QDD_JOINT_NAMES
    n = d["root_z"].shape[0]
    assert n >= 1000
    assert d["root_quat"].shape == (n, 4) and d["joint_pos"].shape == (n, 10)
    assert np.allclose(np.linalg.norm(d["root_quat"], axis=1), 1.0, atol=1e-4)
    kinds = [str(k) for k in d["kinds"]]
    # falls: lying, not standing (torso low and tilted)
    fall = np.isin(d["kind"], [kinds.index(k) for k in ("push", "stride", "drop")])
    assert np.median(d["tilt_deg"][fall]) > 60.0
    assert np.all(d["root_z"][fall] < 0.3)
    # crouch (reverse curriculum): many still upright-ish, none standing tall
    crouch = d["kind"] == kinds.index("crouch")
    assert 0.2 < np.mean(d["tilt_deg"][crouch] < 30.0) < 0.9
    assert np.all(d["root_z"][crouch] < rc.STAND_BASE_Z)
    # every kind represented
    assert kinds == ["push", "stride", "drop", "crouch", "path", "bridge_prone", "bridge_supine"]
    for i, k in enumerate(kinds):
        assert (d["kind"] == i).sum() > 0.05 * n, k
    # velocities: zero for the simulated (resting) kinds, stored for path states
    assert d["root_vel"].shape == (n, 6) and d["joint_vel"].shape == (n, 10)
    path = d["kind"] == kinds.index("path")
    assert np.all(d["root_vel"][~path] == 0) and np.all(d["joint_vel"][~path] == 0)
    assert np.abs(d["joint_vel"][path]).max() > 1.0
    # path states span lying to standing
    assert d["root_z"][path].min() < 0.1 and d["root_z"][path].max() > 0.33
    # within the joint ranges and the knee stop (tolerances as in the generator)
    m = rc.get_qdd_biped_spec().compile()
    lo = np.array([m.joint(j).range[0] for j in rc.QDD_JOINT_NAMES])
    hi = np.array([m.joint(j).range[1] for j in rc.QDD_JOINT_NAMES])
    q = d["joint_pos"]
    assert np.all(q > lo - 0.06) and np.all(q < hi + 0.06)
    for s in (0, 5):
        km = np.array([rc.knee_max(float(h)) for h in q[:, s + 2]])
        assert np.all(q[:, s + 3] <= km + 0.11)


# ---------------------------------------------------------------- cfg / policy_io


def test_getup_cfg():
    cfg = make_qdd_biped_getup_env_cfg()
    assert cfg.commands == {}
    assert cfg.actions["joint_pos"].scale == GETUP_ACTION_SCALE
    assert set(cfg.terminations) == {"time_out", "nan_state", "knee_limit"}
    assert list(cfg.observations["actor"].terms) == [
        "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions",
    ]
    for k in ("root_height", "body_contact", "base_lin_vel"):
        assert k in cfg.observations["critic"].terms
    # sign convention (AGENTS.md): costs >= 0 with a negative weight, rewards positive
    w = {k: v.weight for k, v in cfg.rewards.items()}
    for k in ("getup_progress", "standing", "stand_still", "stand_pose"):
        assert w[k] > 0, k
    # costs: negative at their final stage (ramped from 0, see PENALTY_RAMP)
    final = {k: v.weight for k, v in cfg.rewards.items()}
    for name, c in cfg.curriculum.items():
        if name.endswith("_weight") and "reward_name" in c.params:
            final[c.params["reward_name"]] = c.params["weight_stages"][-1]["weight"]
    for k in ("torso_ang_vel", "body_impact", "action_rate_l2", "joint_torques", "knee_limit", "self_collisions"):
        assert final[k] < 0 and w[k] <= 0, k
    first = next(iter(cfg.events))
    assert first == "reset_getup"  # before the DR events, replaces reset_base / reset_robot_joints
    assert "reset_base" not in cfg.events and "push_robot" not in cfg.events
    play = make_qdd_biped_getup_env_cfg(play=True)
    assert play.events["reset_getup"].params["fixed_order"] is True
    assert "randomize_pd_gains" not in play.events and not play.observations["actor"].enable_corruption


def test_getup_policy_io_json_is_up_to_date():
    assert json.loads(rc.QDD_GETUP_POLICY_IO_JSON.read_text()) == json.loads(getup_policy_io_json())
    io = build_getup_policy_io()
    walk = build_policy_io()
    assert io["kind"] == "getup" and io["observation"]["dim"] == 36
    assert io["action"]["scale"] == GETUP_ACTION_SCALE
    # same robot side as walking: joint order, default pose, gains, rate
    for k in ("joint_order", "default_joint_pos"):
        assert io["action"][k] == walk["action"][k]
    assert io["actuators"] == walk["actuators"]
    assert io["policy_dt_s"] == walk["policy_dt_s"]
    assert "gait_clock" not in io and "commands" not in io
    assert io["getup"]["done"]["max_tilt_deg"] > 0


def test_getup_mirror_is_the_walking_table_prefix():
    perm = np.array(sym.QDD_OBS_PERM[:36])
    assert np.all(perm < 36)
    sign = np.array(sym.QDD_OBS_SIGN[:36])
    x = np.random.default_rng(0).normal(size=36)
    assert np.allclose((x[perm] * sign)[perm] * sign, x)


def test_rough_cfg_keeps_the_walking_contract():
    flat = make_qdd_biped_velocity_env_cfg()
    rough = make_qdd_biped_rough_env_cfg()
    assert list(rough.observations["actor"].terms) == list(flat.observations["actor"].terms)
    assert "height_scan" in rough.observations["critic"].terms
    assert "height_scan" not in rough.observations["actor"].terms
    gen = rough.scene.terrain.terrain_generator
    assert rough.scene.terrain.terrain_type == "generator" and gen.curriculum
    assert "terrain_levels" in rough.curriculum
    for k in ("gait_swing_height", "swing_clearance", "feet_flat"):
        assert rough.rewards[k].params["height_sensor_name"] == "sole_clearance_scan"
        assert "height_sensor_name" not in flat.rewards[k].params
    assert rough.actions["joint_pos"].scale == flat.actions["joint_pos"].scale
    assert rough.rewards["feet_flat"].weight > flat.rewards["feet_flat"].weight  # weaker cost


# ---------------------------------------------------------------- live get-up env


@pytest.fixture(scope="module")
def getup_env():
    from mjlab.envs import ManagerBasedRlEnv

    cfg = make_qdd_biped_getup_env_cfg(play=True)
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    yield env
    env.close()


def test_getup_obs_matches_policy_io(getup_env):
    om = getup_env.observation_manager
    io = build_getup_policy_io()
    assert om.active_terms["actor"] == [t["name"] for t in io["observation"]["terms"]]
    assert om.group_obs_dim["actor"] == (36,)


def test_getup_reset_places_library_states(getup_env):
    env = getup_env
    env.reset()
    term = env.event_manager.get_term_cfg("reset_getup").func
    lib = np.load(rc.QDD_GETUP_STATES_NPZ)
    idx = term.index.cpu().numpy()
    assert np.all(idx == np.arange(env.num_envs))  # play: fixed order
    robot = env.scene["robot"]
    z = (robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).cpu().numpy()
    assert np.allclose(z, lib["root_z"][idx] + 0.003, atol=5e-3)
    q = robot.data.joint_pos.cpu().numpy()
    assert np.allclose(q, lib["joint_pos"][idx], atol=1e-4)
    qd = robot.data.joint_vel.cpu().numpy()
    assert np.allclose(qd, lib["joint_vel"][idx], atol=1e-4)
    # lying robots score ~0 standing, and the first progress step pays 0
    st = microduck_mdp.getup_standing(env, rc.STAND_BASE_Z, "feet_ground_contact")
    assert torch.all(st[lib["kind"][idx] < 3] < 0.1)  # falls (crouch / path starts may be up)
    assert any(k.startswith("Metrics/getup_stood_frac_") for k in env.extras["log"])


def test_getup_rewards_standing_robot(getup_env):
    """A robot standing in the default pose scores ~full standing / pose terms."""
    env = getup_env
    env.reset()
    robot = env.scene["robot"]
    ids = torch.arange(env.num_envs)
    root = torch.zeros(env.num_envs, 13)
    root[:, :3] = env.scene.env_origins
    root[:, 2] += rc.STAND_BASE_Z + 0.002
    root[:, 3] = 1.0
    robot.write_root_state_to_sim(root, env_ids=ids)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros(env.num_envs, 10), env_ids=ids)
    act = torch.zeros(env.num_envs, 10)
    for _ in range(25):
        env.step(act)
    sense = dict(stand_z=rc.STAND_BASE_Z, sensor_name="feet_ground_contact")
    assert torch.all(microduck_mdp.getup_standing(env, **sense) > 0.7)
    assert torch.all(microduck_mdp.getup_stand_pose(env, **sense) > 0.7)  # PD sag
    assert torch.all(microduck_mdp.getup_stand_still(env, **sense) > 0.5)
    assert torch.all(microduck_mdp.body_impact_cost(env, "body_floor_contact", 150.0) == 0.0)


def test_getup_progress_is_potential_based(getup_env):
    env = getup_env
    env.reset()
    term = env.reward_manager.get_term_cfg("getup_progress").func
    term.reset(None)
    r0 = term(env, stand_z=rc.STAND_BASE_Z)
    assert torch.all(r0 == 0.0)  # first step after a reset
    # standing still: no further pay
    r1 = term(env, stand_z=rc.STAND_BASE_Z)
    assert torch.allclose(r1, torch.zeros_like(r1), atol=1e-6)


def test_rough_env_steps():
    from mjlab.envs import ManagerBasedRlEnv

    cfg = make_qdd_biped_rough_env_cfg(play=True)
    cfg.scene.num_envs = 12  # 2 robots per terrain type
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        obs, _ = env.reset()
        assert obs["actor"].shape == (12, 41)
        for _ in range(5):
            obs, rew, *_ = env.step(torch.zeros(12, 10))
        assert torch.isfinite(rew).all() and torch.isfinite(obs["critic"]).all()
        h = env.scene["sole_clearance_scan"].data.heights - 0.05
        assert h.shape == (12, 2) and torch.all(h.abs() < 0.03)  # standing: soles on the ground
    finally:
        env.close()


def test_rough_terrain_levels_follow_survival():
    """Up after a whole tracked episode, down after a fall, unchanged otherwise."""
    from mjlab.envs import ManagerBasedRlEnv

    cfg = make_qdd_biped_rough_env_cfg()
    cfg.scene.num_envs = 6
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        env.reset()
        terrain = env.scene.terrain
        term = env.command_manager.get_term("twist")
        ids = torch.arange(6)
        terrain.terrain_levels[:] = 3
        env.episode_length_buf[:] = env.max_episode_length
        term.metrics["error_vel_xy"][:] = 0.0
        term.metrics["error_vel_xy"][2] = 10.0  # tracked badly
        env.termination_manager._terminated_buf[:] = False
        env.termination_manager._terminated_buf[1] = True  # fell
        env.episode_length_buf[3] = 10  # short episode, no fall (cannot happen, but: no move)
        out = microduck_mdp.qdd_terrain_levels_survival(env, ids)
        assert terrain.terrain_levels.tolist() == [4, 2, 3, 3, 4, 4]
        assert "stairs" in out and "flat" in out
        # looser bar per column: an error of 0.3 m/s moves up on slope, not on flat
        names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
        terrain.terrain_levels[:] = 3
        env.termination_manager._terminated_buf[:] = False
        env.episode_length_buf[:] = env.max_episode_length
        max_cmd_steps = term.cfg.resampling_time_range[1] / env.step_dt
        term.metrics["error_vel_xy"][:] = 0.3 * env.max_episode_length / max_cmd_steps
        terrain.terrain_types[:3] = names.index("slope")
        terrain.terrain_types[3:] = names.index("flat")
        microduck_mdp.qdd_terrain_levels_survival(env, ids, max_err_xy_by_type={"slope": 0.4})
        assert terrain.terrain_levels.tolist() == [4, 4, 4, 3, 3, 3]
    finally:
        env.close()


def test_getup_path_mirror_is_an_involution():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gen", "scripts/qdd_getup_states.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    rng = np.random.default_rng(1)
    q = rng.normal(size=17)
    q[3:7] /= np.linalg.norm(q[3:7])
    v = rng.normal(size=16)
    q2, v2 = gen._mirror(*gen._mirror(q, v))
    assert np.allclose(q2, q) and np.allclose(v2, v)
    qm, _ = gen._mirror(q, v)
    assert np.allclose(qm[7:12], q[12:17] * [-1, -1, 1, 1, 1])  # legs swapped, yaw / roll negated


# ---------------------------------------------------------------- path reverse curriculum


def _path_remain_s():
    import importlib.util
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "scripts" / "qdd_getup_states.py"
    spec = importlib.util.spec_from_file_location("qdd_getup_states", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.path_remain_s


def test_getup_library_remain_s():
    remain = _path_remain_s()
    # demo 0 (prone, 3 frames) ends standing; demo 1 (2 frames) ends at demo 0's start
    r = remain(np.array([0, 0, 0, 1, 1]), dt=0.02)
    assert np.allclose(r, [0.04, 0.02, 0.0, 0.06, 0.04])
    d = np.load(rc.QDD_GETUP_STATES_NPZ)
    path = d["kind"] == list(d["kinds"]).index("path")
    assert np.all(d["remain_s"][~path] == -1.0)
    assert d["remain_s"][path].min() < 0.1 and d["remain_s"][path].max() > 8.0


@pytest.fixture(scope="module")
def getup_train_env():
    from mjlab.envs import ManagerBasedRlEnv

    cfg = make_qdd_biped_getup_env_cfg(play=False)
    cfg.scene.num_envs = 8
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    yield env
    env.close()


def test_getup_path_starts_follow_the_frontier(getup_train_env):
    env = getup_train_env
    term = env.event_manager.get_term_cfg("reset_getup").func
    assert term.frontier_s == 1.0
    idx = term._sample(20000, GETUP_KIND_PROBS, "cpu")
    kind = term.kind[idx]
    p = term.kinds.index("path")
    assert torch.all(term.remain_s[idx[kind == p]] <= 1.0)
    # half of the path draws from the band (0.5, 1.0]
    band = (term.remain_s[idx[kind == p]] > 0.5).float().mean()
    assert 0.6 < band < 0.9
    # per-kind shares as configured
    for k, share in GETUP_KIND_PROBS.items():
        assert abs((kind == term.kinds.index(k)).float().mean() - share) < 0.02, k


def test_getup_path_frontier_moves_back_only_when_band_stands(getup_train_env, monkeypatch):
    env = getup_train_env
    term = env.event_manager.get_term_cfg("reset_getup").func
    cur = env.curriculum_manager.get_term_cfg("path_frontier")
    fn = cur.func
    ids = torch.arange(env.num_envs)
    rounds = cur.params["window"] // env.num_envs + 1
    stood = {"v": 0.0}
    monkeypatch.setattr(
        microduck_mdp, "_getup_standing_score", lambda env, *a, **k: torch.full((env.num_envs,), stood["v"])
    )
    P = term.kinds.index("path")
    term.frontier_s = 1.0
    fn.cnt = {}
    env.episode_length_buf[:] = 10
    term.index[:] = int(torch.nonzero(term.is_path)[0, 0])

    # band starts that end lying: the frontier stays
    for _ in range(rounds):
        term.in_band[:] = P
        out = fn(env, ids, **cur.params)
    assert term.frontier_s == 1.0 and out["band_stood"] == 0.0
    # starts outside the band do not count, even standing
    stood["v"] = 1.0
    fn.cnt = {}
    term.in_band[:] = -1
    for _ in range(rounds):
        fn(env, ids, **cur.params)
    assert fn.n == 0 and term.frontier_s == 1.0
    # band starts that end standing: one step back per window
    for _ in range(rounds):
        term.in_band[:] = P
        out = fn(env, ids, **cur.params)
    assert term.frontier_s == 1.0 + cur.params["step_s"]
    assert out["frontier_s"] == term.frontier_s and out["end_stood_path"] > 0.5
    # the startup reset (episode length 0) is ignored
    env.episode_length_buf[:] = 0
    fn.n = 0
    fn(env, ids, **cur.params)
    assert fn.n == 0
    term.frontier_s = 1.0


def test_getup_library_bridges():
    d = np.load(rc.QDD_GETUP_STATES_NPZ)
    kinds = list(d["kinds"])
    w, x, y, z = d["root_quat"].T
    prone = 2 * (x * z - w * y) < 0
    for face in ("prone", "supine"):
        sel = d["kind"] == kinds.index(f"bridge_{face}")
        assert sel.sum() > 500, face
        a = d["alpha"][sel]
        assert a.min() >= 0.0 and a.max() <= 1.0 and a.min() < 0.05 and a.max() > 0.95
        assert np.all(prone[sel] == (face == "prone"))  # still on the same face
    assert np.all(d["alpha"][d["kind"] < kinds.index("bridge_prone")] == -1.0)


def test_getup_bridge_frontier_moves_on_its_own(getup_train_env, monkeypatch):
    env = getup_train_env
    term = env.event_manager.get_term_cfg("reset_getup").func
    cur = env.curriculum_manager.get_term_cfg("path_frontier")
    fn = cur.func
    ids = torch.arange(env.num_envs)
    monkeypatch.setattr(microduck_mdp, "_getup_standing_score", lambda env, *a, **k: torch.ones(env.num_envs))
    start = dict(term.frontier)
    fn.cnt = {}
    env.episode_length_buf[:] = 10
    bp = term.kinds.index("bridge_prone")
    term.index[:] = int(torch.nonzero(term.kind == bp)[0, 0])
    for _ in range(cur.params["window"] // env.num_envs + 1):
        term.in_band[:] = bp
        out = fn(env, ids, **cur.params)
    assert term.frontier["bridge_prone"] == pytest.approx(start["bridge_prone"] + cur.params["steps"]["bridge_prone"])
    assert term.frontier["bridge_supine"] == start["bridge_supine"]
    assert term.frontier["path"] == start["path"]
    assert "frontier_bridge_prone" in out and "end_stood_bridge_prone" in out
    # bridge draws respect the frontier
    idx = term._sample(5000, {"bridge_prone": 1.0}, "cpu")
    assert torch.all(term.level[idx] <= term.frontier["bridge_prone"] + 1e-6)
    term.frontier.update(start)


def test_getup_demo_tracking_scores_the_demo_and_its_mirror(getup_train_env):
    env = getup_train_env
    term = env.event_manager.get_term_cfg("reset_getup").func
    tr = env.reward_manager.get_term_cfg("demo_tracking").func
    robot = env.scene["robot"]
    demo = np.load(rc.QDD_GETUP_DEMO_NPZ)
    path = torch.nonzero(term.demo_frame >= 0).squeeze(1)
    # a path state from the prone demo, frame 40 (mid-flip)
    j = int(path[(term.demo_frame[path] == 40)][0]) if (term.demo_frame[path] == 40).any() else int(path[0])
    f = int(term.demo_frame[j])
    names = list(rc.QDD_JOINT_NAMES)
    order = [names.index(n) for n in robot.joint_names]
    qp = torch.tensor(demo["qpos"][f], dtype=torch.float32)
    ids = torch.arange(env.num_envs)
    root = torch.zeros(env.num_envs, 13)
    root[:, :3] = env.scene.env_origins
    root[:, 2] += 0.3
    root[:, 3:7] = qp[3:7]
    jp = qp[7:][order].repeat(env.num_envs, 1)
    # env 1: the mirror image (legs swapped, hip yaw/roll negated, y -> -y)
    m = qp.numpy().copy()
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    spec = spec_from_file_location("qgs", Path(__file__).resolve().parents[1] / "scripts" / "qdd_getup_states.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    mq, _ = mod._mirror(m.astype(np.float64), np.zeros(16))
    root[1, 3:7] = torch.tensor(mq[3:7], dtype=torch.float32)
    jp[1] = torch.tensor(mq[7:], dtype=torch.float32)[order]
    robot.write_root_state_to_sim(root, env_ids=ids)
    robot.write_joint_state_to_sim(jp, torch.zeros(env.num_envs, 10), env_ids=ids)
    env.sim.forward()
    term.index[:] = j
    env.episode_length_buf[:] = 1  # first step: reference frame = the state's own frame
    r = tr(env, **env.reward_manager.get_term_cfg("demo_tracking").params)
    assert r[0] > 0.95 and r[1] > 0.95
    # far ahead in time: the reference has ended (prone demo: 170 frames)
    env.episode_length_buf[:] = 400
    assert torch.all(tr(env, **env.reward_manager.get_term_cfg("demo_tracking").params) == 0)
    # crouch and standing starts: nothing
    env.episode_length_buf[:] = 1
    term.index[:] = int(torch.nonzero(term.kind == term.kinds.index("crouch"))[0, 0])
    term.index[0] = -1
    assert torch.all(tr(env, **env.reward_manager.get_term_cfg("demo_tracking").params) == 0)


def test_getup_runner_has_no_mirror_loss():
    """Getting up from a symmetric lying pose starts with one leg: a mirror-
    symmetric policy cannot do that (getup1-6)."""
    from mjlab_microduck.tasks.qdd_biped_getup_env_cfg import QddBipedGetupRlCfg

    sc = QddBipedGetupRlCfg.algorithm.symmetry_cfg
    assert sc is None or not (sc.get("use_mirror_loss") or sc.get("use_data_augmentation"))


def test_getup_attempt_taxes_start_at_zero(getup_train_env):
    """While the get-up is being discovered, thrashing must not cost more than
    following the demo pays (getup1-7)."""
    from mjlab_microduck.tasks.qdd_biped_getup_env_cfg import RAMPED_PENALTIES

    env = getup_train_env
    env.common_step_counter = 0
    env.curriculum_manager.compute(env_ids=torch.arange(env.num_envs))
    for name in RAMPED_PENALTIES:
        assert env.reward_manager.get_term_cfg(name).weight == 0.0, name
    assert env.reward_manager.get_term_cfg("demo_tracking").weight > 0


def test_demo_tracking_waits_for_prep():
    """getup9: a real fall (bridge alpha 1) starts following the demo after the
    prep ramp, a bridge at the demo start (alpha 0) at once."""
    from mjlab.envs import ManagerBasedRlEnv

    from mjlab_microduck.tasks.qdd_biped_getup_env_cfg import DEMO_PREP_S

    cfg = make_qdd_biped_getup_env_cfg(play=False)
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    tr = env.reward_manager.get_term_cfg("demo_tracking").func
    assert isinstance(tr, microduck_mdp.getup_demo_tracking)
    term = tr.term
    prep = int(round(DEMO_PREP_S / env.step_dt))
    k = term.kinds.index("bridge_prone")
    far = torch.nonzero((term.kind == k) & (term.alpha > 0.99))[0, 0]
    near = torch.nonzero((term.kind == k) & (term.alpha < 0.01))[0, 0]
    assert int(tr.state_off[far]) <= -prep + 1
    assert int(tr.state_off[near]) == 0
    fall = torch.nonzero(term.kind == term.kinds.index("push"))[0, 0]
    assert int(tr.state_off[fall]) == -prep
