"""Cfg / model invariants for Mjlab-Velocity-Flat-QddBiped (CPU, plain MuJoCo)."""

import json
import math

import mujoco
import pytest
import numpy as np
import torch

# Import the tasks package first (as mjlab's plugin loader does) to avoid a
# circular import through mjlab -> mjlab_microduck.tasks -> qdd_biped_constants.
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import symmetry as sym
from mjlab_microduck.tasks.qdd_biped_velocity_env_cfg import (
    GAIT_PERIOD_S,
    QddBipedRlCfg,
    build_policy_io,
    make_qdd_biped_velocity_env_cfg,
    policy_io_json,
)
from mjlab_microduck.robot.qdd_biped_constants import (  # noqa: E402
    DEFAULT_JOINT_POS,
    FOOT_GEOMS,
    FOOT_SITES,
    QDD_JOINT_NAMES,
    QDD_POLICY_IO_JSON,
    STAND_BASE_Z,
    get_qdd_biped_spec,
    knee_max,
)


def _model():
    return get_qdd_biped_spec().compile()


def test_model_names_resolve():
    m = _model()
    joints = [m.joint(i).name for i in range(m.njnt) if m.joint(i).name != "root"]
    assert tuple(joints) == QDD_JOINT_NAMES
    for g in FOOT_GEOMS + ("torso_collision", "torso_deck_collision"):
        m.geom(g)
    for s in FOOT_SITES + ("imu",):
        m.site(s)
    for s in ("imu_ang_vel", "imu_lin_vel", "root_angmom"):
        m.sensor(s)
    assert m.nu == 0  # actuators come from the mjlab cfg, not the XML
    assert abs(m.body_subtreemass[1] - 6.607) < 0.01  # robot repo model: + 10 mm TPU sole pads, A7075 yaw/roll brackets



def test_stand_height_puts_soles_on_floor():
    m = _model()
    d = mujoco.MjData(m)
    d.qpos[2] = STAND_BASE_Z
    d.qpos[3] = 1.0
    k = math.radians(45)
    pose = {"hip_pitch": -k / 2, "knee": k, "ankle_pitch": -k / 2}
    for name in QDD_JOINT_NAMES:
        j = m.joint(name)
        d.qpos[j.qposadr[0]] = next((v for key, v in pose.items() if name.endswith(key)), 0.0)
    mujoco.mj_kinematics(m, d)
    for s in FOOT_SITES:
        assert abs(d.site(s).xpos[2]) < 1e-3
    assert DEFAULT_JOINT_POS[r".*_knee"] == k


def test_knee_max_torch_matches_robot_model():
    hp = np.linspace(-1.8, 0.75, 1001)
    ref = np.array([knee_max(float(h)) for h in hp])
    got = microduck_mdp.qdd_knee_max(torch.tensor(hp)).numpy()
    assert np.allclose(got, ref)
    # spot values from robot/model.py
    assert math.isclose(knee_max(math.radians(10)), math.radians(80))
    assert math.isclose(knee_max(math.radians(-50)), math.radians(120))


def test_default_pose_is_inside_knee_limit():
    hp = DEFAULT_JOINT_POS[r".*_hip_pitch"]
    assert DEFAULT_JOINT_POS[r".*_knee"] < knee_max(hp) - 0.05


def test_reward_signs():
    cfg = make_qdd_biped_velocity_env_cfg()
    r = cfg.rewards
    for name in (
        "track_linear_velocity",
        "track_angular_velocity",
        "upright",
        "pose",
        "gait_contact",
        "gait_swing_height",
        "track_angular_velocity_fine",
        "stand_square",
    ):
        assert r[name].weight > 0, name
    # replaced by the gait clock terms
    assert "air_time" not in r and "foot_clearance" not in r
    # mjlab-style COSTS (>= 0) must have negative weights.
    for name in (
        "knee_limit",
        "self_collisions",
        "joint_torques",
        "action_rate_l2",
        "foot_slip",
        "foot_swing_height",
        "feet_flat",
        "touchdown_velocity",
        "feet_spacing",
        "hip_yaw_roll",
        "swing_clearance",
        "body_ang_vel",
        "dof_pos_limits",
    ):
        assert r[name].weight < 0, name


def test_actor_obs_has_no_privileged_terms():
    cfg = make_qdd_biped_velocity_env_cfg()
    actor = cfg.observations["actor"].terms
    assert list(actor) == [
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
        "gait_phase",
    ]
    assert list(cfg.commands) == ["twist", "gait"]  # gait reads the updated twist
    assert "base_lin_vel" in cfg.observations["critic"].terms
    assert actor["joint_pos"].params["biased"] is True
    assert cfg.observations["critic"].terms["joint_pos"].params["biased"] is False


def test_terminations_and_play_mode():
    cfg = make_qdd_biped_velocity_env_cfg()
    for name in ("fell_over", "torso_low", "body_ground_contact", "knee_limit", "nan_state", "time_out"):
        assert name in cfg.terminations, name
    play = make_qdd_biped_velocity_env_cfg(play=True)
    assert "push_robot" not in play.events
    assert play.curriculum == {}


def test_play_mode_is_nominal():
    """Play (export / video) = the nominal robot: no DR events, no obs noise,
    delay or IMU misalignment, fixed command latency, scripted commands."""
    from mjlab_microduck.robot.qdd_biped_constants import QDD_ARTICULATION
    from mjlab_microduck.tasks.qdd_biped_velocity_env_cfg import PLAY_COMMAND_SCHEDULE

    train = make_qdd_biped_velocity_env_cfg()
    play = make_qdd_biped_velocity_env_cfg(play=True)
    assert {"base_com", "encoder_bias", "foot_friction", "randomize_pd_gains"} <= set(train.events)
    assert set(play.events) <= {"reset_base", "reset_robot_joints"}, set(play.events)
    actor = play.observations["actor"]
    assert actor.enable_corruption is False
    assert all(t.delay_max_lag == 0 for t in actor.terms.values())
    assert actor.terms["projected_gravity"].params["max_angle_deg"] == 0.0
    assert actor.terms["joint_pos"].params["biased"] is False
    assert all(a.delay_min_lag == a.delay_max_lag == 1 for a in play.scene.entities["robot"].articulation.actuators)
    # the shared module-level actuator cfgs are untouched (training keeps 0-2)
    assert all(a.delay_max_lag == 2 for a in QDD_ARTICULATION.actuators)
    assert play.commands["twist"].schedule == PLAY_COMMAND_SCHEDULE
    assert train.commands["twist"].rel_forward_envs > 0
    assert "track_linear_velocity_fine" in train.rewards


def test_play_command_schedule_in_env():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab_microduck.tasks.qdd_biped_velocity_env_cfg import PLAY_COMMAND_SCHEDULE

    cfg = make_qdd_biped_velocity_env_cfg(play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        env.reset()
        twist = env.command_manager.get_term("twist")
        gait = env.command_manager.get_term("gait")
        act = torch.zeros(1, 10)
        seen = {}
        for _ in range(round(13.0 / env.step_dt)):
            obs, *_ = env.step(act)
            t = round(float(env.episode_length_buf[0]) * env.step_dt, 6)
            seen[t] = (twist.command[0].tolist(), bool(gait.active[0]))
            assert torch.allclose(obs["actor"][0, 36:39], twist.command[0])
        for t0, vx, vy, wz in PLAY_COMMAND_SCHEDULE:
            t = round(t0 + 0.5, 6)
            if t in seen:
                assert seen[t][0] == pytest.approx([vx, vy, wz]), t
                assert seen[t][1] == (abs(vx) + abs(wz) > 0), t
    finally:
        env.close()


def test_rl_cfg():
    assert QddBipedRlCfg.experiment_name == "qdd_biped_velocity"
    assert QddBipedRlCfg.actor.obs_normalization is True
    sc = QddBipedRlCfg.algorithm.symmetry_cfg
    assert sc is not None and sc["use_mirror_loss"] and not sc["use_data_augmentation"]
    assert sc["data_augmentation_func"].endswith("qdd_biped_symmetry")


# --- symmetry -----------------------------------------------------------------


def test_mirror_tables_are_involutions():
    obs = torch.randn(7, 41)
    act = torch.randn(7, 10)
    o1, a1 = sym.qdd_biped_symmetry(None, {"actor": obs, "critic": obs}, act)
    o2, a2 = sym.qdd_biped_symmetry(None, {"actor": o1["actor"][7:], "critic": obs}, a1[7:])
    assert torch.allclose(o2["actor"][7:], obs) and torch.allclose(a2[7:], act)
    assert sorted(sym.QDD_OBS_PERM) == list(range(41))


def test_mirror_map_matches_robot_geometry():
    """Reflecting the robot about its sagittal plane in MuJoCo = the joint mirror map."""
    m = _model()
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    q = {n: float(rng.uniform(-0.3, 0.3)) for n in QDD_JOINT_NAMES}
    d.qpos[2], d.qpos[3] = 1.0, 1.0
    for n, v in q.items():
        d.qpos[m.joint(n).qposadr[0]] = v
    mujoco.mj_kinematics(m, d)
    left = d.site("l_foot_site").xpos.copy()
    # mirrored joint vector: new[i] = sign[i] * old[perm[i]]
    for i, n in enumerate(QDD_JOINT_NAMES):
        d.qpos[m.joint(n).qposadr[0]] = sym.QDD_JOINT_SIGN[i] * q[QDD_JOINT_NAMES[sym.QDD_JOINT_PERM[i]]]
    mujoco.mj_kinematics(m, d)
    right = d.site("r_foot_site").xpos.copy()
    assert np.allclose(left * [1, -1, 1], right, atol=1e-6)


def test_gait_phase_mirror_is_half_period_shift():
    phi = torch.rand(5)
    ang = 2 * math.pi * phi
    ang2 = 2 * math.pi * (phi + 0.5)
    sign = torch.tensor(sym.QDD_OBS_SIGN[39:41])
    assert torch.allclose(torch.stack((torch.sin(ang), torch.cos(ang)), 1) * sign,
                          torch.stack((torch.sin(ang2), torch.cos(ang2)), 1), atol=1e-6)


# --- policy I/O contract ---------------------------------------------------------


def test_policy_io_json_is_up_to_date():
    """robot/qdd_biped/policy_io.json == build_policy_io(); regenerate with
    `python -m mjlab_microduck.tasks.qdd_biped_velocity_env_cfg`."""
    assert QDD_POLICY_IO_JSON.read_text(encoding="utf-8") == policy_io_json()
    io = json.loads(QDD_POLICY_IO_JSON.read_text(encoding="utf-8"))
    assert io["control_rate_hz"] == 50.0
    assert io["action"]["joint_order"] == list(QDD_JOINT_NAMES)
    assert io["gait_clock"]["period_s"] == GAIT_PERIOD_S
    assert [t["name"] for t in io["observation"]["terms"]][-1] == "gait_phase"


@pytest.fixture(scope="module")
def live_env():
    from mjlab.envs import ManagerBasedRlEnv

    cfg = make_qdd_biped_velocity_env_cfg(play=True)
    cfg.scene.num_envs = 2
    cfg.commands["twist"].schedule = ()  # tests drive the twist command by hand
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    yield env
    env.close()


def test_policy_io_matches_live_env(live_env):
    """The JSON obs layout is the actor's real input order/size; action/PD too."""
    from mjlab.rl.exporter_utils import get_base_metadata

    env = live_env
    io = build_policy_io()
    om = env.observation_manager
    names = om.active_terms["actor"]
    dims = [int(np.prod(d)) for d in om.group_obs_term_dim["actor"]]
    assert names == [t["name"] for t in io["observation"]["terms"]]
    assert dims == [t["size"] for t in io["observation"]["terms"]]
    starts = list(np.cumsum([0] + dims[:-1]))
    assert starts == [t["start"] for t in io["observation"]["terms"]]
    assert om.group_obs_dim["actor"] == (io["observation"]["dim"],) == (41,)
    assert io["onnx"]["input"]["shape"] == [1, 41]
    assert env.step_dt == pytest.approx(io["policy_dt_s"])
    meta = get_base_metadata(env, run_path="test")
    assert meta["joint_names"] == io["action"]["joint_order"]
    assert np.allclose(meta["default_joint_pos"], io["action"]["default_joint_pos"])
    assert np.allclose(meta["joint_stiffness"], io["actuators"]["kp_nm_per_rad"])
    assert np.allclose(meta["joint_damping"], io["actuators"]["kd_nm_s_per_rad"])
    assert meta["action_scale"] == io["action"]["scale"]


def test_gait_clock_runs_and_pauses(live_env):
    env = live_env
    env.reset()
    twist = env.command_manager.get_term("twist")
    gait = env.command_manager.get_term("gait")
    act = torch.zeros(env.num_envs, 10)
    # env 0 walks, env 1 stands
    for _ in range(3):
        twist.vel_command_b[0] = torch.tensor([0.3, 0.0, 0.0])
        twist.is_standing_env[0] = False
        twist.is_standing_env[1] = True
        twist.time_left[:] = 100.0
        before = gait.phase[:, 0].clone()
        obs, *_ = env.step(act)
        after = gait.phase[:, 0]
        assert torch.isclose(
            torch.remainder(after[0] - before[0], 1.0), torch.tensor(env.step_dt / GAIT_PERIOD_S)
        )
        assert after[1] == before[1]
        ang = 2 * math.pi * after
        assert torch.allclose(obs["actor"][:, 39:41], torch.stack((torch.sin(ang), torch.cos(ang)), 1))


def test_standing_robot_satisfies_gait_terms(live_env):
    """Zero command, default pose: both feet down = full contact/height reward,
    flat feet, spacing inside the band."""
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    env = live_env
    env.reset()
    twist = env.command_manager.get_term("twist")
    act = torch.zeros(env.num_envs, 10)
    for _ in range(25):
        twist.is_standing_env[:] = True
        env.step(act)
    feet = SceneEntityCfg("robot", site_names=FOOT_SITES)
    feet.resolve(env.scene)
    assert torch.all(microduck_mdp.gait_contact_schedule(env, "feet_ground_contact") == 1.0)
    h = microduck_mdp.gait_foot_height_tracking(env, 0.05, 0.02, asset_cfg=feet)
    assert torch.all(h > 0.95)
    assert torch.all(microduck_mdp.feet_tilt_near_ground(env, "feet_ground_contact", asset_cfg=feet) < 0.01)
    assert torch.all(microduck_mdp.feet_lateral_spacing(env, 0.08, 0.20, asset_cfg=feet) == 0.0)


# --- gait3 terms: swing clearance, fine yaw tracking, square stance at a stop --


def _feet(env):
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    feet = SceneEntityCfg("robot", site_names=FOOT_SITES)
    feet.resolve(env.scene)
    return feet


def _stand(env, steps=25):
    twist = env.command_manager.get_term("twist")
    act = torch.zeros(env.num_envs, 10)
    for _ in range(steps):
        twist.is_standing_env[:] = True
        twist.time_left[:] = 100.0
        env.step(act)


def test_stand_time_counts_while_paused(live_env):
    env = live_env
    env.reset()
    gait = env.command_manager.get_term("gait")
    twist = env.command_manager.get_term("twist")
    act = torch.zeros(env.num_envs, 10)
    twist.vel_command_b[0] = torch.tensor([0.3, 0.0, 0.0])
    twist.is_standing_env[0] = False
    twist.is_standing_env[1] = True
    twist.time_left[:] = 100.0
    env.step(act)
    assert gait.stand_time[0] == 0.0
    t1 = float(gait.stand_time[1])
    twist.vel_command_b[0] = 0.0
    twist.is_standing_env[0] = True
    for k in range(3):
        env.step(act)
        assert gait.stand_time[0] == pytest.approx((k + 1) * env.step_dt)
    assert gait.stand_time[1] > t1


def test_contact_schedule_settle_window(live_env):
    """Right after a stop any contact pattern is fine (one settling step);
    after the window both feet must be down again."""
    env = live_env
    _stand(env)
    gait = env.command_manager.get_term("gait")
    sensor = env.scene["feet_ground_contact"]
    sensor.data.found[:, 0] = 0  # pretend the left foot is lifted
    gait.stand_time[:] = 0.1
    assert torch.all(microduck_mdp.gait_contact_schedule(env, "feet_ground_contact", settle_s=0.6) == 1.0)
    gait.stand_time[:] = 1.0
    assert torch.all(microduck_mdp.gait_contact_schedule(env, "feet_ground_contact", settle_s=0.6) == 0.5)
    assert torch.all(microduck_mdp.gait_contact_schedule(env, "feet_ground_contact") == 0.5)


def test_swing_clearance_deficit(live_env):
    env = live_env
    _stand(env)
    feet = _feet(env)
    # standing: nothing swings -> no cost
    assert torch.all(microduck_mdp.gait_swing_clearance_deficit(env, 0.035, asset_cfg=feet) == 0.0)
    # walking at phase 0.2 = left leg at mid-swing while its sole is on the floor
    gait = env.command_manager.get_term("gait")
    gait.active[:] = True
    gait.phase[:, 0] = 0.2
    c = microduck_mdp.gait_swing_clearance_deficit(env, 0.035, asset_cfg=feet)
    assert torch.allclose(c, torch.full_like(c, 0.035), atol=2e-3)
    # half a period later the right leg is the one at mid-swing
    gait.phase[:, 0] = 0.7
    c = microduck_mdp.gait_swing_clearance_deficit(env, 0.035, asset_cfg=feet)
    assert torch.allclose(c, torch.full_like(c, 0.035), atol=2e-3)
    gait.active[:] = False


def test_fine_yaw_tracking(live_env):
    env = live_env
    _stand(env)
    twist = env.command_manager.get_term("twist")
    r0 = microduck_mdp.track_ang_vel_z_exp(env, 0.15, "twist")
    assert torch.all(r0 > 0.9)
    twist.vel_command_b[:, 2] = 0.8
    r1 = microduck_mdp.track_ang_vel_z_exp(env, 0.15, "twist")
    assert torch.all(r1 < 1e-6)
    # 10% short of 0.8 rad/s still loses a fifth of the fine reward
    assert math.exp(-(0.08 / 0.15) ** 2) < 0.8
    twist.vel_command_b[:, 2] = 0.0


def test_stand_square_rewards_square_stance_only(live_env):
    env = live_env
    _stand(env, steps=50)
    feet = _feet(env)
    kw = dict(offset_std=0.02, hip_pitch_std=0.05, load_std=0.15, pose_std=0.2, asset_cfg=feet)
    r_square = microduck_mdp.stand_square_stance(env, "feet_ground_contact", **kw)
    assert torch.all(r_square > 0.8), r_square
    # staggered (mid-stride) stance: hips 0.2 rad apart, feet ~5 cm apart fore-aft
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(("l_hip_pitch", "r_hip_pitch"), preserve_order=True)
    q = robot.data.joint_pos.clone()
    q[:, ids[0]] -= 0.1
    q[:, ids[1]] += 0.1
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    env.sim.forward()
    r_stagger = microduck_mdp.stand_square_stance(env, "feet_ground_contact", **kw)
    assert torch.all(r_stagger < r_square - 0.3), (r_stagger, r_square)
    # walking envs: not paid
    gait = env.command_manager.get_term("gait")
    gait.active[:] = True
    assert torch.all(microduck_mdp.stand_square_stance(env, "feet_ground_contact", **kw) == 0.0)
    gait.active[:] = False
    env.reset()


def test_stand_square_coarse_kernels_keep_a_gradient(live_env):
    """gait4: coarse_scale=1 is the gait3 reward; coarse_scale=3 still pays (and
    grades) a mid-stride stop that the fine kernels alone score ~0."""
    env = live_env
    _stand(env, steps=50)
    feet = _feet(env)
    kw = dict(offset_std=0.02, hip_pitch_std=0.05, load_std=0.15, pose_std=0.2, asset_cfg=feet)
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(("l_hip_pitch", "r_hip_pitch"), preserve_order=True)
    q0 = robot.data.joint_pos.clone()
    env.extras.setdefault("log", {})

    def r_at(split, **extra):
        q = q0.clone()
        q[:, ids[0]] -= split
        q[:, ids[1]] += split
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        env.sim.forward()
        return microduck_mdp.stand_square_stance(env, "feet_ground_contact", **kw, **extra)

    assert torch.allclose(r_at(0.1), r_at(0.1, coarse_scale=1.0))
    fine_far, fine_near = r_at(0.12), r_at(0.08)
    coarse_far, coarse_near = r_at(0.12, coarse_scale=3.0), r_at(0.08, coarse_scale=3.0)
    # the coarse kernels add a clear gradient towards square (with the fine
    # ones only the pose kernel grades 0.24 vs 0.16 rad apart)
    gain_fine, gain_coarse = fine_near - fine_far, coarse_near - coarse_far
    assert torch.all(gain_coarse > gain_fine + 0.05), (gain_coarse, gain_fine)

    assert torch.all(coarse_far > fine_far)
    # square stance still scores high
    assert torch.all(r_at(0.0, coarse_scale=3.0) > 0.8)
    env.reset()


def test_foot_solref_randomization_cfg():
    from mjlab_microduck.tasks.qdd_biped_velocity_env_cfg import FOOT_SOLREF_TIMECONST_RANGE

    train = make_qdd_biped_velocity_env_cfg()
    ev = train.events["foot_solref"]
    assert ev.func is microduck_mdp.geom_solref_timeconst
    assert tuple(ev.params["asset_cfg"].geom_names) == FOOT_GEOMS
    assert ev.params["ranges"] == FOOT_SOLREF_TIMECONST_RANGE == (0.004, 0.020)
    assert "foot_solref" not in make_qdd_biped_velocity_env_cfg(play=True).events
    # foot geoms win the contact-parameter priority against the floor
    m = _model()
    for g in FOOT_GEOMS:
        assert np.allclose(m.geom(g).solref, (0.02, 1.0))


def test_foot_solref_randomization_live():
    """The event writes solref[0] of both feet (shared per env) in range and
    leaves the damping ratio and all other geoms alone."""
    from mjlab.envs import ManagerBasedRlEnv

    cfg = make_qdd_biped_velocity_env_cfg()
    cfg.scene.num_envs = 8
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        env.reset()
        robot = env.scene["robot"]
        sol = env.sim.model.geom_solref  # [num_envs, ngeom, 2]
        m = env.sim.mj_model
        foot = [m.geom(f"robot/{g}").id for g in FOOT_GEOMS]
        tc = sol[:, foot, 0]
        assert torch.all((tc >= 0.004) & (tc <= 0.020)), tc
        assert torch.allclose(tc[:, 0], tc[:, 1])  # one pad material per robot
        assert tc[:, 0].std() > 1e-3  # varies across envs
        assert torch.allclose(sol[:, foot, 1], torch.ones_like(sol[:, foot, 1]))
        others = [i for i in range(m.ngeom) if i not in foot]
        assert torch.allclose(sol[:, others, 0], torch.full_like(sol[:, others, 0], 0.02))
        del robot
    finally:
        env.close()
