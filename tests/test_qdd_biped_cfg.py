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
    assert abs(m.body_subtreemass[1] - 6.479) < 0.01  # robot repo model: strengthened, GIM3510-8 ankle, cable covers


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
