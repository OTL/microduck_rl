"""Cfg / model invariants for Mjlab-Velocity-Flat-QddBiped (CPU, plain MuJoCo)."""

import math

import mujoco
import numpy as np
import torch

# Import the tasks package first (as mjlab's plugin loader does) to avoid a
# circular import through mjlab -> mjlab_microduck.tasks -> qdd_biped_constants.
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.qdd_biped_velocity_env_cfg import (
    QddBipedRlCfg,
    make_qdd_biped_velocity_env_cfg,
)
from mjlab_microduck.robot.qdd_biped_constants import (  # noqa: E402
    DEFAULT_JOINT_POS,
    FOOT_GEOMS,
    FOOT_SITES,
    QDD_JOINT_NAMES,
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
    assert math.isclose(knee_max(math.radians(10)), math.radians(85))
    assert math.isclose(knee_max(math.radians(-50)), math.radians(120))


def test_default_pose_is_inside_knee_limit():
    hp = DEFAULT_JOINT_POS[r".*_hip_pitch"]
    assert DEFAULT_JOINT_POS[r".*_knee"] < knee_max(hp) - 0.05


def test_reward_signs():
    cfg = make_qdd_biped_velocity_env_cfg()
    r = cfg.rewards
    for name in ("track_linear_velocity", "track_angular_velocity", "upright", "pose", "air_time"):
        assert r[name].weight > 0, name
    # mjlab-style COSTS (>= 0) must have negative weights.
    for name in (
        "knee_limit",
        "self_collisions",
        "joint_torques",
        "action_rate_l2",
        "foot_slip",
        "foot_clearance",
        "body_ang_vel",
        "dof_pos_limits",
    ):
        assert r[name].weight < 0, name


def test_actor_obs_has_no_privileged_terms():
    cfg = make_qdd_biped_velocity_env_cfg()
    actor = cfg.observations["actor"].terms
    assert list(actor) == ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command"]
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
