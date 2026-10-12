"""Cfg + mdp invariants for the headstand task.

CPU only: builds cfgs and compiles the MuJoCo model, never steps an env.
"""
import re

import mujoco
import numpy as np
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab_microduck.robot.microduck_constants import HOME_FRAME, get_allcollisions_spec
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_headstand_env_cfg as hs
from mjlab_microduck.tasks.microduck_roulade_env_cfg import make_microduck_roulade_env_cfg


def _servo_names(model):
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
    return [
        n for j, n in enumerate(names)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and not n.startswith("passive_")
    ]


def _home(names):
    out = []
    for n in names:
        out.append(next(v for pat, v in HOME_FRAME.joint_pos.items() if re.fullmatch(pat, n)))
    return np.array(out)


# ── Rewards / terminations ───────────────────────────────────────────────────

def test_no_standing_reward_or_fall_termination():
    cfg = hs.make_microduck_headstand_env_cfg()
    for name in ("upright", "pose", "track_linear_velocity", "air_time"):
        assert name not in cfg.rewards, name
    assert "fell_over" not in cfg.terminations
    assert "headstand_lost" in cfg.terminations
    assert "nan_state" in cfg.terminations


def test_reward_signs():
    cfg = hs.make_microduck_headstand_env_cfg()
    assert cfg.rewards["headstand_hold"].weight > 0.0
    assert cfg.rewards["headstand_progress"].weight > 0.0
    # mjlab-style costs (>= 0) → negative weight
    for name in ("head_impact", "action_rate_l2", "body_ang_vel", "angular_momentum", "dof_pos_limits"):
        assert cfg.rewards[name].weight < 0.0, name
    assert cfg.rewards["joint_torque_rate_l2"].weight <= 0.0
    # self-negating penalty → positive weight
    assert cfg.rewards["gentle_motion"].func is microduck_mdp.trunk_vertical_accel_penalty
    assert cfg.rewards["gentle_motion"].weight > 0.0
    for name, key in (
        ("torque_rate_weight", -1.0), ("head_impact_weight", -1.0),
        ("action_rate_weight", -1.0), ("gentle_motion_weight", 1.0),
    ):
        for stage in cfg.curriculum[name].params["weight_stages"]:
            assert stage["weight"] * key >= 0.0, (name, stage)


# ── Spawns ───────────────────────────────────────────────────────────────────

def test_on_head_spawns_are_the_majority_at_every_stage():
    cfg = hs.make_microduck_headstand_env_cfg()
    params = cfg.events["set_headstand_state"].params
    assert params["headstand_prob"] > params["standing_prob"] > 0.0
    for stage in cfg.curriculum["headstand_spawn_mix"].params["param_stages"]:
        p = stage["params"]
        assert p["headstand_prob"] > p["standing_prob"] > 0.0


def test_play_starts_standing_like_deployment():
    cfg = hs.make_microduck_headstand_env_cfg(play=True)
    params = cfg.events["set_headstand_state"].params
    assert params["standing_prob"] == 1.0 and params["headstand_prob"] == 0.0
    assert "headstand_spawn_mix" not in cfg.curriculum


def test_spawn_table_points_the_head_top_down():
    model = get_allcollisions_spec().compile()
    names = _servo_names(model)
    assert len(names) == 14
    tab = microduck_mdp.headstand_spawn_table(
        model, names, _home(names), hs.HEADSTAND_LEG_TUCK, neck_grid=5, tuck_factors=(0.0, 1.0),
    )
    k = len(tab["joints"])
    assert k >= 1
    assert tab["joints"].shape == (k, 14)
    assert (tab["joints"] >= tab["lo"] - 1e-9).all() and (tab["joints"] <= tab["hi"] + 1e-9).all()
    assert tab["leg_mask"].sum() == 10
    for i in range(k):
        assert abs(np.linalg.norm(tab["quat"][i]) - 1.0) < 1e-9
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, tab["quat"][i])
        axis_world = rot.reshape(3, 3) @ (tab["head_rot"][i] @ tab["top_axis"])
        assert np.allclose(axis_world, (0.0, 0.0, -1.0), atol=1e-6)


# ── Potential ────────────────────────────────────────────────────────────────

def test_potential_is_zero_standing_and_one_on_the_head():
    t = torch.tensor
    standing = microduck_mdp.headstand_potential(t([-1.0]), t([False]), t([0.0]))
    mounted = microduck_mdp.headstand_potential(t([1.0]), t([True]), t([1.0]))
    tripod = microduck_mdp.headstand_potential(t([1.0]), t([True]), t([0.5]))
    beak = microduck_mdp.headstand_potential(t([0.0]), t([True]), t([1.0]))
    assert standing.item() == 0.0
    assert abs(mounted.item() - 1.0) < 1e-6
    assert standing.item() < tripod.item() < mounted.item()
    # head loaded but the top not facing down: only the orientation share
    assert abs(beak.item() - 0.15) < 1e-6


# ── Contract ─────────────────────────────────────────────────────────────────

def test_actor_obs_layout_matches_roulade():
    cfg = hs.make_microduck_headstand_env_cfg()
    ref = make_microduck_roulade_env_cfg()
    assert list(cfg.observations["actor"].terms) == list(ref.observations["actor"].terms)
    assert cfg.observations["actor"].terms["head_command"].params["dim"] == 4
    assert cfg.observations["actor"].terms["body_command"].params["dim"] == 6
    assert "headstand_state" not in cfg.observations["actor"].terms
    assert "headstand_state" in cfg.observations["critic"].terms


def test_only_head_gate_sensors():
    cfg = hs.make_microduck_headstand_env_cfg()
    sensors = {s.name: s for s in cfg.scene.sensors}
    assert sensors["head_ground_contact"].primary.pattern == "jaw_soft"
    assert "jaw_soft" in sensors["other_ground_contact"].primary.exclude
    for name in ("head_ground_contact", "other_ground_contact"):
        assert {"found", "force"} <= set(sensors[name].fields)


def test_runner_has_its_own_experiment_name():
    assert hs.MicroduckHeadstandRlCfg.experiment_name == "microduck_headstand"
