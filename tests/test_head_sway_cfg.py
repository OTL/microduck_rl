from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_head_sway_env_cfg import (
    TWIST_TINY,
    make_microduck_head_sway_env_cfg,
    MicroduckHeadSwayRlCfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


def test_obs_layout_matches_velocity_family():
    sway = make_microduck_head_sway_env_cfg()
    vel = make_microduck_velocity_env_cfg()
    for grp in ("actor", "critic"):
        assert list(sway.observations[grp].terms.keys()) == list(
            vel.observations[grp].terms.keys()
        ), grp


def test_head_sway_rewards_and_signs():
    cfg = make_microduck_head_sway_env_cfg()
    assert cfg.rewards["head_sway_track"].weight > 0.0
    assert cfg.rewards["head_sway_speed"].weight > 0.0
    # mjlab-style costs (≥ 0) → negative weights
    assert cfg.rewards["neck_joint_pos_l2"].weight < 0.0
    assert cfg.rewards["no_stepping"].weight < 0.0
    assert "head_yaw" not in cfg.rewards["neck_joint_pos_l2"].params["pattern"]


def test_locomotion_and_head_command_terms_removed():
    cfg = make_microduck_head_sway_env_cfg()
    for name in ("air_time", "foot_clearance", "foot_swing_height",
                 "angular_momentum", "head_pose_tracking", "head_pose_bias"):
        assert name not in cfg.rewards, name
    for name in ("standing_envs", "head_pose_range", "head_pose_bias_weight"):
        assert name not in cfg.curriculum, name


def test_twist_command_stays_below_standing_gates():
    cfg = make_microduck_head_sway_env_cfg()
    r = cfg.commands["twist"].ranges
    worst = (2 * TWIST_TINY**2) ** 0.5 + TWIST_TINY
    assert worst < 0.01
    assert max(abs(v) for v in r.lin_vel_x + r.lin_vel_y + r.ang_vel_z) == TWIST_TINY
    assert cfg.commands["twist"].rel_turn_in_place_envs == 0.0


def test_speed_bootstrap_fades_out():
    cfg = make_microduck_head_sway_env_cfg()
    stages = cfg.curriculum["head_sway_speed_weight"].params["weight_stages"]
    weights = [s["weight"] for s in stages]
    assert weights[0] == cfg.rewards["head_sway_speed"].weight
    assert weights == sorted(weights, reverse=True)
    assert weights[-1] == 0.0


def test_slow_target():
    # "ゆっくり": peak head speed well under 1.5 rad/s
    import math
    peak = microduck_mdp.HEAD_SWAY_AMPLITUDE * 2 * math.pi / microduck_mdp.HEAD_SWAY_PERIOD
    assert peak < 1.5


def test_symmetry_off_and_distinct_experiment():
    assert MicroduckHeadSwayRlCfg.algorithm.symmetry_cfg is None
    assert MicroduckHeadSwayRlCfg.experiment_name == "head_sway"
