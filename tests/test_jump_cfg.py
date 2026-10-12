"""Cfg invariants for the Jump task (CPU, no sim build)."""

import inspect

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_jump_env_cfg as jump


def _cfg():
    return jump.make_microduck_jump_env_cfg()


def test_jump_is_paid_per_jump_peak_not_per_airborne_step():
    cfg = _cfg()
    assert cfg.rewards["jump_foot_clearance"].func is microduck_mdp.jump_foot_clearance_peak
    assert cfg.rewards["jump_top_height"].func is microduck_mdp.jump_top_clearance_peak
    assert cfg.rewards["jump_pushoff"].func is microduck_mdp.jump_pushoff_power
    # No floating-only bonus anywhere.
    for term in cfg.rewards.values():
        assert "hop_bonus" not in (term.params or {})
    # The per-jump frontier must be cleared on reset.
    assert cfg.events["reset_jump_state"].func is microduck_mdp.reset_jump_state


def test_jump_terms_share_the_gate_and_use_the_measured_reference():
    cfg = _cfg()
    sensor_names = {s.name for s in cfg.scene.sensors}
    foot = cfg.rewards["jump_foot_clearance"]
    top = cfg.rewards["jump_top_height"]
    for term in (foot, top, cfg.rewards["jump_pushoff"]):
        assert term.weight > 0.0
        assert term.params["sensor_name"] == "robot_ground_contact"
        assert term.params["sensor_name"] in sensor_names
    # Both peak terms drive ONE state machine → identical shared params.
    shared = set(foot.params) - {"height_scale"}
    assert shared == set(top.params) - {"height_scale"}
    for key in shared:
        assert foot.params[key] == top.params[key]
    assert top.params["top_ref_height"] == 0.236
    assert foot.params["foot_body_names"] == ("ankle_left", "ankle_right")
    assert 0.0 < foot.params["stroke_base"] < 1.0


def test_jump_params_match_function_signatures():
    cfg = _cfg()
    for name in ("jump_foot_clearance", "jump_top_height", "jump_pushoff"):
        term = cfg.rewards[name]
        accepted = set(inspect.signature(term.func).parameters)
        assert set(term.params) <= accepted, name


def test_penalty_sign_conventions():
    cfg = _cfg()
    # Self-negating (returns -|a_z|) → positive weight, incl. every curriculum stage.
    assert cfg.rewards["gentle_landing"].weight > 0.0
    stages = cfg.curriculum["gentle_landing_weight"].params["weight_stages"]
    assert all(s["weight"] > 0.0 for s in stages)
    # mjlab-style costs (return >= 0) → negative weight.
    for name in ("action_rate_l2", "body_ang_vel", "angular_momentum", "self_collisions"):
        assert cfg.rewards[name].weight < 0.0
    for cur in ("action_rate_weight", "torque_rate_weight"):
        assert all(s["weight"] <= 0.0 for s in cfg.curriculum[cur].params["weight_stages"])


def test_walking_terms_removed_and_fall_termination_kept():
    cfg = _cfg()
    for name in ("track_linear_velocity", "track_angular_velocity", "air_time", "pose"):
        assert name not in cfg.rewards
    assert "fell_over" in cfg.terminations
    assert "nan_state" in cfg.terminations
    assert "expand_bam_friction_fields" in cfg.events


def test_command_block_is_zero_padded_to_13d():
    cfg = _cfg()
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        assert terms["head_command"].params["dim"] == 4
        assert terms["body_command"].params["dim"] == 6
    assert "base_lin_vel" not in cfg.observations["actor"].terms


def test_runner_cfg_is_distinct():
    assert jump.MicroduckJumpRlCfg.experiment_name == "microduck_jump"
