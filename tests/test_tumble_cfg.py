from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    make_microduck_roulade_env_cfg,
)
from mjlab_microduck.tasks.microduck_tumble_env_cfg import (
    make_microduck_tumble_env_cfg,
    MicroduckTumbleRlCfg,
)


def test_cfg_has_the_tumble_rewards_with_positive_weights():
    cfg = make_microduck_tumble_env_cfg()
    assert cfg.rewards["tumble_progress"].func is microduck_mdp.tumble_progress
    assert cfg.rewards["tumble_progress"].weight > 0.0
    assert cfg.rewards["tumble_forward_travel"].weight > 0.0
    # the progress term dominates the travel term
    assert cfg.rewards["tumble_progress"].weight > cfg.rewards["tumble_forward_travel"].weight


def test_episodic_roulade_terms_are_removed():
    # a perpetual tumble has no landing: any completion-gated standing reward
    # or speed tax left over from the roulade would fight "keep rolling, fast"
    cfg = make_microduck_tumble_env_cfg()
    for name in (
        "roulade_progress",
        "roulade_overspeed",
        "roulade_landing_composite",
        "roulade_landing_sharp",
        "roulade_stand_tax",
        "arrival_damping",
    ):
        assert name not in cfg.rewards, name
    assert "arrival_damping_weight" not in cfg.curriculum


def test_penalty_sign_conventions():
    cfg = make_microduck_tumble_env_cfg()
    # positive-cost functions → negative weight
    for name in ("roulade_sagittal", "roulade_lateral_vel", "roulade_flatness", "action_rate_l2"):
        assert cfg.rewards[name].weight < 0.0, name
    # self-negating |a_z| penalty → positive weight
    assert cfg.rewards["gentle_landing"].weight > 0.0
    for stage in cfg.curriculum["action_rate_weight"].params["weight_stages"]:
        assert stage["weight"] < 0.0


def test_tumble_reset_runs_after_the_roulade_spawn():
    # reset_tumble_state re-anchors on the accumulator set_roulade_state wrote
    cfg = make_microduck_tumble_env_cfg()
    names = list(cfg.events.keys())
    assert names.index("reset_tumble_state") > names.index("set_roulade_state")


def test_falling_over_does_not_terminate():
    cfg = make_microduck_tumble_env_cfg()
    assert "fell_over" not in cfg.terminations
    assert "nan_state" in cfg.terminations


def test_obs_parity_with_roulade():
    tumble = make_microduck_tumble_env_cfg()
    roulade = make_microduck_roulade_env_cfg()
    for grp in ("actor", "critic"):
        assert list(tumble.observations[grp].terms.keys()) == list(
            roulade.observations[grp].terms.keys()
        ), grp


def test_play_starts_from_standing():
    cfg = make_microduck_tumble_env_cfg(play=True)
    params = cfg.events["set_roulade_state"].params
    assert params["standing_prob"] == 1.0 and params["midroll_prob"] == 0.0
    assert "roulade_spawn_mix" not in cfg.curriculum


def test_runner_cfg_is_distinct():
    assert MicroduckTumbleRlCfg.experiment_name == "microduck_tumble"
