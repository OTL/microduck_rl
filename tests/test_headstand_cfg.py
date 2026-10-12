import mujoco

from mjlab_microduck.robot.microduck_constants import MICRODUCK_GROUNDCONTACT_XML
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_headstand_env_cfg import (
    HOLD_S,
    NON_HEAD_GROUND_BODIES,
    MicroduckHeadstandRlCfg,
    make_microduck_headstand_env_cfg,
)
from mjlab_microduck.tasks.microduck_roulade_env_cfg import MicroduckRouladeRlCfg


def _body_names():
    model = mujoco.MjModel.from_xml_path(str(MICRODUCK_GROUNDCONTACT_XML))
    return {model.body(i).name for i in range(model.nbody)}


def test_cfg_has_the_headstand_rewards_and_no_roulade_task_terms():
    cfg = make_microduck_headstand_env_cfg()
    for name in (
        "headstand_progress",
        "headstand_hold",
        "headstand_stand_composite",
        "headstand_upright_after_hold",
        "headstand_stand_sharp",
        "headstand_not_standing_tax",
    ):
        assert name in cfg.rewards, name
        assert cfg.rewards[name].params["hold_s"] == HOLD_S
    for name in ("roulade_progress", "roulade_landing_composite", "roulade_stand_tax"):
        assert name not in cfg.rewards, name


def test_penalty_sign_conventions():
    cfg = make_microduck_headstand_env_cfg()
    # self-negating penalties → POSITIVE weight
    assert cfg.rewards["headstand_not_standing_tax"].weight > 0.0
    assert cfg.rewards["gentle_landing"].weight > 0.0
    # positive-cost functions → negative weight
    assert cfg.rewards["action_rate_l2"].weight < 0.0
    assert cfg.rewards["roulade_flatness"].weight < 0.0


def test_flops_are_net_negative_after_the_hold():
    # Lying down / staying inverted keeps at most half of the upright layer
    # and pays the full tax: the tax must outweigh it.
    cfg = make_microduck_headstand_env_cfg()
    upright = cfg.rewards["headstand_upright_after_hold"].weight
    tax = cfg.rewards["headstand_not_standing_tax"].weight
    assert 0.5 * upright < tax


def test_spawn_event_replaces_the_roulade_spawn():
    cfg = make_microduck_headstand_env_cfg()
    assert "set_roulade_state" not in cfg.events
    assert "roulade_spawn_mix" not in cfg.curriculum
    event = cfg.events["set_headstand_state"]
    assert event.func is microduck_mdp.reset_headstand_state
    assert cfg.curriculum["headstand_spawn_mix"].params["event_name"] == "set_headstand_state"


def test_play_starts_standing_only():
    cfg = make_microduck_headstand_env_cfg(play=True)
    params = cfg.events["set_headstand_state"].params
    assert params["standing_prob"] == 1.0
    assert params["prehold_prob"] == 0.0 and params["posthold_prob"] == 0.0
    assert "headstand_spawn_mix" not in cfg.curriculum


def test_sensor_and_foot_bodies_exist_on_the_model():
    names = _body_names()
    for body in NON_HEAD_GROUND_BODIES + ("ankle_left", "ankle_right", "jaw_soft"):
        assert body in names, body
    cfg = make_microduck_headstand_env_cfg()
    sensor_names = {s.name for s in cfg.scene.sensors}
    for name in (
        microduck_mdp._HS_HEAD_SENSOR,
        microduck_mdp._HS_FEET_SENSOR,
        microduck_mdp._HS_BODY_SENSOR,
    ):
        assert name in sensor_names, name


def test_runner_cfg_is_distinct():
    assert MicroduckHeadstandRlCfg.experiment_name == "microduck_headstand"
    assert MicroduckHeadstandRlCfg.experiment_name != MicroduckRouladeRlCfg.experiment_name
