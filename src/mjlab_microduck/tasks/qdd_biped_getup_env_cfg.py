"""QDD biped get-up after a fall — Mjlab-Getup-QddBiped.

The robot starts LYING STILL on a flat floor in a pose a real fall leaves it in
(robot/qdd_biped/getup_states.npz from scripts/qdd_getup_states.py: falls with
the controller's fall check -> damping mode, simulated to rest; pushed while
standing, while walking, and odd drops) and has to stand up into the default
walking pose and stay there. 10% of resets start already standing.

On the robot (controller): walking -> fall detected -> damping until the robot
lies still -> this policy -> standing in the default pose -> walking policy.
Its own policy_io (robot/qdd_biped/policy_io_getup.json, kind "getup"): the
walking obs WITHOUT command and gait phase (36-D), action scale GETUP_ACTION_SCALE.

Recipe notes (AGENTS.md "Reward design"):
  - no reward is paid for a bad state: the get-up is driven by potential-based
    shaping (getup_progress: Delta of uprightness + torso height, unfarmable)
    plus a multiplicative standing term (upright x height x feet down), and
    once standing, still-torso and default-pose terms,
  - the path is free (no keyframes); violence is priced by a torso angular-
    velocity cap, body-on-floor impact forces, torque and action-rate costs,
  - the knee_max(hip_pitch) cost of the walking task stays (hardware stop),
  - collision model with friction for every part (robot/qdd_biped_constants.py
    get_qdd_biped_getup_robot_cfg): motor housings and the rabbit top added.
No arms: whether this robot CAN get up from every pose (on its back above all)
is what training finds out; the per-kind success is logged
(Metrics/getup_stood_frac, and scripts/eval_qdd_getup.py).
"""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.robot.qdd_biped_constants import (
    QDD_GETUP_STATES_NPZ,
    STAND_BASE_Z,
    TORSO_BODY,
    get_qdd_biped_getup_robot_cfg,
    getup_scene_spec_fn,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.qdd_biped_velocity_env_cfg import (
    NUM_STEPS_PER_ENV,
    make_qdd_biped_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import QDD_SYMMETRY_CFG, PpoWithSymmetryCfg

TASK_ID = "Mjlab-Getup-QddBiped"

EPISODE_LENGTH_S = 8.0
# Action = offset from the default pose x scale. Getting up needs the whole
# joint range (hip pitch -100 deg, knee 120 deg vs default -22.5 / 45 deg):
# with the walking 0.25 rad that is an action of ~5; 0.5 halves it.
GETUP_ACTION_SCALE = 0.5
STANDING_START_PROB = 0.1

# Standing (getup_standing): torso height at the default pose = STAND_BASE_Z.
STAND_UP_SCALE = 0.05  # exp(-(1 - cos tilt) / s): 10 deg -> 0.74, 20 deg -> 0.30
STAND_Z_STD = 0.06  # m
LYING_Z = 0.05  # m, torso origin height lying flat (getup_progress zero)

# Weights. Progress pays at most ~2 x PROGRESS_WEIGHT over a whole get-up;
# standing pays STANDING_WEIGHT per step (~400 steps of an 8 s episode).
PROGRESS_WEIGHT = 20.0
STANDING_WEIGHT = 3.0
STAND_STILL_WEIGHT = 1.0
STAND_POSE_WEIGHT = 1.5
STAND_POSE_STD = 0.25  # rad, sqrt of the summed squared joint errors

# Violence / hardware.
TORSO_ANG_VEL_LIMIT = 4.0  # rad/s, cost above (falling over is ~3 rad/s)
TORSO_ANG_VEL_WEIGHT = -0.5
BODY_IMPACT_THRESHOLD = 150.0  # N per part; standing on two soles is ~65 N
BODY_IMPACT_WEIGHT = -0.5
BODY_FRICTION_RANGE = (0.3, 1.0)  # PA12 / TPU on tiles .. carpet
KNEE_LIMIT_TERMINATION_TOL = 0.3  # rad (lying poses can sit near the stop)

# Play / export video: these library states in this order, one per env.
PLAY_EPISODE_S = 6.0


def make_qdd_biped_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_qdd_biped_velocity_env_cfg(play=False)

    # --- Scene -------------------------------------------------------------------
    cfg.scene.entities = {"robot": get_qdd_biped_getup_robot_cfg()}
    cfg.scene.spec_fn = getup_scene_spec_fn
    feet_ground_cfg = next(s for s in cfg.scene.sensors if s.name == "feet_ground_contact")
    self_collision_cfg = next(s for s in cfg.scene.sensors if s.name == "self_collision")
    body_floor_cfg = ContactSensorCfg(
        name="body_floor_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(?!(l|r)_foot_collision$).*_collision$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        history_length=4,  # = decimation: peak over the substeps
    )
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, body_floor_cfg)
    cfg.episode_length_s = EPISODE_LENGTH_S
    # Lying robots: many simultaneous contacts (torso + motors + legs).
    cfg.sim.nconmax = 80
    cfg.sim.mujoco.iterations = 20
    cfg.sim.mujoco.ls_iterations = 30

    # --- Actions -------------------------------------------------------------------
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = GETUP_ACTION_SCALE

    # --- Commands: none (the gait clock and twist are walking-only) ----------------
    cfg.commands = {}

    # --- Observations: walking actor without command / gait phase (36-D) ------------
    actor = cfg.observations["actor"].terms
    critic = cfg.observations["critic"].terms
    for name in ("command", "gait_phase"):
        actor.pop(name, None)
        critic.pop(name, None)
    for name in ("foot_height", "foot_air_time"):  # need the foot height sensor / clock
        critic.pop(name, None)
    critic["root_height"] = ObservationTermCfg(func=microduck_mdp.qdd_root_height_obs)
    critic["body_contact"] = ObservationTermCfg(
        func=mdp.foot_contact, params={"sensor_name": body_floor_cfg.name}
    )

    # --- Rewards ---------------------------------------------------------------------
    keep = {"action_rate_l2", "joint_torques", "dof_pos_limits", "self_collisions", "knee_limit"}
    cfg.rewards = {k: v for k, v in cfg.rewards.items() if k in keep}
    cfg.rewards["action_rate_l2"].weight = -0.01  # ramped by the curriculum below
    cfg.rewards["joint_torques"].weight = -1.0e-4
    sense = {"stand_z": STAND_BASE_Z, "sensor_name": feet_ground_cfg.name}
    cfg.rewards["getup_progress"] = RewardTermCfg(
        func=microduck_mdp.getup_progress,
        weight=PROGRESS_WEIGHT,
        params={"stand_z": STAND_BASE_Z, "z_low": LYING_Z},
    )
    cfg.rewards["standing"] = RewardTermCfg(
        func=microduck_mdp.getup_standing,
        weight=STANDING_WEIGHT,
        params={**sense, "up_scale": STAND_UP_SCALE, "z_std": STAND_Z_STD},
    )
    cfg.rewards["stand_still"] = RewardTermCfg(
        func=microduck_mdp.getup_stand_still, weight=STAND_STILL_WEIGHT, params=dict(sense)
    )
    cfg.rewards["stand_pose"] = RewardTermCfg(
        func=microduck_mdp.getup_stand_pose,
        weight=STAND_POSE_WEIGHT,
        params={**sense, "std": STAND_POSE_STD},
    )
    cfg.rewards["torso_ang_vel"] = RewardTermCfg(
        func=microduck_mdp.torso_ang_vel_excess,
        weight=TORSO_ANG_VEL_WEIGHT,
        params={"limit": TORSO_ANG_VEL_LIMIT},
    )
    cfg.rewards["body_impact"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=BODY_IMPACT_WEIGHT,
        params={"sensor_name": body_floor_cfg.name, "threshold": BODY_IMPACT_THRESHOLD},
    )

    # --- Terminations: only the time limit, NaN and the knee stop ---------------------
    cfg.terminations = {
        "time_out": cfg.terminations["time_out"],
        "nan_state": cfg.terminations["nan_state"],
        "knee_limit": TerminationTermCfg(
            func=microduck_mdp.qdd_knee_limit_exceeded,
            params={"tolerance": KNEE_LIMIT_TERMINATION_TOL},
        ),
    }

    # --- Events ----------------------------------------------------------------------
    for name in ("reset_base", "reset_robot_joints", "push_robot"):
        cfg.events.pop(name, None)
    cfg.events = {
        "reset_getup": EventTermCfg(
            func=microduck_mdp.reset_from_getup_states,
            mode="reset",
            params={
                "states_path": str(QDD_GETUP_STATES_NPZ),
                "standing_prob": STANDING_START_PROB,
                "kind_probs": None,
            },
        ),
        **cfg.events,
    }
    cfg.events["body_friction"] = EventTermCfg(
        mode="startup",
        func=deepcopy(cfg.events["foot_friction"].func),
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=(r"^(?!(l|r)_foot_collision$).*_collision$",)),
            "operation": "abs",
            "ranges": BODY_FRICTION_RANGE,
            "shared_random": True,
        },
    )

    # --- Curricula -----------------------------------------------------------------
    cfg.curriculum = {
        # Smoothness only after the get-up exists (AGENTS.md: no attempt-tax
        # while a hard skill is being discovered).
        "action_rate_weight": CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "action_rate_l2",
                "weight_stages": [
                    {"step": 0, "weight": -0.01},
                    {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.03},
                    {"step": 2000 * NUM_STEPS_PER_ENV, "weight": -0.06},
                ],
            },
        ),
    }

    cfg.viewer.body_name = TORSO_BODY
    cfg.viewer.distance = 1.6

    if play:
        _make_play_nominal(cfg)
    return cfg


def _make_play_nominal(cfg: ManagerBasedRlEnvCfg) -> None:
    """Nominal robot, no DR / noise, a fixed library state per env, a reset every
    PLAY_EPISODE_S (the video shows one get-up per episode)."""
    from dataclasses import replace

    cfg.episode_length_s = PLAY_EPISODE_S
    cfg.curriculum = {}
    for name in (
        "base_com",
        "encoder_bias",
        "foot_friction",
        "foot_solref",
        "body_friction",
        "randomize_mass_inertia",
        "randomize_pd_gains",
        "randomize_effort_limits",
        "randomize_armature",
        "randomize_joint_friction",
    ):
        cfg.events.pop(name, None)
    cfg.events["reset_getup"].params["fixed_order"] = True
    actor = cfg.observations["actor"]
    actor.enable_corruption = False
    for term in actor.terms.values():
        term.delay_min_lag = term.delay_max_lag = 0
    for name in ("base_ang_vel", "projected_gravity"):
        if "max_angle_deg" in actor.terms[name].params:
            actor.terms[name].params["max_angle_deg"] = 0.0
    actor.terms["joint_pos"].params["biased"] = False
    robot = cfg.scene.entities["robot"]
    art = robot.articulation
    robot.articulation = replace(
        art, actuators=tuple(replace(a, delay_min_lag=1, delay_max_lag=1) for a in art.actuators)
    )


QddBipedGetupRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=dict(QDD_SYMMETRY_CFG),
    ),
    wandb_project="mjlab_qdd_biped",
    experiment_name="qdd_biped_getup",
    run_name="qdd_biped_getup",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=3_000,
)


# =============================================================================
# Policy I/O contract for the controller (robot/qdd_biped/policy_io_getup.json).
# Regenerate after an obs / action / gain change with
#   uv run python -m mjlab_microduck.tasks.qdd_biped_getup_env_cfg
# tests/test_qdd_getup.py checks the file against this function and the env.
# =============================================================================
GETUP_DONE = {
    # The controller hands over to standing / walking once ALL of these hold
    # for hold_s (the get-up reward's "standing" in numbers).
    "max_tilt_deg": 12.0,
    "max_joint_err_rad": 0.25,
    "max_gyro_rad_s": 0.5,
    "hold_s": 0.5,
}
GETUP_START = {
    # Start only when the robot lies still (the fallen-state library is
    # robots at rest), in damping mode.
    "max_gyro_rad_s": 0.3,
    "still_s": 1.0,
    "min_tilt_deg": 30.0,
}


def build_getup_policy_io() -> dict:
    from mjlab_microduck.tasks.qdd_biped_velocity_env_cfg import _ACTOR_OBS_SPEC, build_policy_io

    io = build_policy_io()
    cfg = make_qdd_biped_getup_env_cfg(play=True)
    terms, start = [], 0
    for name in cfg.observations["actor"].terms:
        spec = _ACTOR_OBS_SPEC[name]
        terms.append({"name": name, "start": start, **spec})
        start += spec["size"]
    io["task"] = TASK_ID
    io["kind"] = "getup"
    io["onnx"]["input"]["shape"] = [1, start]
    io["action"]["scale"] = float(GETUP_ACTION_SCALE)
    io["observation"]["dim"] = start
    io["observation"]["terms"] = terms
    io.pop("gait_clock", None)
    io.pop("commands", None)
    io["getup"] = {
        "episode_length_s": EPISODE_LENGTH_S,
        "start": dict(GETUP_START),
        "done": dict(GETUP_DONE),
        "note": "trained from resting poses after falls (fall check -> damping), flat floor only; "
        "last action = zeros at the start; hand over to the walking policy in the default pose",
    }
    return io


def getup_policy_io_json() -> str:
    import json

    return json.dumps(build_getup_policy_io(), indent=2) + "\n"


def write_getup_policy_io(path=None) -> str:
    from pathlib import Path

    from mjlab_microduck.robot.qdd_biped_constants import QDD_GETUP_POLICY_IO_JSON

    path = Path(path) if path is not None else QDD_GETUP_POLICY_IO_JSON
    path.write_text(getup_policy_io_json(), encoding="utf-8", newline="\n")
    return str(path)


def getup_policy_io_onnx_metadata() -> dict[str, str]:
    import hashlib

    from mjlab_microduck.robot.qdd_biped_constants import QDD_GETUP_POLICY_IO_JSON

    text = QDD_GETUP_POLICY_IO_JSON.read_text(encoding="utf-8")
    return {"policy_io": text, "policy_io_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


if __name__ == "__main__":
    print("written", write_getup_policy_io())
