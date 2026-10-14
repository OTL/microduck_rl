"""QDD biped velocity (walking) environment — Mjlab-Velocity-Flat-QddBiped.

A different robot from the microduck: 5.66 kg, ~0.52 m tall, 10 quasi-direct-
drive joints ({l,r}_{hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch}), no ankle
roll (lateral balance must be dynamic, like Disney BDX / Open Duck Mini), flat
150 x 70 mm feet centred under the ankle. Robot model:
robot/qdd_biped_constants.py + robot/qdd_biped/qdd_biped.xml.

Recipe = the microduck velocity recipe (microduck_velocity_env_cfg.py), ported
to this robot, keeping its sim2real choices:
  - actor obs = what the real robot has: IMU gyro + projected gravity (with a
    per-env IMU mounting misalignment and 0-1 step delay), joint pos (with
    per-env encoder bias) / vel, last action, twist command. NO base lin vel in
    the actor (critic-only, privileged), no height scan.
  - unfiltered actions (no EMA), obs normalisation baked into the ONNX by
    scripts/export.py.
  - fixed modest command ranges + explicit standing (curriculum-ramped) and
    turn-in-place buckets, tracking + upright + variable-posture + gait/feet
    terms, curriculum-ramped action-rate smoothing.
Differences from microduck (different robot, not part of the 61-D microduck
policy family, so the actor obs is 3+3+10+10+10+3 = 39-D, no head/body slots):
  - built-in MuJoCo position PD actuators (the drivers run PD in MIT mode)
    instead of BAM; Kp/Kd/effort limits per motor in qdd_biped_constants.py,
    with PD-gain / effort-limit DR and 0-10 ms command latency,
  - pose-dependent knee limit knee_max(hip_pitch) from robot/model.py: cost
    term + termination,
  - falls terminate on torso tilt > 60 deg, torso height < 0.20 m or any
    non-foot ground contact.
"""

import math
from copy import deepcopy

NUM_STEPS_PER_ENV = 24

TURN_IN_PLACE_FRACTION = 0.15

# Commands (0.5 m robot, 0.13 m links)
LIN_VEL_X_RANGE = (-0.3, 0.5)
LIN_VEL_Y_RANGE = (-0.2, 0.2)
ANG_VEL_Z_RANGE = (-1.0, 1.0)

# Domain randomization
ENABLE_COM_RANDOMIZATION = True
ENABLE_MASS_INERTIA_RANDOMIZATION = True
ENABLE_PD_GAIN_RANDOMIZATION = True
ENABLE_EFFORT_LIMIT_RANDOMIZATION = True
ENABLE_ARMATURE_RANDOMIZATION = True
ENABLE_JOINT_FRICTION_RANDOMIZATION = True
ENABLE_VELOCITY_PUSHES = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS = True

COM_RANDOMIZATION_RANGE = {0: (-0.02, 0.02), 1: (-0.015, 0.015), 2: (-0.02, 0.02)}  # torso, m
MASS_INERTIA_RANDOMIZATION_RANGE = (0.9, 1.1)  # torso mass+inertia scale (battery/electronics)
KP_RANDOMIZATION_RANGE = (0.85, 1.15)
KD_RANDOMIZATION_RANGE = (0.8, 1.2)
EFFORT_LIMIT_RANDOMIZATION_RANGE = (0.8, 1.05)  # battery sag / driver current limit
ARMATURE_RANDOMIZATION_RANGE = (0.8, 1.2)  # rotor inertia is an estimate
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.5, 2.0)  # x XML frictionloss 0.05 Nm
FOOT_FRICTION_RANGE = (0.4, 1.2)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)
VELOCITY_PUSH_RANGE = (-0.3, 0.3)  # m/s, x and y
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 3.0  # deg, random axis, constant per env
ENCODER_BIAS_RANGE = (-0.01, 0.01)  # rad; absolute encoders on the QDD output

# Knee limit
KNEE_LIMIT_MARGIN = 0.05  # rad: cost starts 3 deg before knee_max(hip_pitch)
KNEE_LIMIT_TERMINATION_TOL = 0.15  # rad: terminate 8.6 deg past knee_max

# Gait
FOOT_TARGET_HEIGHT = 0.04  # m swing clearance (microduck 0.02 at half the size)
AIR_TIME_WINDOW = (0.15, 0.45)  # s

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
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
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.qdd_biped_constants import (
    FOOT_GEOMS,
    FOOT_SITES,
    QDD_ACTION_SCALE,
    TORSO_BODY,
    get_qdd_biped_robot_cfg,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg


def make_qdd_biped_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Flat-ground velocity tracking for the QDD biped."""

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=r"^(l_foot_collision|r_foot_collision)$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern=TORSO_BODY, entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern=TORSO_BODY, entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    # Anything but the soles touching the ground = fall (shins/knees included).
    body_ground_cfg = ContactSensorCfg(
        name="body_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(torso_collision|torso_deck_collision|(l|r)_(thigh|knee_motor|shank)_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    foot_height_scan_cfg = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITES),
        pattern=RingPatternCfg.single_ring(radius=0.06, num_samples=4),
        ray_alignment="yaw",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=False,
    )

    cfg = make_velocity_env_cfg()

    # --- Scene -----------------------------------------------------------------
    cfg.scene.entities = {"robot": get_qdd_biped_robot_cfg()}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, body_ground_cfg, foot_height_scan_cfg)
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.viewer.body_name = TORSO_BODY
    cfg.viewer.distance = 1.5

    # --- Actions -----------------------------------------------------------------
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = QDD_ACTION_SCALE

    # --- Commands ----------------------------------------------------------------
    command: UniformVelocityCommandCfg = deepcopy(cfg.commands["twist"])
    command.rel_standing_envs = 0.05  # ramped by the standing_envs curriculum
    command.rel_heading_envs = 0.0
    command.ranges.lin_vel_x = LIN_VEL_X_RANGE
    command.ranges.lin_vel_y = LIN_VEL_Y_RANGE
    command.ranges.ang_vel_z = ANG_VEL_Z_RANGE
    command.viz.z_offset = 0.6
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))
    cfg.commands["twist"].rel_turn_in_place_envs = TURN_IN_PLACE_FRACTION

    # --- Rewards -----------------------------------------------------------------
    cfg.rewards["track_linear_velocity"].weight = 2.0
    cfg.rewards["track_linear_velocity"].params["std"] = 0.25  # m/s
    cfg.rewards["track_angular_velocity"].weight = 2.0
    cfg.rewards["track_angular_velocity"].params["std"] = 0.5  # rad/s

    cfg.rewards["upright"].params["asset_cfg"].body_names = (TORSO_BODY,)
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)

    # Posture around the 45 deg crouch. hip_roll is NOT tight while walking: with
    # no ankle roll the lateral weight shift has to come from the hips.
    cfg.rewards["pose"].params["std_standing"] = {
        r".*hip_yaw.*": 0.1,
        r".*hip_roll.*": 0.08,
        r".*hip_pitch.*": 0.15,
        r".*knee.*": 0.15,
        r".*ankle.*": 0.1,
    }
    std_walking = {
        r".*hip_yaw.*": 0.25,
        r".*hip_roll.*": 0.15,
        r".*hip_pitch.*": 0.4,
        r".*knee.*": 0.4,
        r".*ankle.*": 0.3,
    }
    cfg.rewards["pose"].params["std_walking"] = std_walking
    cfg.rewards["pose"].params["std_running"] = std_walking
    cfg.rewards["pose"].params["walking_threshold"] = 0.05
    cfg.rewards["pose"].weight = 1.0

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (TORSO_BODY,)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02

    for name in ("foot_clearance", "foot_slip"):
        cfg.rewards[name].params["asset_cfg"].site_names = FOOT_SITES
    cfg.rewards["foot_clearance"].params["target_height"] = FOOT_TARGET_HEIGHT
    cfg.rewards["foot_clearance"].params["command_threshold"] = 0.05
    cfg.rewards["foot_swing_height"].params["target_height"] = FOOT_TARGET_HEIGHT
    cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.05
    cfg.rewards["foot_slip"].weight = -0.2
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.05

    cfg.rewards["air_time"].weight = 2.0
    cfg.rewards["air_time"].params["command_threshold"] = 0.05
    cfg.rewards["air_time"].params["threshold_min"] = AIR_TIME_WINDOW[0]
    cfg.rewards["air_time"].params["threshold_max"] = AIR_TIME_WINDOW[1]

    cfg.rewards["action_rate_l2"].weight = -0.05  # ramped by curriculum below

    # Torques: 10 joints at ~3 Nm rms -> ~90 Nm^2 -> -0.018/step.
    cfg.rewards["joint_torques"] = RewardTermCfg(func=mdp.joint_torques_l2, weight=-2.0e-4)

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
    )

    # Pose-dependent knee limit (robot/model.py knee_max). COST (>= 0) -> negative
    # weight. 0.1 rad overshoot costs 1.0/step, comparable to a tracking term.
    cfg.rewards["knee_limit"] = RewardTermCfg(
        func=microduck_mdp.qdd_knee_limit_cost,
        weight=-10.0,
        params={"margin": KNEE_LIMIT_MARGIN},
    )

    # --- Terminations ------------------------------------------------------------
    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(60.0)
    cfg.terminations["torso_low"] = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.20},  # standing 0.344, full 120 deg crouch ~0.23
    )
    cfg.terminations["body_ground_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": body_ground_cfg.name},
    )
    cfg.terminations["knee_limit"] = TerminationTermCfg(
        func=microduck_mdp.qdd_knee_limit_exceeded,
        params={"tolerance": KNEE_LIMIT_TERMINATION_TOL},
    )
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": (feet_ground_cfg.name,)},
    )
    cfg.terminations.pop("out_of_terrain_bounds", None)

    # --- Events / domain randomization -------------------------------------------
    # Spawn: the default root pos already has the soles on the floor (STAND_BASE_Z).
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.005)
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOMS
    cfg.events["foot_friction"].params["ranges"] = FOOT_FRICTION_RANGE

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["base_com"].params["asset_cfg"].body_names = (TORSO_BODY,)
        cfg.events["base_com"].params["ranges"] = COM_RANDOMIZATION_RANGE
    else:
        cfg.events.pop("base_com", None)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
    else:
        cfg.events.pop("encoder_bias", None)

    if ENABLE_VELOCITY_PUSHES and not play:
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=VELOCITY_PUSH_INTERVAL_S,
            params={
                "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
    else:
        cfg.events.pop("push_robot", None)

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        lo, hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(TORSO_BODY,)),
                "alpha_range": (math.log(lo) / 2.0, math.log(hi) / 2.0),
            },
        )

    if ENABLE_PD_GAIN_RANDOMIZATION:
        cfg.events["randomize_pd_gains"] = EventTermCfg(
            func=dr.pd_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "kp_range": KP_RANDOMIZATION_RANGE,
                "kd_range": KD_RANDOMIZATION_RANGE,
                "operation": "scale",
            },
        )

    if ENABLE_EFFORT_LIMIT_RANDOMIZATION:
        cfg.events["randomize_effort_limits"] = EventTermCfg(
            func=dr.effort_limits,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "effort_limit_range": EFFORT_LIMIT_RANDOMIZATION_RANGE,
                "operation": "scale",
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=dr.joint_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    # --- Observations ------------------------------------------------------------
    actor = cfg.observations["actor"].terms
    critic = cfg.observations["critic"].terms
    del actor["base_lin_vel"]  # not measurable on the robot
    del actor["height_scan"]
    del critic["height_scan"]
    critic["base_lin_vel"] = ObservationTermCfg(func=mdp.base_lin_vel)  # privileged

    # Actor and critic share term objects from the template: copy before mutating.
    for name in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
        actor[name] = deepcopy(actor[name])
        critic[name] = deepcopy(critic[name])

    # IMU path (BNO085 / ICM-42688 on the Pi): 0-1 policy-step delay.
    for name in ("base_ang_vel", "projected_gravity"):
        actor[name].delay_min_lag = 0
        actor[name].delay_max_lag = 1
        actor[name].delay_update_period = 64
    actor["base_ang_vel"].noise = Unoise(n_min=-0.05, n_max=0.05)
    actor["projected_gravity"].noise = Unoise(n_min=-0.02, n_max=0.02)
    actor["joint_pos"].noise = Unoise(n_min=-0.005, n_max=0.005)
    actor["joint_vel"].noise = Unoise(n_min=-0.3, n_max=0.3)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        actor["base_ang_vel"].func = microduck_mdp.base_ang_vel_imu_misaligned
        actor["base_ang_vel"].params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        actor["projected_gravity"].func = microduck_mdp.projected_gravity_imu_misaligned
        actor["projected_gravity"].params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # Encoder bias is what the policy sees; the critic keeps the true joint pos.
    actor["joint_pos"].params["biased"] = ENABLE_ENCODER_BIAS
    critic["joint_pos"].params["biased"] = False

    for term, safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_height", microduck_mdp.foot_height_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if term in critic:
            critic[term] = deepcopy(critic[term])
            critic[term].func = safe

    # --- Curricula ---------------------------------------------------------------
    cfg.curriculum.pop("terrain_levels", None)
    cfg.curriculum.pop("command_vel", None)
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.05},
                {"step": 500 * NUM_STEPS_PER_ENV, "weight": -0.1},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.2},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -0.3},
            ],
        },
    )
    cfg.curriculum["standing_envs"] = CurriculumTermCfg(
        func=microduck_mdp.standing_envs_curriculum,
        params={
            "command_name": "twist",
            "standing_stages": [
                {"step": 0, "rel_standing_envs": 0.05},
                {"step": 500 * NUM_STEPS_PER_ENV, "rel_standing_envs": 0.1},
                {"step": 1000 * NUM_STEPS_PER_ENV, "rel_standing_envs": 0.15},
                {"step": 1500 * NUM_STEPS_PER_ENV, "rel_standing_envs": 0.2},
            ],
        },
    )

    # --- Play mode (export / video) ------------------------------------------------
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum = {}
        cfg.commands["twist"].rel_standing_envs = 0.1

    return cfg


QddBipedRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
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
        symmetry_cfg=None,  # symmetry.py tables are microduck-61D specific
    ),
    wandb_project="mjlab_qdd_biped",
    experiment_name="qdd_biped_velocity",
    run_name="qdd_biped_velocity",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=10_000,
)
