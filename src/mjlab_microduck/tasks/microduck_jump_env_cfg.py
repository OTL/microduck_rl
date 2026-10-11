"""Microduck Jump task — jump as high as possible, over and over.

Perpetual policy: the robot starts STANDING (HOME pose + noise) and hops in
place, each hop as high as it can, landing back on its feet. The owner's
metric is "the highest point any part of the body reaches", so the paid
quantity is the world z of the TOPMOST body (max over all body frames), not
the trunk height — stretching the neck / legs at the apex is part of the task.

Why perpetual and not "one jump then stand": the actor is a memoryless MLP
with a constant command and no clock, so it cannot know that it has already
jumped. A once-per-episode record reward is therefore not Markov for it;
a per-flight reward is. The runtime can still run it for a fixed duration.

Key design decisions:
  - HARD airborne gate (whole-robot ground-contact sensor): height is paid
    only while NO robot geom touches the terrain. Tip-toe / neck-stretch /
    head-tripod poses raise the top point too, but earn nothing — only a real
    ballistic flight counts.
  - Paid per airborne step: hop_bonus + clearance of the top point above
    JUMP_TOP_Z_REF. The per-flight integral grows ~v³ with take-off speed, so
    one high jump beats many micro-hops; the clearance is CLAMPED
    (JUMP_MAX_CLEARANCE) so a contact-solver launch is not a jackpot.
  - Upright gate on every jump term + fell_over termination kept: a flight
    that tumbles pays nothing and ends the episode, so the landing has to be
    on the feet (that is what keeps the policy deployable).
  - Push-off bootstrap: capped upward trunk velocity while supported — any
    upward drive pays immediately, before the first real take-off exists.
  - Motion-blockers (body_ang_vel, angular_momentum, |a_z|, action_rate) stay
    LOW during discovery and are ramped by curriculum afterwards (standup /
    roulade timing lesson: an attempt-tax during discovery makes "stand
    still" win).
  - Obs layout is the unified 61D actor layout (twist + zero-padded head/body
    command slots) so the runtime can hot-swap ONNX files with one buffer.

Physical expectation: XL330 servos on an ~800 g robot give a few cm of
clearance at best — the reward has no target height, it just pays for more.

DR / noise / delays: velocity-parity, copied from the ball-kick / roulade envs.
"""

import math
from copy import deepcopy

# Symmetry — the jump is left/right symmetric; the mirror loss fights
# one-legged / twisting take-offs (same use as roulade).
ENABLE_SYMMETRY = True

# ── Domain randomisation (matched to velocity / standup) ─────────────────────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_KD_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = False  # a velocity kick mid-flight is incoherent
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

# ── Ranges (matched to velocity / standup) ───────────────────────────────────
COM_RANDOMIZATION_RANGE             = 0.003           # ramped to 0.015 via curriculum
HEAD_COM_RANDOMIZATION_RANGE        = 0.003           # ramped to 0.01 via curriculum
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)    # unused (kp DR off)
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)      # unused (kd DR off)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# ── Task constants ────────────────────────────────────────────────────────────
# Room for ~8-10 crouch/jump/land cycles per episode.
EPISODE_LENGTH_S = 6.0

# Reference height of the TOPMOST body frame when standing at HOME (m above
# the floor). ESTIMATE from the MJCF kinematics (trunk 0.115 + neck 0.032 +
# neck link 0.05 ≈ 0.20), NOT yet measured in sim — re-measure before a long
# run. The reward is robust to a wrong value by construction: clearance is
# clamped at 0 below the reference and the airborne hop bonus still pays, so
# a too-high reference only delays the height gradient by that many mm.
JUMP_TOP_Z_REF = 0.20
# Clearance that is worth +1 on top of the hop bonus.
JUMP_HEIGHT_SCALE = 0.02
# Paid clearance clamp: far above anything the servos can do, low enough that
# a contact-solver launch is not a jackpot.
JUMP_MAX_CLEARANCE = 0.15
# Paid per airborne step regardless of height — the gradient for the very
# first lift-off. Kept small so vibrating on the floor cannot compete with a
# real flight.
JUMP_HOP_BONUS = 0.2
# Cap on the rewarded push-off speed (m/s). 1.5 m/s ≈ an 11 cm ballistic
# rise — beyond the hardware, so the cap only clips solver spikes.
JUMP_MAX_PUSHOFF_VZ = 1.5
# Upright gate (trunk tilt): full pay below 25°, zero above 50°.
JUMP_TILT_FULL_DEG = 25.0
JUMP_TILT_ZERO_DEG = 50.0

_LEG_JOINTS  = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]

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
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_jump_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Microduck jump environment configuration."""

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # Whole-robot ground contact — the AIRBORNE GATE: jump height is paid only
    # while no robot geom touches the terrain (same sensor as roulade's
    # support gate, read the other way round).
    robot_ground_cfg = ContactSensorCfg(
        name="robot_ground_contact",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    # Groundcontact model (same spec as standup / ball-kick / roulade): a bad
    # landing must be able to hit the floor with something else than the feet.
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors  = (feet_ground_cfg, self_collision_cfg, robot_ground_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: drop walking-specific terms ──────────────────────────────────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",           # gait-conditioned; replaced by pose_target_match below
        "soft_landing",   # velocity removes it
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Rewards: jump objective ───────────────────────────────────────────────
    # THE task term: height of the topmost body point, paid only in flight and
    # only while upright. A 3 cm jump is ~8 airborne steps at ~(0.2 + 1.0)
    # → ≈ +1.5/step averaged over a 0.5 s cycle at weight 4, vs ≈ 2/step for
    # the whole standing stack below — the jump is worth its transient cost.
    jump_gate_params = {
        "sensor_name":   robot_ground_cfg.name,
        "tilt_full_deg": JUMP_TILT_FULL_DEG,
        "tilt_zero_deg": JUMP_TILT_ZERO_DEG,
    }
    cfg.rewards["jump_top_height"] = RewardTermCfg(
        func=microduck_mdp.jump_top_height_airborne,
        weight=4.0,
        params={
            **jump_gate_params,
            "ref_height":    JUMP_TOP_Z_REF,
            "height_scale":  JUMP_HEIGHT_SCALE,
            "max_clearance": JUMP_MAX_CLEARANCE,
            "hop_bonus":     JUMP_HOP_BONUS,
        },
    )

    # Push-off bootstrap: capped upward trunk velocity while SUPPORTED. Gives
    # gradient toward an explosive leg extension before any flight exists;
    # bobbing on the floor farms ~0.1 m/s of it, a real take-off 5× more plus
    # the flight term above.
    cfg.rewards["jump_pushoff"] = RewardTermCfg(
        func=microduck_mdp.jump_pushoff_velocity,
        weight=2.0,
        params={**jump_gate_params, "max_vz": JUMP_MAX_PUSHOFF_VZ},
    )

    # ── Rewards: stay a robot that can jump again ─────────────────────────────
    # Deliberately LIGHT and loose: the crouch and the stretched flight are
    # ~1 rad leg excursions and must stay affordable.
    cfg.rewards["alive"] = RewardTermCfg(func=microduck_mdp.is_alive, weight=0.5)

    cfg.rewards["pose_stand_legs"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=0.5,
        params={
            "std": 1.0,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,   # HOME = standing
        },
    )
    cfg.rewards["pose_stand_neck"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=0.3,
        params={
            "std": 0.6,
            "joint_indices": _NECK_JOINTS,
            "target_overrides": None,
        },
    )

    # Upright — velocity's recipe at half weight (take-off and landing both
    # need a vertical trunk; the jump terms are gated on it as well).
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 1.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)

    # ── Sim2real regularisers ─────────────────────────────────────────────────
    # Motion-blockers near zero during discovery (the jump IS a large-velocity,
    # large-impact event); smoothness is ramped in by curriculum below.
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.02)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=0.0
    )
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.02
    cfg.rewards["angular_momentum"].weight = -0.005

    # |a_z| impact shaping. NOTE: trunk_vertical_accel_penalty is SELF-NEGATING
    # (returns -|a_z|) → POSITIVE weight. Tiny on purpose: a jump cannot exist
    # without a take-off and a landing spike; this only prefers the softer
    # landing among equal heights.
    cfg.rewards["gentle_landing"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.001,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.5,
        params={"sensor_name": self_collision_cfg.name},
    )

    # ── Observations (unified 61D actor layout) ───────────────────────────────
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    # No terrain-height sensor in this env (flat only) — drop the base
    # template's sensor-backed terms, like standup/ground-pick do.
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    # IMU obs delay — match velocity's 2026-07 audit values.
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    # Obs noise — matched to the velocity env.
    cfg.observations["actor"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise       = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise       = Unoise(n_min=-0.25, n_max=0.25)

    # IMU mounting-misalignment DR (obs-level, actor only).
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # 1-ctrl-step lag on joint_vel (Dynamixel moving-average, see velocity env).
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    # Deepcopy joint_pos/joint_vel per group so the encoder-bias `biased` flag
    # below applies to the actor only.
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # Command obs slots — unified layout parity: [twist(3), head(4), body(6)],
    # head/body zero-padded (the head is free to stretch for height).
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # ── Command: tiny noise around zero (obs-shape parity only) ───────────────
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    command.heading_command   = False
    command.ranges.heading    = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # ── Terminations ──────────────────────────────────────────────────────────
    # fell_over KEPT: a jump that ends on the floor loses the rest of the
    # episode — the landing is part of the task.
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)  # match velocity

    # Joint noise on the standing start: deployment hands off from the walk /
    # velstand policy, whose settled stand won't match HOME exactly.
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)

    # Standing-only start (standup env's ground-state machinery: random yaw,
    # ± tilt noise, z near equilibrium).
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob":   0.0,
            "face_up_prob":     0.0,
            "sitting_prob":     0.0,
            "standing_prob":    1.0,
            "sitting_tilt_max": math.radians(5),  # ±5° pitch/roll on the stand
            "standing_z_min":   0.11,
            "standing_z_max":   0.12,
        },
    )

    if "push_robot" in cfg.events:
        del cfg.events["push_robot"]

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
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

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    # ── Terrain: flat only ────────────────────────────────────────────────────
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Curriculum ────────────────────────────────────────────────────────────
    if "terrain_levels" in cfg.curriculum:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # Smoothness / impact polish — introduced only after the jump exists
    # (standup timing lesson). Ceilings stay well under velocity's -1.0: the
    # push-off is a one-step full-range action change by nature.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.02},
                {"step": 1000 * 24,  "weight": -0.05},
                {"step": 2000 * 24,  "weight": -0.1},
            ],
        },
    )
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 2000 * 24,  "weight": -5e-4},
            ],
        },
    )
    cfg.curriculum["gentle_landing_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            # POSITIVE weights: the func is self-negating (returns -|a_z|).
            "reward_name":   "gentle_landing",
            "weight_stages": [
                {"step": 0,          "weight": 0.001},
                {"step": 2000 * 24,  "weight": 0.003},
            ],
        },
    )

    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                    {"step": 1500 * 24, "range": 0.015},
                ],
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckJumpRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normalizer MUST be baked into ONNX by export.py
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
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_jump",
    run_name="microduck_jump",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=5_000,
)
