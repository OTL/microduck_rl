"""Microduck headstand task — run 1 (untrained design, see "UNVERIFIED" below).

Perpetual constant-command policy: the robot starts standing, plants the flat
top of its head on the floor, lifts both feet, and stays balanced on the head
alone. Triggered at deployment like the other tricks (policy switch = it goes
down onto its head and stays there until another policy is selected).

Built on the roulade env (same robot family, same 61D obs, same DR stack, same
head-top axis convention) with the roulade task terms replaced.

Design:
  • ONE per-step annuity, `headstand_hold`, paid only in the goal state:
    head (jaw_soft) touches the floor AND no other body touches it, scaled by
    how squarely the head top faces down and by stillness. Hard contact gate —
    a head+feet tripod pays exactly zero per step (the "head-tripod" exploit).
  • ONE potential-based shaping term, `headstand_progress`, paying Δ of a
    slewed potential (head top turning down → head top touching → share of the
    ground load carried by the head). Holding any intermediate pose pays zero,
    falling back off the head pays negative, and the slew makes a violent mount
    pay no faster than a slow one (no jackpot).
  • NO standing terms at all (owner's request): no upright / height / HOME-pose
    reward, no fallen tax, no fell_over termination. The base template's
    `upright` and `pose` rewards are deleted.
  • Reverse curriculum: most episodes START on the head (owner's request) —
    statically balanced inverted poses computed from the compiled model at the
    first reset (mdp.headstand_spawn_table: FK grid over neck_pitch/head_pitch
    × leg tuck, keeps configurations whose CoM sits over the head-top patch and
    whose other parts clear the floor). On-head spawns are pre-latched, and the
    `headstand_lost` termination recycles an episode 1 s after a latched
    headstand is lost — so a fall costs the remaining annuity and the on-head
    slice gets many short attempts instead of 10 s lying on the floor.
  • Robot = robot_allcollisions.xml: "only the head touches" is only honest if
    every part can touch (on groundcontact the neck servos pass through the
    floor).

UNVERIFIED (written from a read-only checkout, nothing was simulated):
  - that a head-only static balance exists for this model — the spawn table
    prints how many balanced configurations it found; 0 means the fallback
    (best CoM offset) is used and the on-head spawns tip over immediately;
  - that mdp._HEAD_TOP_AXIS (measured 2026-08-13 for roulade) is still the
    flat-top normal — by FK it sits 28° behind trunk-up at today's HOME neck;
  - whether the XL330 neck can lift the trunk+legs from a standing start.
  Check the `[mdp] headstand spawn table` line and Metrics/headstand_only_head
  before trusting a long run.
"""

import math
from copy import deepcopy

# Symmetry — the headstand is left-right symmetric, so the mirror loss is
# applicable (roulade uses it), but OFF by default per AGENTS.md.
ENABLE_SYMMETRY = False

# ── Domain randomisation (matched to roulade/standup for sim2real parity) ────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_KD_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

COM_RANDOMIZATION_RANGE             = 0.003   # ramped to 0.01 via curriculum
HEAD_COM_RANDOMIZATION_RANGE        = 0.003   # ramped to 0.01 via curriculum
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)  # unused (kp DR off)
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)    # unused (kd DR off)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# Perpetual hold: long enough to mount (~2 s) and prove a sustained balance.
EPISODE_LENGTH_S = 10.0

# ── Task weights ──────────────────────────────────────────────────────────────
# hold pays HOLD_WEIGHT per second in the goal state; progress pays
# PROGRESS_WEIGHT once for a full mount (= 2 s of holding) and charges the same
# for falling off.
HOLD_WEIGHT     = 6.0
PROGRESS_WEIGHT = 12.0

# Episode recycle: a headstand that was established (or spawned) and then lost
# for this long ends the episode.
HEADSTAND_LOST_GRACE_S = 1.0

# Head slam protection: static headstand load ≈ 8 N (body weight), a face-plant
# spike is ~60 N (velstand measurement). Billed above 20 N only.
HEAD_IMPACT_THRESH_N = 20.0

# ── Spawn mix (reverse curriculum) ────────────────────────────────────────────
# On-head starts stay the MAJORITY at every stage (owner's request); the
# standing slice grows as the hold consolidates, and the on-head perturbation
# widens with it.
STANDING_Z_RANGE = (0.116, 0.124)   # allcollisions stands at 0.116 (velstand)
SPAWN_MIX_STAGES = [
    {"step": 0,         "params": {"standing_prob": 0.25, "headstand_prob": 0.75, "headstand_tilt_max": math.radians(3.0), "headstand_ang_vel_max": 0.0}},
    {"step": 2000 * 24, "params": {"standing_prob": 0.35, "headstand_prob": 0.65, "headstand_tilt_max": math.radians(6.0), "headstand_ang_vel_max": 0.3}},
    {"step": 4000 * 24, "params": {"standing_prob": 0.45, "headstand_prob": 0.55, "headstand_tilt_max": math.radians(8.0), "headstand_ang_vel_max": 0.5}},
]
# Leg tuck used by the on-head spawn table (servo-index keyed, roulade's tuck
# anchor, legs only): each table entry lerps HOME→tuck by a fixed factor.
HEADSTAND_LEG_TUCK = {
    2:  -1.15,  # left  hip_pitch
    3:   1.25,  # left  knee
    4:   1.05,  # left  ankle
    11:  1.15,  # right hip_pitch
    12: -1.25,  # right knee
    13: -1.05,  # right ankle
}

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

from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_headstand_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck headstand environment configuration (flat only)."""
    assert not rough, "headstand is a flat-floor task"

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

    # Head-ground contact — the support. jaw_soft carries every head geom.
    # NAME IS LOAD-BEARING: mdp._headstand_measure reads it.
    head_ground_cfg = ContactSensorCfg(
        name="head_ground_contact",
        primary=ContactMatch(mode="body", pattern="jaw_soft", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )

    # Every OTHER body vs the floor — the "head only" gate (feet, legs, trunk,
    # neck links). NAME IS LOAD-BEARING: mdp._headstand_measure reads it.
    other_ground_cfg = ContactSensorCfg(
        name="other_ground_contact",
        primary=ContactMatch(mode="body", pattern=r".*", entity="robot", exclude=("jaw_soft",)),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_ALLCOLLISIONS_ROBOT_CFG}
    cfg.scene.sensors  = (feet_ground_cfg, head_ground_cfg, other_ground_cfg)
    cfg.viewer.body_name = "trunk_base"
    # 70 collision geoms piled on the floor: velstand's contact budget.
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 200)

    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: drop walking AND standing terms ──────────────────────────────
    # upright / pose are the base template's standing attractors — the owner
    # asked for no standing reward and no not-standing penalty.
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "soft_landing",
        "pose",
        "upright",
    ]:
        cfg.rewards.pop(name, None)

    # ── Rewards: headstand task set ───────────────────────────────────────────
    cfg.rewards["headstand_hold"] = RewardTermCfg(
        func=microduck_mdp.headstand_hold,
        weight=HOLD_WEIGHT,
        params={
            "tilt_std":      0.5,   # rad — head top 28° off flat still scores ~0.4
            "ang_vel_std":   2.0,   # rad/s trunk — balance wobble stays visible
            "joint_vel_std": 2.0,   # rad/s servo RMS
            "still_floor":   0.4,   # a wobbling hold keeps 40%, a still one 100%
        },
    )
    cfg.rewards["headstand_progress"] = RewardTermCfg(
        func=microduck_mdp.headstand_progress,
        weight=PROGRESS_WEIGHT,
    )

    # ── Anti-violence / sim2real regularisers ─────────────────────────────────
    # mjlab-style costs (>= 0) → NEGATIVE weight.
    cfg.rewards["head_impact"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=-0.005,  # ramped by head_impact_weight
        params={"sensor_name": head_ground_cfg.name, "threshold": HEAD_IMPACT_THRESH_N},
    )
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=0.0
    )
    # Motion-blockers stay light: the mount turns the trunk ~180°.
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.02
    cfg.rewards["angular_momentum"].weight = -0.001
    # ⚠️ POSITIVE weight: trunk_vertical_accel_penalty returns -|a_z| already.
    cfg.rewards["gentle_motion"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.002,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # ── Observations (identical 61D layout to walking / standup / roulade) ────
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    cfg.observations["actor"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise       = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise       = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

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

    # Command obs slots: zero padding for head (4) and body (6) — the head is
    # the support here, not a commandable. 61D layout parity is kept so the
    # runtime stack works unchanged (send zeros).
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # Critic-only privileged task state (head-top alignment, contacts, head
    # load share) — the value function can't infer "feet just left the floor"
    # from proprioception alone.
    cfg.observations["critic"].terms["headstand_state"] = ObservationTermCfg(
        func=microduck_mdp.headstand_critic_state,
    )

    # ── Command: tiny noise around zero (kept for obs-shape parity) ──────────
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
    # Leaving the feet is the task — no fall termination.
    cfg.terminations.pop("fell_over", None)
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": (feet_ground_cfg.name, head_ground_cfg.name, other_ground_cfg.name)},
    )
    # Recycle: a headstand that existed and is gone for 1 s ends the episode.
    # Not a standing condition — an episode that never reached the head runs
    # to the timeout untouched.
    cfg.terminations["headstand_lost"] = TerminationTermCfg(
        func=microduck_mdp.headstand_lost,
        time_out=False,
        params={"grace_s": HEADSTAND_LOST_GRACE_S},
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
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    # Standing start or on-head start; also resets the headstand latch /
    # potential (must run after reset_robot_joints — dict insertion order).
    _stage0 = SPAWN_MIX_STAGES[0]["params"]
    cfg.events["set_headstand_state"] = EventTermCfg(
        func=microduck_mdp.reset_headstand_state,
        mode="reset",
        params={
            # play = deployment view: every episode starts standing.
            "standing_prob":         1.0 if play else _stage0["standing_prob"],
            "headstand_prob":        0.0 if play else _stage0["headstand_prob"],
            "standing_z_range":      STANDING_Z_RANGE,
            "standing_tilt_max":     math.radians(3.0),
            "headstand_tilt_max":    _stage0["headstand_tilt_max"],
            "headstand_drop_range":  (0.002, 0.006),
            "headstand_ang_vel_max": _stage0["headstand_ang_vel_max"],
            "leg_joint_noise_std":   0.05,
            "leg_overrides":         HEADSTAND_LEG_TUCK,
        },
    )

    cfg.events.pop("push_robot", None)

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

    # ── Terrain ───────────────────────────────────────────────────────────────
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Curriculum ────────────────────────────────────────────────────────────
    cfg.curriculum.pop("terrain_levels", None)
    del cfg.curriculum["command_vel"]

    if not play:
        cfg.curriculum["headstand_spawn_mix"] = CurriculumTermCfg(
            func=microduck_mdp.event_param_curriculum,
            params={
                "event_name": "set_headstand_state",
                "param_stages": SPAWN_MIX_STAGES,
            },
        )

    # CoM DR capped at ±10 mm (roulade/standup go to 15): the support here is
    # a head top, not two feet — a larger offset leaves the patch entirely.
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 1000 * 24, "range": 0.005},
                    {"step": 2500 * 24, "range": 0.01},
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
                    {"step": 1000 * 24, "range": 0.005},
                    {"step": 2500 * 24, "range": 0.01},
                ],
            },
        )

    # Smoothness after discovery (AGENTS.md): light from step 0, tightened once
    # the hold exists. If Metrics/headstand_only_head steps DOWN at a stage
    # boundary, stretch the stages — never move them earlier.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.1},
                {"step": 1500 * 24,  "weight": -0.2},
                {"step": 3000 * 24,  "weight": -0.4},
            ],
        },
    )
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 2500 * 24,  "weight": -5e-4},
                {"step": 3500 * 24,  "weight": -1e-3},
            ],
        },
    )
    cfg.curriculum["head_impact_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "head_impact",
            "weight_stages": [
                {"step": 0,          "weight": -0.005},
                {"step": 1500 * 24,  "weight": -0.02},
            ],
        },
    )
    cfg.curriculum["gentle_motion_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            # POSITIVE weights: the func is self-negating (returns -|a_z|).
            "reward_name":   "gentle_motion",
            "weight_stages": [
                {"step": 0,          "weight": 0.002},
                {"step": 2500 * 24,  "weight": 0.005},
            ],
        },
    )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckHeadstandRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="microduck_headstand",
    run_name="microduck_headstand",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=8_000,
)
