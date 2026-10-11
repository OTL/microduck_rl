"""Microduck HEAD SWAY task — stand in place and keep slowly shaking the head
left/right (head_yaw), forever.

Perpetual constant-command policy: deployed with an all-zero command block, so
the policy has NO clock input. The oscillation must be a limit cycle the policy
generates from its own state (head_yaw pos/vel + last action are in the 48D
proprioception). The reward is therefore phase-free, defined on the head_yaw
phase plane (see head_sway_track in mdp.py):

  x = q / A,  y = -q̇ / (A·ω)       (q = head_yaw delta from HOME)

A sinusoid q = A·cos(ωt) is the unit circle traversed at angular rate ω.
Reward = Gaussian(radius − 1) × Gaussian(EMA phase rate − ω). Holding the head
still at ±A sits ON the circle but has zero phase rate → pays ~0 (no static
farm); fast jitter has the wrong phase rate → pays ~0.

Built on make_microduck_velocity_env_cfg (walk model) so DR / obs / noise /
delays / NaN guard / BAM friction stay in sync. Everything locomotion-specific
is removed; commands are pinned below every 0.01 gate so the inherited
standing terms (pose std_standing, velocity tracking at ~0) hold the robot
still. The 61D obs layout is untouched.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, RewardTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    NUM_STEPS_PER_ENV,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG

# Symmetry mirror-loss: OFF (default for new tasks).
ENABLE_SYMMETRY = False

# Head sway target: slow ±0.6 rad (~34°) yaw swing, 4 s per left-right-left
# cycle → peak head speed A·ω ≈ 0.94 rad/s.
HEAD_SWAY_AMPLITUDE = microduck_mdp.HEAD_SWAY_AMPLITUDE
HEAD_SWAY_PERIOD = microduck_mdp.HEAD_SWAY_PERIOD

# Gaussian tolerances of the phase-plane reward. radius_std is wide enough
# that a half-amplitude swing at the right tempo still scores exp(-(0.5/0.4)²)
# ≈ 0.21 — gradient to grow the swing from a small oscillation.
HEAD_SWAY_RADIUS_STD = 0.4
HEAD_SWAY_RATE_STD_FRAC = 0.5
HEAD_SWAY_RATE_TAU_S = 0.2

# All twist components stay below the 0.01 command gates of the inherited
# terms (|xy| + |wz| ≤ 0.003·√2 + 0.003 ≈ 0.0072): always "standing", while the
# twist input neurons still see non-zero values.
TWIST_TINY = 0.003
# Fraction of envs that get the exact all-zero twist (the deployment state).
ZERO_TWIST_FRACTION = 0.5
# Head command slot: unused by this task, kept alive with a tiny range.
HEAD_CMD_TINY = 0.02
# Gentler pushes than walking (±0.3): a standing robot shouldn't be forced to step.
PUSH_RANGE = (-0.15, 0.15)

NECK_PATTERN_NO_YAW = r"^(neck_pitch|head_pitch|head_roll)$"


def make_microduck_head_sway_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stand still and swing the head slowly left/right (perpetual)."""

    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    # === REWARDS ===
    # Locomotion terms out. angular_momentum is a motion-blocker that would tax
    # the head swing itself; head_pose_* would pull head_yaw back to the command.
    for name in (
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "angular_momentum",
        "head_pose_tracking",
        "head_pose_bias",
    ):
        cfg.rewards.pop(name, None)

    # Inherited standing terms: pose (std_standing on legs, command ~0),
    # upright, track_linear/angular_velocity at ~0 command (no trunk drift or
    # counter-rotation), body_ang_vel, action_rate_l2 (curriculum), self_collisions.
    cfg.rewards["track_linear_velocity"].weight = 1.0
    cfg.rewards["track_angular_velocity"].weight = 1.0

    # Main objective: on the unit orbit in the head_yaw phase plane, at tempo ω.
    cfg.rewards["head_sway_track"] = RewardTermCfg(
        func=microduck_mdp.head_sway_track,
        weight=4.0,
        params={
            "amplitude": HEAD_SWAY_AMPLITUDE,
            "period": HEAD_SWAY_PERIOD,
            "radius_std": HEAD_SWAY_RADIUS_STD,
            "rate_std_frac": HEAD_SWAY_RATE_STD_FRAC,
            "rate_tau_s": HEAD_SWAY_RATE_TAU_S,
        },
    )
    # Bootstrap: pay head_yaw speed up to the target peak speed (capped, so
    # swinging faster pays nothing more). Faded out by curriculum once the
    # orbit term carries the gradient.
    cfg.rewards["head_sway_speed"] = RewardTermCfg(
        func=microduck_mdp.head_sway_speed,
        weight=0.5,
        params={"amplitude": HEAD_SWAY_AMPLITUDE, "period": HEAD_SWAY_PERIOD},
    )
    # The other neck/head joints stay at HOME (cost ≥ 0 → negative weight).
    cfg.rewards["neck_joint_pos_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_joint_pos_l2,
        weight=-1.0,
        params={"pattern": NECK_PATTERN_NO_YAW},
    )
    # Feet stay planted (count of airborne feet ≥ 0 → negative weight).
    cfg.rewards["no_stepping"] = RewardTermCfg(
        func=microduck_mdp.no_stepping_penalty,
        weight=-1.0,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    )

    # === COMMANDS ===
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (-TWIST_TINY, TWIST_TINY)
    twist.ranges.lin_vel_y = (-TWIST_TINY, TWIST_TINY)
    twist.ranges.ang_vel_z = (-TWIST_TINY, TWIST_TINY)
    twist.rel_standing_envs = ZERO_TWIST_FRACTION
    twist.rel_heading_envs = 0.0
    twist.rel_turn_in_place_envs = 0.0

    cfg.commands["head_pose"].ranges = (
        (-HEAD_CMD_TINY, HEAD_CMD_TINY),  # neck_pitch
        (-HEAD_CMD_TINY, HEAD_CMD_TINY),  # head_pitch
        (-HEAD_CMD_TINY, HEAD_CMD_TINY),  # head_yaw
        (-HEAD_CMD_TINY / 2, HEAD_CMD_TINY / 2),  # head_roll
    )

    # === EVENTS ===
    if "push_robot" in cfg.events:
        cfg.events["push_robot"].params["velocity_range"] = {
            "x": PUSH_RANGE,
            "y": PUSH_RANGE,
        }

    # === CURRICULUM ===
    # Walking-only schedules out: standing_envs would overwrite the zero-twist
    # fraction, head_pose_range would widen the unused head command.
    for name in ("standing_envs", "head_pose_range", "head_pose_bias_weight"):
        cfg.curriculum.pop(name, None)

    cfg.curriculum["head_sway_speed_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "head_sway_speed",
            "weight_stages": [
                {"step": 0, "weight": 0.5},
                {"step": 300 * NUM_STEPS_PER_ENV, "weight": 0.25},
                {"step": 600 * NUM_STEPS_PER_ENV, "weight": 0.0},
            ],
        },
    )

    return cfg


MicroduckHeadSwayRlCfg = RslRlOnPolicyRunnerCfg(
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
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="head_sway",
    run_name="head_sway",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=2_000,
)
