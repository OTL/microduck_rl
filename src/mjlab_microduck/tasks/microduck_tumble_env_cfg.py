"""Microduck perpetual forward tumble — tuck into a ball and keep rolling.

Perpetual policy (constant zero command): from a standing start the robot
dives forward, tucks, and chains forward rolls for as long as the policy
runs. Faster is better. Built on the roulade env (same robot model, DR, obs,
noise, spawns, rotation accumulator) with the EPISODIC half removed: no
2π progress cap, no completion-gated landing stack, no stand tax.

Design (roulade lessons carried over, see that cfg's docstring):
  • ONE dense task signal — the advance rate of the roulade rotation
    frontier (support-gated + sagittal-flatness-gated integral of forward
    pitch rate), uncapped in total: every further radian pays. Rocking under
    the frontier, camping in any pose, ballistic flips and shoulder rolls all
    pay zero.
  • "Curled up" is a MULTIPLIER on that signal (Gaussian on the distance to
    the tuck pose), never a standalone reward — a parked tuck earns nothing.
  • Forward travel pays only while the frontier is advancing (EMA gate), so
    walking or sliding forward cannot farm it, and it is measured along the
    horizontal normal of the body's lateral axis (body x spins with the roll).
  • No rotation-speed tax (AGENTS.md: a 25 cm robot tumbles at 3.5–5.5 rad/s
    naturally). Anti-violence pressure lives in |a_z|, action_rate and the
    support gate; TUMBLE_MAX_PAID_RATE only bounds solver spikes.
  • Reverse-curriculum mid-roll spawns kept from the roulade: the second half
    of each turn (supine → over the feet again) is the hard part and gets
    on-policy data from iteration 0.
"""

import math

# Symmetry — a forward tumble is left-right symmetric (same as the roulade).
ENABLE_SYMMETRY = True

# Episode: long enough for several chained rolls from a standing start.
EPISODE_LENGTH_S = 10.0

# ── Task reward constants ─────────────────────────────────────────────────────
# Progress is paid in revolutions/s (× tuck multiplier): ~0.5 rev/s → ~5/step.
TUMBLE_PROGRESS_WEIGHT = 10.0
# Bounds contact-solver ω spikes only — well above the natural tumble rate.
TUMBLE_MAX_PAID_RATE = 15.0   # rad/s
# Tuck multiplier: progress × (1 − bonus + bonus × exp(−mse / std²)). Loose
# std: the legs MUST leave the tuck to push every turn (inherent error).
TUMBLE_TUCK_BONUS = 0.4
TUMBLE_TUCK_STD = 0.7         # rad
# Forward travel (m/s, clamped) × frontier-advance gate.
TUMBLE_TRAVEL_WEIGHT = 4.0
TUMBLE_TRAVEL_MAX_SPEED = 1.0  # m/s
TUMBLE_TRAVEL_RATE_NORM = 2.0  # rad/s of frontier advance for a full gate

# ── Spawns ────────────────────────────────────────────────────────────────────
# Any phase of the turn, with forward momentum up to the natural tumble rate.
MIDROLL_PITCH_MIN = math.radians(30.0)
MIDROLL_PITCH_MAX = math.radians(350.0)
MIDROLL_OMEGA_RANGE = (0.0, 5.0)

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    TUCK_OVERRIDES,
    make_microduck_roulade_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG

# Episodic-roulade terms that have no meaning in a perpetual tumble.
_ROULADE_ONLY_REWARDS = (
    "roulade_progress",            # capped at one turn
    "roulade_overspeed",           # speed tax — "faster is better" here
    "roulade_head_pivot",          # first-turn window shaping
    "roulade_landing_composite",   # completion-gated standing annuity
    "roulade_upright_after_roll",
    "roulade_height_after_roll",
    "roulade_landing_sharp",
    "roulade_stand_tax",
    "roulade_rise_velocity",
    "arrival_damping",             # taxes ω at standing height: every turn passes there
)


def make_microduck_tumble_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Microduck perpetual forward-tumble environment configuration."""

    # Robot, sensors (robot_ground_contact = the support gate), DR, 61D obs,
    # noise/delays, NaN guard, BAM friction expansion: all from the roulade.
    cfg = make_microduck_roulade_env_cfg(play=play)

    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.commands["twist"].resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)

    # ── Rewards ───────────────────────────────────────────────────────────────
    for name in _ROULADE_ONLY_REWARDS:
        cfg.rewards.pop(name, None)

    cfg.rewards["tumble_progress"] = RewardTermCfg(
        func=microduck_mdp.tumble_progress,
        weight=TUMBLE_PROGRESS_WEIGHT,
        params={
            "max_paid_rate":  TUMBLE_MAX_PAID_RATE,
            "tuck_overrides": TUCK_OVERRIDES,
            "tuck_std":       TUMBLE_TUCK_STD,
            "tuck_bonus":     TUMBLE_TUCK_BONUS,
        },
    )
    cfg.rewards["tumble_forward_travel"] = RewardTermCfg(
        func=microduck_mdp.tumble_forward_travel,
        weight=TUMBLE_TRAVEL_WEIGHT,
        params={
            "max_speed": TUMBLE_TRAVEL_MAX_SPEED,
            "rate_norm": TUMBLE_TRAVEL_RATE_NORM,
        },
    )

    # Straightness penalties (roulade_sagittal / _lateral_vel / _flatness),
    # |a_z| impact shaping (self-negating → positive weight), self-collision
    # and the smoothness terms are inherited unchanged. The two motion
    # blockers go to ~0: ω and angular momentum ARE the task here.
    cfg.rewards["body_ang_vel"].weight = -0.0005
    cfg.rewards["angular_momentum"].weight = -0.0005

    # ── Events ────────────────────────────────────────────────────────────────
    spawn = cfg.events["set_roulade_state"].params
    spawn["midroll_pitch_min"] = MIDROLL_PITCH_MIN
    spawn["midroll_pitch_max"] = MIDROLL_PITCH_MAX
    spawn["midroll_omega_range"] = MIDROLL_OMEGA_RANGE
    # Must run after set_roulade_state (dict insertion order): it re-anchors
    # the paid pointer on the accumulator value that event just wrote.
    cfg.events["reset_tumble_state"] = EventTermCfg(
        func=microduck_mdp.reset_tumble_state,
        mode="reset",
    )

    # ── Curriculum ────────────────────────────────────────────────────────────
    cfg.curriculum.pop("arrival_damping_weight", None)
    # Softer smoothness ceiling than the roulade: there is no settle phase.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.1},
                {"step": 3000 * 24,  "weight": -0.2},
            ],
        },
    )

    if play:
        # Deployment view: always from standing, like the policy switch.
        cfg.curriculum.pop("roulade_spawn_mix", None)
        spawn["standing_prob"] = 1.0
        spawn["midroll_prob"] = 0.0

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckTumbleRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="microduck_tumble",
    run_name="microduck_tumble",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=6_000,
)
