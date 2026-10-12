"""Microduck headstand task — dive onto the head top, hold, stand back up.

Episodic policy (constant command, no phase clock, no reference motion):
robot starts standing, plants the flat top of its head on the floor, lifts
both feet off the ground and holds still for ~HOLD_S seconds, then gets back
on its feet and stays there. Triggered at deployment like standup / roulade
(policy switch = the trick starts immediately).

Built on the roulade env (same robot model, DR, obs layout, sensors, sim2real
regularisers and their late curricula) — the first half of a roulade IS the
entry of a headstand, and its chin-tuck / head-top-axis measurements are
reused as-is. Only the task rewards, the spawn event and its curriculum are
replaced.

Feasibility note: the duck has no arms and its head sits in FRONT of the
trunk, so a human-style vertical headstand does not exist for this body. The
target is the closest feasible thing: weight on the flat head top (chin
tucked), both feet in the air, nothing else on the floor — balanced by the
4 neck servos and the leg posture.

Design (see the headstand section of mdp.py):
  • ONE dense entry signal — rate-limited increments of the max-so-far entry
    potential (forward pitch → head-top contact → feet up). Potential-based:
    camping anywhere pays zero per step.
  • BUDGETED hold annuity — the hold quality (head-top contact × feet
    airborne × feet height × head-only support × stillness) pays per step
    only until HOLD_S seconds of hold time are banked. No jackpot, nothing
    to farm: staying longer pays zero.
  • Stand rewards (composite product, upright, sharp layer) and the
    not-standing tax are gated on the HOLD-DONE latch — "do nothing" earns
    nothing, the standing spawn cannot farm them, and no upright pressure
    opposes the dive. How to get up (push back or finish the roll) is left
    to RL.
  • Reverse curriculum: a slice of episodes spawns on the head (hold not
    banked → practices the balance), another spawns anywhere in the roll
    with the hold already banked (→ practices the rise).

The policy has no clock: the hold duration is enforced by the reward budget
(leaving early forfeits the rest of the hold AND keeps the stand gate shut,
overstaying is taxed), so the deployed hold time is approximate.
"""

import math

# Symmetry — the trick is sagittal / left-right symmetric (same as roulade).
ENABLE_SYMMETRY = True

# Episode: dive ~1.5 s + hold 2 s + rise ~2.5 s + settle.
EPISODE_LENGTH_S = 8.0

# Banked hold time that opens the stand phase.
HOLD_S = 2.0

# ── Reverse-curriculum spawns ─────────────────────────────────────────────────
# Pre-hold: on / near the head (90° = balanced on the head in the roulade
# convention), chin tucked. Post-hold: anywhere a dismount can end up
# (face-down → on the head → on the back → supine).
PREHOLD_PITCH_RANGE  = (math.radians(70.0), math.radians(130.0))
POSTHOLD_PITCH_RANGE = (math.radians(60.0), math.radians(300.0))
SPAWN_Z_RANGE        = (0.05, 0.10)

# Bodies whose ground contact means "not a head-only support" (trunk + legs;
# the feet have their own sensor, the neck/head chain is the allowed support).
# Names from robot_groundcontact.xml.
NON_HEAD_GROUND_BODIES = (
    "trunk_base",
    "yaw2roll",
    "hip_l",
    "upper_leg_left",
    "leg",
    "bearing_roll",
    "hip_l_2",
    "upper_leg_right",
    "leg_2",
)

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    STAND_Z,
    TUCK_OVERRIDES,
    _LEG_JOINTS,
    make_microduck_roulade_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG

# Roulade task terms replaced by the headstand set (the generic straightness /
# overspeed penalties and every regulariser are kept).
_ROULADE_TASK_REWARDS = (
    "roulade_progress",
    "roulade_head_pivot",
    "roulade_landing_composite",
    "roulade_upright_after_roll",
    "roulade_height_after_roll",
    "roulade_landing_sharp",
    "roulade_stand_tax",
    "roulade_rise_velocity",
)


def make_microduck_headstand_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Microduck headstand environment configuration."""

    # ── Base config: the roulade env (robot, DR, obs, sensors, regularisers) ──
    cfg = make_microduck_roulade_env_cfg(play=play)

    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.commands["twist"].resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)

    # Trunk / leg ground contact — the HEAD-ONLY support signal. NAME IS
    # LOAD-BEARING: _update_headstand reads it (as it reads the roulade env's
    # head_ground_contact and feet_ground_contact sensors).
    body_ground_cfg = ContactSensorCfg(
        name="body_ground_contact",
        primary=ContactMatch(
            mode="body",
            pattern=rf"^({'|'.join(NON_HEAD_GROUND_BODIES)})$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (body_ground_cfg,)

    # ── Rewards: drop the roulade task set ────────────────────────────────────
    for name in _ROULADE_TASK_REWARDS:
        cfg.rewards.pop(name, None)

    # ── Rewards: headstand task set ───────────────────────────────────────────
    # Entry progress — episode total is weight × 1.0 (the full potential),
    # whatever the path; paid at most max_paid_rate per second so arriving
    # faster pays nothing more.
    cfg.rewards["headstand_progress"] = RewardTermCfg(
        func=microduck_mdp.headstand_progress,
        weight=10.0,
        params={"hold_s": HOLD_S, "max_paid_rate": 1.0},
    )

    # Budgeted hold annuity — episode total ≤ weight × HOLD_S.
    cfg.rewards["headstand_hold"] = RewardTermCfg(
        func=microduck_mdp.headstand_hold,
        weight=6.0,
        params={"hold_s": HOLD_S},
    )

    # Hold-gated standing annuity — the dominant attractor once the hold is
    # banked. Stds as in the roulade landing (partial stand must score visibly).
    cfg.rewards["headstand_stand_composite"] = RewardTermCfg(
        func=microduck_mdp.headstand_stand_composite,
        weight=4.0,
        params={
            "hold_s":        HOLD_S,
            "target_height": STAND_Z,
            "height_std":    0.04,
            "upright_std":   0.40,
            "pose_std":      0.40,
            "joint_indices": _LEG_JOINTS,
        },
    )

    # Hold-gated bootstrap layer: linear in cos(tilt) from ANY orientation
    # (the composite product is ≈0 while still inverted).
    cfg.rewards["headstand_upright_after_hold"] = RewardTermCfg(
        func=microduck_mdp.headstand_upright_after_hold,
        weight=1.5,
        params={"hold_s": HOLD_S},
    )

    # Sharp last-mile layer (roulade run-4 lesson: broad stds park at a lean).
    cfg.rewards["headstand_stand_sharp"] = RewardTermCfg(
        func=microduck_mdp.headstand_stand_sharp,
        weight=2.0,
        params={
            "hold_s":        HOLD_S,
            "target_height": STAND_Z,
            "height_std":    0.015,
            "upright_std":   0.3,
        },
    )

    # Hold-gated not-standing tax: once the hold is banked, every step not
    # upright at standing height costs — overstaying on the head and every
    # flop (face / back / side) are net NEGATIVE against the upright layer.
    # NOTE: SELF-NEGATING function → POSITIVE weight.
    cfg.rewards["headstand_not_standing_tax"] = RewardTermCfg(
        func=microduck_mdp.headstand_not_standing_tax,
        weight=2.0,
        params={"hold_s": HOLD_S, "target_height": STAND_Z},
    )

    # ── Events: headstand spawns replace the roulade spawn ────────────────────
    # Must run after reset_robot_joints (dict insertion order) — the tucked
    # spawns lerp FROM the HOME pose it wrote.
    del cfg.events["set_roulade_state"]
    cfg.events["set_headstand_state"] = EventTermCfg(
        func=microduck_mdp.reset_headstand_state,
        mode="reset",
        params={
            "standing_prob":        1.0 if play else 0.3,
            "prehold_prob":         0.0 if play else 0.4,
            "posthold_prob":        0.0 if play else 0.3,
            "hold_s":               HOLD_S,
            "standing_tilt_max":    math.radians(5.0),
            "prehold_pitch_range":  PREHOLD_PITCH_RANGE,
            "posthold_pitch_range": POSTHOLD_PITCH_RANGE,
            "spawn_z_range":        SPAWN_Z_RANGE,
            "tuck_overrides":       TUCK_OVERRIDES,
            "tuck_factor_range":    (0.3, 1.0),
            "joint_noise_std":      0.08,
        },
    )

    # ── Curriculum ────────────────────────────────────────────────────────────
    # Reverse-curriculum mix: heavy on-head / post-hold spawns early, shift
    # toward standing starts as the full chain consolidates. The partial
    # spawns never go to zero (they keep the balance and the rise practiced).
    del cfg.curriculum["roulade_spawn_mix"]
    if not play:
        cfg.curriculum["headstand_spawn_mix"] = CurriculumTermCfg(
            func=microduck_mdp.event_param_curriculum,
            params={
                "event_name": "set_headstand_state",
                "param_stages": [
                    {"step": 0,         "params": {"standing_prob": 0.30, "prehold_prob": 0.40, "posthold_prob": 0.30}},
                    {"step": 2500 * 24, "params": {"standing_prob": 0.50, "prehold_prob": 0.25, "posthold_prob": 0.25}},
                    {"step": 4500 * 24, "params": {"standing_prob": 0.70, "prehold_prob": 0.15, "posthold_prob": 0.15}},
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
