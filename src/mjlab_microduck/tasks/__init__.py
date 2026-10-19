from mjlab_microduck.train_hook import maybe_submit_to_hf_jobs

# `train <task> ... --hf-jobs` submits to HF Jobs and exits here, before any
# of the cfg imports below: this module is what mjlab's plugin loader pulls
# in, and it is the only train path no install order can take from us (see
# train_hook.py). A no-op without the flag.
maybe_submit_to_hf_jobs()

from dataclasses import replace

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class MicroduckOnPolicyRunner(VelocityOnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        # resolve_symmetry_config injects _env into train_cfg["algorithm"]["symmetry_cfg"]
        # in-place, sharing the same dict object with self.alg.symmetry.  Replace the
        # train_cfg reference with a copy that omits _env so dump_yaml can serialize the
        # config (MjSpec is not picklable), without touching the PPO's internal reference.
        alg = train_cfg.get("algorithm", {})
        sym = alg.get("symmetry_cfg") if isinstance(alg, dict) else None
        if isinstance(sym, dict) and "_env" in sym:
            alg["symmetry_cfg"] = {k: v for k, v in sym.items() if k != "_env"}


from .microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    MicroduckRlCfg,
)
from .microduck_standup_env_cfg import (
    make_microduck_standup_env_cfg,
    MicroduckStandUpRlCfg,
)
from .microduck_velstand_env_cfg import (
    make_microduck_velstand_env_cfg,
    MicroduckVelStandRlCfg,
)
from .microduck_ground_pick_env_cfg import (
    make_microduck_ground_pick_env_cfg,
    MicroduckGroundPickRlCfg,
)
from .microduck_ball_kick_env_cfg import (
    make_microduck_ball_kick_env_cfg,
    MicroduckBallKickRlCfg,
)
from .microduck_sitstand_env_cfg import (
    make_microduck_sitstand_env_cfg,
    MicroduckSitStandRlCfg,
)
from .microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
    MicroduckRollersRlCfg,
)
from .microduck_velocity_swizzle_env_cfg import (
    make_microduck_velocity_swizzle_env_cfg,
    MicroduckSwizzleRlCfg,
)
from .microduck_roller_crouch_env_cfg import (
    make_microduck_roller_crouch_env_cfg,
    MicroduckRollerCrouchRlCfg,
)
from .microduck_roller_slope_env_cfg import (
    make_microduck_roller_slope_env_cfg,
    MicroduckRollerSlopeRlCfg,
)
from .microduck_roller_standup_env_cfg import (
    make_microduck_roller_standup_env_cfg,
    MicroduckRollerStandUpRlCfg,
)
from .microduck_spin_env_cfg import (
    make_microduck_spin_env_cfg,
    MicroduckSpinRlCfg,
)
from .microduck_roulade_env_cfg import (
    make_microduck_roulade_env_cfg,
    MicroduckRouladeRlCfg,
)
from .backlash import make_backlash_variant

# Standard velocity task
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(rough=True),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# VelStand — walking + fall recovery + body pose control in one policy.
register_mjlab_task(
    task_id="Mjlab-VelStand-Flat-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-VelStand-Rough-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Stand-up task — robot starts inverted (lying on back) and must stand up
register_mjlab_task(
    task_id="Mjlab-StandUp-Flat-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(),
    play_env_cfg=make_microduck_standup_env_cfg(play=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-StandUp-Rough-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(rough=True),
    play_env_cfg=make_microduck_standup_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# SitStand task — commanded sit ↔ stand in one policy, gently, head commandable
register_mjlab_task(
    task_id="Mjlab-SitStand-Flat-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-SitStand-Rough-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Ground-pick task — crouch, touch the ground with the mouth tip, return to stand
register_mjlab_task(
    task_id="Mjlab-GroundPick-Flat-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# BallKick task — kick a 70mm/15g ball forward hard with the right foot from a
# standing start (flat terrain only — a ball on rough terrain is another task).
register_mjlab_task(
    task_id="Mjlab-BallKick-Flat-MicroDuck",
    env_cfg=make_microduck_ball_kick_env_cfg(),
    play_env_cfg=make_microduck_ball_kick_env_cfg(play=True),
    rl_cfg=MicroduckBallKickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-GroundPick-Rough-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(rough=True),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller skate velocity task (passive-wheel model; historical task id kept)
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_velocity_rollers_env_cfg(),
    play_env_cfg=make_microduck_velocity_rollers_env_cfg(play=True),
    rl_cfg=MicroduckRollersRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller SWIZZLE task — clean classic swizzle (symmetric, feet grounded).
register_mjlab_task(
    task_id="Mjlab-Velocity-Swizzle-MicroDuck",
    env_cfg=make_microduck_velocity_swizzle_env_cfg(),
    play_env_cfg=make_microduck_velocity_swizzle_env_cfg(play=True),
    rl_cfg=MicroduckSwizzleRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerCrouch-Flat-MicroDuck",
    env_cfg=make_microduck_roller_crouch_env_cfg(),
    play_env_cfg=make_microduck_roller_crouch_env_cfg(play=True),
    rl_cfg=MicroduckRollerCrouchRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerSlope-Flat-MicroDuck",
    env_cfg=make_microduck_roller_slope_env_cfg(),
    play_env_cfg=make_microduck_roller_slope_env_cfg(play=True),
    rl_cfg=MicroduckRollerSlopeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller STANDUP — se relever sur rollers (policy dédiée, départ au sol).
register_mjlab_task(
    task_id="Mjlab-RollerStandUp-Flat-MicroDuck",
    env_cfg=make_microduck_roller_standup_env_cfg(),
    play_env_cfg=make_microduck_roller_standup_env_cfg(play=True),
    rl_cfg=MicroduckRollerStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Spin task — rotation rapide sur place, sur rollers (slot ground-pick).
register_mjlab_task(
    task_id="Mjlab-Spin-Flat-MicroDuck",
    env_cfg=make_microduck_spin_env_cfg(),
    play_env_cfg=make_microduck_spin_env_cfg(play=True),
    rl_cfg=MicroduckSpinRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roulade — forward roll over the flat head top, land back on the feet.
register_mjlab_task(
    task_id="Mjlab-Roulade-Flat-MicroDuck",
    env_cfg=make_microduck_roulade_env_cfg(),
    play_env_cfg=make_microduck_roulade_env_cfg(play=True),
    rl_cfg=MicroduckRouladeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Backlash variants — ±1° serial gear play per servo + encoder-through-backlash
# actuator feedback and joint obs (see tasks/backlash.py). Each family keeps its
# base task's collision model: Velocity → robot_walk_backlash.xml,
# VelStand → robot_allcollisions_backlash.xml, StandUp → robot_groundcontact_backlash.xml. Obs/action dims are
# unchanged vs the base tasks.
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG,
    MICRODUCK_BACKLASH_ROBOT_CFG,
    MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG,
    MICRODUCK_WALK_BACKLASH_ROBOT_CFG,
)

# (task_id, make_fn, make_kwargs, rl_cfg, backlash robot cfg). Task ids mirror
# the base ids with "-Backlash" inserted. Walk-model tasks get the walk
# backlash robot, roller tasks the wheels+backlash robot, the rest the
# groundcontact backlash robot — same model as their base task in each case.
_BL_GROUNDCONTACT = MICRODUCK_BACKLASH_ROBOT_CFG
_BL_WALK = MICRODUCK_WALK_BACKLASH_ROBOT_CFG
_BL_ROLLERS = MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG
_BL_ALLCOLLISIONS = MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG
_BACKLASH_TASKS = (
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-Velocity-Rough-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {"rough": True}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-VelStand-Flat-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {}, MicroduckVelStandRlCfg, _BL_ALLCOLLISIONS),
    ("Mjlab-VelStand-Rough-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {"rough": True}, MicroduckVelStandRlCfg, _BL_ALLCOLLISIONS),
    ("Mjlab-StandUp-Flat-Backlash-MicroDuck", make_microduck_standup_env_cfg, {}, MicroduckStandUpRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-StandUp-Rough-Backlash-MicroDuck", make_microduck_standup_env_cfg, {"rough": True}, MicroduckStandUpRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-SitStand-Flat-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {}, MicroduckSitStandRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-SitStand-Rough-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {"rough": True}, MicroduckSitStandRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-GroundPick-Flat-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {}, MicroduckGroundPickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-GroundPick-Rough-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {"rough": True}, MicroduckGroundPickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-BallKick-Flat-Backlash-MicroDuck", make_microduck_ball_kick_env_cfg, {}, MicroduckBallKickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck-Rollers", make_microduck_velocity_rollers_env_cfg, {}, MicroduckRollersRlCfg, _BL_ROLLERS),
    ("Mjlab-Velocity-Swizzle-Backlash-MicroDuck", make_microduck_velocity_swizzle_env_cfg, {}, MicroduckSwizzleRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerCrouch-Flat-Backlash-MicroDuck", make_microduck_roller_crouch_env_cfg, {}, MicroduckRollerCrouchRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerSlope-Flat-Backlash-MicroDuck", make_microduck_roller_slope_env_cfg, {}, MicroduckRollerSlopeRlCfg, _BL_ROLLERS),
)
for _task_id, _make_cfg, _kw, _rl_cfg, _robot_cfg in _BACKLASH_TASKS:
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=make_backlash_variant(_make_cfg(**_kw), _robot_cfg),
        play_env_cfg=make_backlash_variant(_make_cfg(play=True, **_kw), _robot_cfg),
        rl_cfg=_rl_cfg,
        runner_cls=MicroduckOnPolicyRunner,
    )

# QDD biped (a different robot: 5.66 kg, 10 QDD joints) — flat-ground velocity
# tracking. Own obs layout (41-D actor incl. gait clock), not part of the microduck 61-D family.
from .qdd_biped_velocity_env_cfg import (
    make_qdd_biped_velocity_env_cfg,
    policy_io_onnx_metadata as _qdd_policy_io_onnx_metadata,
    QddBipedRlCfg,
)


class QddBipedOnPolicyRunner(MicroduckOnPolicyRunner):
    """Adds the policy I/O contract (robot/qdd_biped/policy_io.json) to the
    ONNX the velocity runner writes next to every checkpoint."""

    def save(self, path: str, infos=None):
        super().save(path, infos)
        _, _, onnx_path = self._get_export_paths(path)
        try:
            from mjlab.rl.exporter_utils import attach_metadata_to_onnx

            if onnx_path.exists():
                attach_metadata_to_onnx(str(onnx_path), _qdd_policy_io_onnx_metadata())
        except Exception as e:  # never kill training over metadata
            print(f"[WARN] policy_io ONNX metadata failed: {e}")


register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-QddBiped",
    env_cfg=make_qdd_biped_velocity_env_cfg(),
    play_env_cfg=make_qdd_biped_velocity_env_cfg(play=True),
    rl_cfg=QddBipedRlCfg,
    runner_cls=QddBipedOnPolicyRunner,
)


# QDD biped on rough terrain: same actor obs / policy_io as the flat task (the
# controller swaps the walking ONNX only), terrain curriculum.
from .qdd_biped_rough_env_cfg import make_qdd_biped_rough_env_cfg

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-QddBiped",
    env_cfg=make_qdd_biped_rough_env_cfg(),
    play_env_cfg=make_qdd_biped_rough_env_cfg(play=True),
    rl_cfg=replace(
        QddBipedRlCfg,
        experiment_name="qdd_biped_rough",
        run_name="qdd_biped_rough",
        max_iterations=3_000,
    ),
    runner_cls=QddBipedOnPolicyRunner,
)


# QDD biped get-up after a fall: own policy_io (kind "getup", 36-D actor).
from .qdd_biped_getup_env_cfg import (
    getup_policy_io_onnx_metadata as _qdd_getup_policy_io_onnx_metadata,
    make_qdd_biped_getup_env_cfg,
    QddBipedGetupRlCfg,
)


class QddBipedGetupOnPolicyRunner(MicroduckOnPolicyRunner):
    """Attaches robot/qdd_biped/policy_io_getup.json to every checkpoint ONNX."""

    def save(self, path: str, infos=None):
        super().save(path, infos)
        _, _, onnx_path = self._get_export_paths(path)
        try:
            from mjlab.rl.exporter_utils import attach_metadata_to_onnx

            if onnx_path.exists():
                attach_metadata_to_onnx(str(onnx_path), _qdd_getup_policy_io_onnx_metadata())
        except Exception as e:  # never kill training over metadata
            print(f"[qdd_getup] policy_io metadata not attached: {e}")


register_mjlab_task(
    task_id="Mjlab-Getup-QddBiped",
    env_cfg=make_qdd_biped_getup_env_cfg(),
    play_env_cfg=make_qdd_biped_getup_env_cfg(play=True),
    rl_cfg=QddBipedGetupRlCfg,
    runner_cls=QddBipedGetupOnPolicyRunner,
)


# QDD biped skills (keyframe tracking from a JSON spec, tasks/qdd_skill.py). The
# skill is chosen by $QDD_SKILL_SPEC (path or name in robot/qdd_biped/skills/,
# default bow) when this package is imported. Same 41-D obs / 10-D action as the
# walking policy; the ONNX carries the skill's policy_io + spec.
from .qdd_biped_skill_env_cfg import (
    TASK_ID as _QDD_SKILL_TASK_ID,
    JUMP_SPEC as _QDD_JUMP_SPEC,
    JUMP_TASK_ID as _QDD_JUMP_TASK_ID,
    QddJumpRlCfg,
    QddSkillRlCfg,
    make_qdd_biped_skill_env_cfg,
    skill_onnx_metadata as _qdd_skill_onnx_metadata,
    warm_start_onnx_path as _qdd_skill_warm_start_onnx,
)


def init_actor_from_onnx(actor, onnx_path, normalizer_count: int = 100_000) -> list[str]:
    """Copy an exported actor (obs normaliser + MLP, mjlab/rsl_rl ONNX naming) into
    an rsl_rl MLPModel with the same layout. Returns the loaded tensor names."""
    import numpy as np
    import onnx
    import torch
    from onnx import numpy_helper

    m = onnx.load(str(onnx_path))
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    state = actor.state_dict()
    loaded = []
    with torch.no_grad():
        for name, value in init.items():
            if name.startswith("mlp.") and name in state:
                if tuple(state[name].shape) != value.shape:
                    raise ValueError(f"{name}: ONNX {value.shape} vs actor {tuple(state[name].shape)}")
                state[name].copy_(torch.from_numpy(np.array(value)))
                loaded.append(name)
        norm = getattr(actor, "obs_normalizer", None)
        mean = init.get("obs_normalizer._mean")
        if norm is not None and mean is not None and hasattr(norm, "_mean"):
            # (x - mean) / (std + eps): the Div constant is std + eps
            div = next(
                (init[n.input[1]] for n in m.graph.node if n.op_type == "Div" and n.input[1] in init),
                None,
            )
            if div is None:
                raise ValueError("ONNX has a normaliser mean but no Div constant")
            std = torch.from_numpy(np.array(div)).to(norm._std) - norm.eps
            norm._mean.copy_(torch.from_numpy(np.array(mean)).to(norm._mean))
            norm._std.copy_(std)
            norm._var.copy_(std.square())
            norm.count.fill_(normalizer_count)
            loaded.append("obs_normalizer")
    expected = {k for k in state if k.startswith("mlp.")}
    if not expected.issubset(loaded):
        raise ValueError(f"ONNX does not cover the actor MLP: missing {sorted(expected - set(loaded))}")
    return loaded


class QddSkillOnPolicyRunner(MicroduckOnPolicyRunner):
    """Skill runner: warm-starts the actor from the walking ONNX (if present, see
    qdd_biped_skill_env_cfg.WARM_START_ONNX_ENV) and embeds the skill's policy_io
    + spec in every exported ONNX. A later ``load()`` (resume / play / export)
    overrides the warm start as usual."""

    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        path = _qdd_skill_warm_start_onnx()
        try:
            feedforward = env.unwrapped.command_manager.get_term("skill").spec.get("_table") is not None
        except Exception:
            feedforward = False
        if feedforward:
            # residual policy around a feed-forward table: from scratch, output ~ 0 (= the table alone)
            import torch

            from .qdd_biped_skill_env_cfg import REF_OUTPUT_INIT_GAIN

            last = [m for m in self.alg.actor.modules() if isinstance(m, torch.nn.Linear)][-1]
            with torch.no_grad():
                last.weight.mul_(REF_OUTPUT_INIT_GAIN)
                last.bias.zero_()
            print(f"[skill] feed-forward skill: actor from scratch, output layer x {REF_OUTPUT_INIT_GAIN}")
            path = None
        elif path is not None:
            try:
                loaded = init_actor_from_onnx(self.alg.actor, path)
                print(f"[skill] warm start: actor initialised from {path} ({len(loaded)} tensors)")
            except Exception as e:  # a mismatch must not silently train garbage
                raise RuntimeError(f"[skill] warm start from {path} failed: {e}") from e
        elif not feedforward:
            print("[skill] no warm start (training the actor from scratch)")

    def save(self, path: str, infos=None):
        super().save(path, infos)
        _, _, onnx_path = self._get_export_paths(path)
        try:
            from mjlab.rl.exporter_utils import attach_metadata_to_onnx

            if onnx_path.exists():
                cmd = self.env.unwrapped.command_manager.get_term("skill")
                attach_metadata_to_onnx(str(onnx_path), _qdd_skill_onnx_metadata(cmd.spec))
        except Exception as e:  # never kill training over metadata
            print(f"[WARN] skill ONNX metadata failed: {e}")


register_mjlab_task(
    task_id=_QDD_SKILL_TASK_ID,
    env_cfg=make_qdd_biped_skill_env_cfg(),
    play_env_cfg=make_qdd_biped_skill_env_cfg(play=True),
    rl_cfg=QddSkillRlCfg,
    runner_cls=QddSkillOnPolicyRunner,
)

# Jump skill (spec schema 2 with a flight phase): its own task id so yatai trains it
# without $QDD_SKILL_SPEC (skills/hop.json).
register_mjlab_task(
    task_id=_QDD_JUMP_TASK_ID,
    env_cfg=make_qdd_biped_skill_env_cfg(_QDD_JUMP_SPEC),
    play_env_cfg=make_qdd_biped_skill_env_cfg(_QDD_JUMP_SPEC, play=True),
    rl_cfg=QddJumpRlCfg,
    runner_cls=QddSkillOnPolicyRunner,
)


# QDD biped WITH ARMS (tasks/qdd_biped_arms.py): walking, rough walking, the jump
# and the get-up, all with the arms in the policy (14-D action; 53-D obs, get-up
# 48-D), torque-speed envelope actuators and self-colliding motor housings. The
# ONNX carries the leg-only policy_io with the arms appended.
from .qdd_biped_arms import (
    GETUP_ARMS_TASK_ID as _QDD_GETUP_ARMS_TASK_ID,
    JUMP_ARMS_TASK_ID as _QDD_JUMP_ARMS_TASK_ID,
    ROUGH_ARMS_TASK_ID as _QDD_ROUGH_ARMS_TASK_ID,
    VELOCITY_ARMS_TASK_ID as _QDD_VELOCITY_ARMS_TASK_ID,
    arms_policy_io_onnx_metadata as _qdd_arms_policy_io_onnx_metadata,
    getup_arms_policy_io_onnx_metadata as _qdd_getup_arms_policy_io_onnx_metadata,
    GETUP_SUPINE_ARMS_TASK_ID as _QDD_GETUP_SUPINE_ARMS_TASK_ID,
    GETUP_SUPINE_INIT_STD as _QDD_GETUP_SUPINE_INIT_STD,
    getup_supine_arms_policy_io_onnx_metadata as _qdd_getup_supine_arms_policy_io_onnx_metadata,
    make_qdd_biped_getup_supine_arms_env_cfg,
    GETUP_CHAIN_ARMS_TASK_ID as _QDD_GETUP_CHAIN_ARMS_TASK_ID,
    getup_chain_arms_policy_io_onnx_metadata as _qdd_getup_chain_arms_policy_io_onnx_metadata,
    make_qdd_biped_getup_chain_arms_env_cfg,
    make_qdd_biped_getup_arms_env_cfg,
    make_qdd_biped_jump_arms_env_cfg,
    make_qdd_biped_rough_arms_env_cfg,
    make_qdd_biped_velocity_arms_env_cfg,
)
from .symmetry import QDD_ARMS_SYMMETRY_CFG as _QDD_ARMS_SYMMETRY_CFG


def _attach_onnx_metadata(runner, path: str, metadata_fn, tag: str) -> None:
    _, _, onnx_path = runner._get_export_paths(path)
    try:
        from mjlab.rl.exporter_utils import attach_metadata_to_onnx

        if onnx_path.exists():
            attach_metadata_to_onnx(str(onnx_path), metadata_fn())
    except Exception as e:  # never kill training over metadata
        print(f"[WARN] {tag} ONNX metadata failed: {e}")


class QddBipedArmsOnPolicyRunner(MicroduckOnPolicyRunner):
    """Walking / rough walking with arms: the walking policy_io with the arms."""

    def save(self, path: str, infos=None):
        super().save(path, infos)
        _attach_onnx_metadata(
            self, path, lambda: _qdd_arms_policy_io_onnx_metadata(_QDD_VELOCITY_ARMS_TASK_ID), "policy_io (arms)"
        )


class QddRoughArmsOnPolicyRunner(MicroduckOnPolicyRunner):
    """Rough walking with arms: the walking policy_io with the arms."""

    def save(self, path: str, infos=None):
        super().save(path, infos)
        _attach_onnx_metadata(
            self, path, lambda: _qdd_arms_policy_io_onnx_metadata(_QDD_ROUGH_ARMS_TASK_ID), "policy_io (arms)"
        )


class QddSkillArmsOnPolicyRunner(MicroduckOnPolicyRunner):
    """Jump with arms: residual around the leg table from scratch (output layer
    scaled like the leg-only feed-forward skill), the jump policy_io with the arms."""

    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        import torch

        from .qdd_biped_skill_env_cfg import REF_OUTPUT_INIT_GAIN

        last = [m for m in self.alg.actor.modules() if isinstance(m, torch.nn.Linear)][-1]
        with torch.no_grad():
            last.weight.mul_(REF_OUTPUT_INIT_GAIN)
            last.bias.zero_()
        print(f"[skill arms] actor from scratch, output layer x {REF_OUTPUT_INIT_GAIN}")

    def save(self, path: str, infos=None):
        super().save(path, infos)
        spec = self.env.unwrapped.command_manager.get_term("skill").spec
        _attach_onnx_metadata(
            self, path, lambda: _qdd_arms_policy_io_onnx_metadata(_QDD_JUMP_ARMS_TASK_ID, spec), "policy_io (arms)"
        )


class QddBipedGetupArmsOnPolicyRunner(MicroduckOnPolicyRunner):
    """Attaches robot/qdd_biped/policy_io_getup_arms.json to every checkpoint ONNX."""

    def save(self, path: str, infos=None):
        super().save(path, infos)
        _attach_onnx_metadata(self, path, _qdd_getup_arms_policy_io_onnx_metadata, "policy_io (getup arms)")


class QddBipedGetupSupineArmsOnPolicyRunner(MicroduckOnPolicyRunner):
    """Attaches the supine get-up policy_io (with its reference) to every checkpoint ONNX."""

    def save(self, path: str, infos=None):
        super().save(path, infos)
        _attach_onnx_metadata(
            self, path, _qdd_getup_supine_arms_policy_io_onnx_metadata, "policy_io (getup supine arms)"
        )


class QddBipedGetupChainArmsOnPolicyRunner(MicroduckOnPolicyRunner):
    """Attaches the chained get-up policy_io (roll + get-up reference) to every checkpoint ONNX."""

    def save(self, path: str, infos=None):
        super().save(path, infos)
        _attach_onnx_metadata(
            self, path, _qdd_getup_chain_arms_policy_io_onnx_metadata, "policy_io (getup chain arms)"
        )


def _arms_rl(base, **kw):
    alg = base.algorithm
    sym = alg.symmetry_cfg
    if sym is not None:
        alg = replace(alg, symmetry_cfg=dict(_QDD_ARMS_SYMMETRY_CFG))
    return replace(base, algorithm=alg, **kw)


register_mjlab_task(
    task_id=_QDD_VELOCITY_ARMS_TASK_ID,
    env_cfg=make_qdd_biped_velocity_arms_env_cfg(),
    play_env_cfg=make_qdd_biped_velocity_arms_env_cfg(play=True),
    rl_cfg=_arms_rl(QddBipedRlCfg, experiment_name="qdd_biped_arms", run_name="qdd_biped_arms"),
    runner_cls=QddBipedArmsOnPolicyRunner,
)
register_mjlab_task(
    task_id=_QDD_ROUGH_ARMS_TASK_ID,
    env_cfg=make_qdd_biped_rough_arms_env_cfg(),
    play_env_cfg=make_qdd_biped_rough_arms_env_cfg(play=True),
    rl_cfg=_arms_rl(
        QddBipedRlCfg, experiment_name="qdd_biped_rough_arms", run_name="qdd_biped_rough_arms", max_iterations=3_000
    ),
    runner_cls=QddRoughArmsOnPolicyRunner,
)
register_mjlab_task(
    task_id=_QDD_JUMP_ARMS_TASK_ID,
    env_cfg=make_qdd_biped_jump_arms_env_cfg(),
    play_env_cfg=make_qdd_biped_jump_arms_env_cfg(play=True),
    rl_cfg=_arms_rl(QddJumpRlCfg, experiment_name="qdd_jump_arms", run_name="qdd_jump_arms"),
    runner_cls=QddSkillArmsOnPolicyRunner,
)
register_mjlab_task(
    task_id=_QDD_GETUP_ARMS_TASK_ID,
    env_cfg=make_qdd_biped_getup_arms_env_cfg(),
    play_env_cfg=make_qdd_biped_getup_arms_env_cfg(play=True),
    rl_cfg=replace(QddBipedGetupRlCfg, experiment_name="qdd_biped_getup_arms", run_name="qdd_biped_getup_arms"),
    runner_cls=QddBipedGetupArmsOnPolicyRunner,
)
register_mjlab_task(
    task_id=_QDD_GETUP_SUPINE_ARMS_TASK_ID,
    env_cfg=make_qdd_biped_getup_supine_arms_env_cfg(),
    play_env_cfg=make_qdd_biped_getup_supine_arms_env_cfg(play=True),
    rl_cfg=replace(
        QddBipedGetupRlCfg,
        actor=replace(
            QddBipedGetupRlCfg.actor,
            distribution_cfg={
                **QddBipedGetupRlCfg.actor.distribution_cfg,
                "init_std": _QDD_GETUP_SUPINE_INIT_STD,
            },
        ),
        experiment_name="qdd_biped_getup_supine_arms",
        run_name="qdd_biped_getup_supine_arms",
    ),
    runner_cls=QddBipedGetupSupineArmsOnPolicyRunner,
)
register_mjlab_task(
    task_id=_QDD_GETUP_CHAIN_ARMS_TASK_ID,
    env_cfg=make_qdd_biped_getup_chain_arms_env_cfg(),
    play_env_cfg=make_qdd_biped_getup_chain_arms_env_cfg(play=True),
    rl_cfg=replace(
        QddBipedGetupRlCfg,
        actor=replace(
            QddBipedGetupRlCfg.actor,
            distribution_cfg={
                **QddBipedGetupRlCfg.actor.distribution_cfg,
                "init_std": _QDD_GETUP_SUPINE_INIT_STD,
            },
        ),
        experiment_name="qdd_biped_getup_chain_arms",
        run_name="qdd_biped_getup_chain_arms",
    ),
    runner_cls=QddBipedGetupChainArmsOnPolicyRunner,
)
