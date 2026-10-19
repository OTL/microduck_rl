"""QDD biped with arms (2026-10-18): the task variants on the robot with arms.

The arms (robot/qdd_biped_constants.py add_arms: shoulder pitch + elbow each,
GIM3510-8, 25 cm, 0.74 kg per arm (CAD v2); 6.61 -> 8.10 kg) are there so the
robot can get up with both feet together (design study: scripts/qdd_getup_search/).

All four tasks have the arms IN THE POLICY (2026-10-18 evening, OTL: "歩行などでは
腕も使うようにして"): 14 joints in the obs and the action (walking / rough / jump:
the 41-D leg-only obs + 3 x 4 arm terms = 53-D; get-up: 36-D + 12 = 48-D), each with
its policy_io (the leg-only one with the arms appended: extend_policy_io_with_arms).
Walking keeps the mirror loss with the arms in the table (symmetry.py
qdd_biped_arms_symmetry); a loose arm posture term keeps them from flailing.

Physics, same for all four (OTL: limbs went through each other and moved too fast
in the first get-up videos):
  - torque-speed envelope actuators (actuator/qdd_motor.py) on every joint, the
    arms as GIM3510-8, instead of the plain torque clamp;
  - the motor housings (and the torso top) collide with the robot itself, not only
    with the floor (robot/qdd_biped_constants.py GETUP_SELF_CONTYPE), in every
    arms model.

Get-up: its own fallen-state library and demo with arms (getup_states_arms.npz
from scripts/qdd_getup_states.py --arms; getup_demo_traj_arms.npz from
scripts/qdd_getup_search/export_arms_demo.py: the supine get-up found by the
search on 20 cm arms, a guide for the policy, not a motion to replay). No prone
demo with arms: the prone get-up is left to the policy.
"""

from __future__ import annotations

import json
from copy import deepcopy
import re
from dataclasses import replace
from functools import partial

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.robot import qdd_biped_constants as rc

# Task ids
VELOCITY_ARMS_TASK_ID = "Mjlab-Velocity-Flat-QddBipedArms"
ROUGH_ARMS_TASK_ID = "Mjlab-Velocity-Rough-QddBipedArms"
JUMP_ARMS_TASK_ID = "Mjlab-Skill-Hop-QddBipedArms"
GETUP_ARMS_TASK_ID = "Mjlab-Getup-QddBipedArms"

# Get-up with arms: the start mix of the leg-only get-up without the prone bridge
# (no prone demo with arms); its share goes to the real falls.
GETUP_ARMS_KIND_PROBS = {
    "push": 0.15,
    "stride": 0.1,
    "drop": 0.1,
    "crouch": 0.1,
    "path": 0.3,
    "bridge_supine": 0.25,
}

# Rewards whose joint set follows the policy's joints when the arms are held.
_LEG_REWARDS = ("pose", "dof_pos_limits", "joint_torques")

# Arm posture (policy arms, walking / jump): loose Gaussian around hanging down, so
# the arms can swing and help balance but do not flail.
ARM_POSE_STD_STANDING = {r".*_shoulder_pitch": 0.3, r".*_elbow": 0.3}
ARM_POSE_STD_WALKING = {r".*_shoulder_pitch": 0.6, r".*_elbow": 0.5}
ARM_POSE_STD = 0.5  # rad, the jump's arm_pose term
ARM_POSE_WEIGHT = 0.2  # the jump's arm_pose term
IMPLICIT_KD_MIN_DT = 0.004  # s: physics steps this long put the legs' kd in the joints

# Arms while walking / jumping (OTL 2026-10-18: elbows bent forward, a reference):
# default pose (action offset, standing) and the swing reference
# (mdp.qdd_arm_swing_tracking: each shoulder against its own leg, amplitude ~ the
# commanded forward speed).
ARM_WALK_SHOULDER0 = 0.15  # rad, slightly back (shoulder + = back; OTL 2026-10-18)
ARM_WALK_ELBOW0 = 1.0  # rad (57 deg), forearm bent forward
ARM_SWING_AMP = 0.35  # rad (20 deg) each way at ARM_SWING_V_FULL
ARM_SWING_V_FULL = 0.5  # m/s, the top of the forward command range
ARM_SWING_ELBOW_AMP = 0.2  # rad more elbow bend while the arm is forward
ARM_SWING_STD = 0.25  # rad
ARM_SWING_WEIGHT = 1.0
ARM_WALK_HOME = {r".*_shoulder_pitch": ARM_WALK_SHOULDER0, r".*_elbow": ARM_WALK_ELBOW0}


def _arms_spec(inner, getup: bool, collide: bool = True, implicit_kd: bool = False, leg_kd: tuple = ()):
    spec = inner()
    if getup:
        rc.add_arms_getup(spec)
    else:
        rc.add_arms(spec)
        if collide:
            rc.add_getup_collision_geoms(spec)  # motor housings: floor + robot
    if implicit_kd:
        rc.add_arm_implicit_kd(spec)
    for expr, kd in leg_kd:  # the legs' kd as joint damping (see add_arms_to_env_cfg)
        for j in spec.joints:
            if any(re.fullmatch(e, j.name) for e in expr):
                d = list(j.damping)
                d[0] += kd
                j.damping = d
    return spec


def held_joints_metadata() -> dict:
    """What the controller must do with the arm joints of an arms-held policy."""
    return {
        "joints": list(rc.QDD_ARM_JOINT_NAMES),
        "target_rad": [0.0] * len(rc.QDD_ARM_JOINT_NAMES),
        "kp_nm_per_rad": rc.ARM_KP,
        "kd_nm_s_per_rad": rc.ARM_KD,
        "effort_limit_nm": rc.ARM_PEAK,
        "note": "not in the policy: hold at the target with this PD (arms hanging down)",
    }


def _envelope_like(acts: tuple) -> tuple:
    """The legs' actuators as torque-speed envelope actuators with the delays of
    the given ones (unchanged when they already are)."""
    from mjlab_microduck.actuator.qdd_motor import QddMotorActuatorCfg

    first = acts[0]
    if isinstance(first, QddMotorActuatorCfg):
        return tuple(acts)
    env = rc.qdd_envelope_actuators(delay_max_lag=getattr(first, "delay_max_lag", 2))
    return tuple(replace(a, delay_min_lag=getattr(first, "delay_min_lag", 0)) for a in env)


def add_arms_to_env_cfg(
    cfg: ManagerBasedRlEnvCfg,
    policy_arms: bool,
    getup: bool = False,
    envelope: bool = True,
    collide: bool = True,
) -> ManagerBasedRlEnvCfg:
    """Put the arms on the robot of a QDD biped env cfg (in place; returns it).

    ``envelope``: every joint gets the torque-speed envelope actuator (the arms as
    GIM3510-8). ``collide``: the motor housings collide with the robot too (the
    get-up model has them already). The arm actuator copies the style and the delay
    of the legs' first actuator. ``policy_arms`` False: action, joint obs and the
    joint-set rewards stay on the legs; the arm targets stay at 0. ``getup``: the
    robot entity is the get-up collision model."""
    from mjlab_microduck.actuator.qdd_motor import QddMotorActuatorCfg

    robot = cfg.scene.entities["robot"]
    art = robot.articulation
    legs = _envelope_like(art.actuators) if envelope else tuple(art.actuators)
    implicit_kd = isinstance(legs[0], QddMotorActuatorCfg)  # arm_actuator_like: kd in the joint then
    leg_kd: tuple = ()
    if implicit_kd and cfg.sim.mujoco.timestep >= IMPLICIT_KD_MIN_DT:
        # Explicit PD (the envelope actuator) at 5 ms: the light feet chatter at 15 rad/s
        # when they are in the air (lying robot). The kd goes into the joint damping,
        # which MuJoCo integrates implicitly; same torque while braking (the envelope
        # allows the full clamp there). The jump (2.5 ms) keeps the explicit kd: its bus
        # current model reads the actuator torque.
        leg_kd = tuple((tuple(a.target_names_expr), float(a.damping)) for a in legs)
        legs = tuple(replace(a, damping=0.0) for a in legs)
    robot.spec_fn = partial(_arms_spec, robot.spec_fn, getup, collide, implicit_kd, leg_kd)
    if implicit_kd and "randomize_motor_envelope" not in cfg.events:
        # as the jump: the driver clamp is fixed, the envelope (bus voltage, strength) varies
        from mjlab.managers import EventTermCfg

        from mjlab_microduck.tasks import mdp as microduck_mdp
        from mjlab_microduck.tasks.qdd_biped_skill_env_cfg import MOTOR_STRENGTH_RANGE, MOTOR_V_BUS_RANGE

        cfg.events.pop("randomize_effort_limits", None)
        cfg.events["randomize_motor_envelope"] = EventTermCfg(
            func=microduck_mdp.randomize_qdd_motor_envelope,
            mode="reset",
            params={"v_bus_range": MOTOR_V_BUS_RANGE, "strength_range": MOTOR_STRENGTH_RANGE},
        )
    robot.articulation = replace(art, actuators=legs + (rc.arm_actuator_like(legs[0]),))
    if collide or getup:
        # more contact shapes (motor housings, arms): room for a robot lying on the floor
        cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 120)
    if policy_arms:
        pose = cfg.rewards.get("pose")
        if pose is not None:
            for key, arm in (("std_standing", ARM_POSE_STD_STANDING), ("std_walking", ARM_POSE_STD_WALKING),
                             ("std_running", ARM_POSE_STD_WALKING)):
                if isinstance(pose.params.get(key), dict):
                    pose.params[key] = {**pose.params[key], **arm}
        return cfg
    legs_expr = rc.QDD_LEG_JOINT_EXPR
    cfg.actions["joint_pos"].actuator_names = legs_expr
    for group in cfg.observations.values():
        for name in ("joint_pos", "joint_vel"):
            if name in group.terms:
                group.terms[name].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=legs_expr)
    for name in _LEG_REWARDS:
        if name in cfg.rewards:
            ac = cfg.rewards[name].params.get("asset_cfg")
            if ac is None or not ac.joint_names or tuple(ac.joint_names) == (".*",):
                cfg.rewards[name].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=legs_expr)
    return cfg


# Walking with straighter knees (OTL 2026-10-19: "walk with the knees bent as little as
# possible"): the walking / rough default pose (action offset, reset pose, pose reward
# centre) has the knee at WALK_KNEE instead of 45 deg, feet flat (hip = ankle = -knee / 2),
# and the knee's pose kernel while walking is tighter. Torso height in that pose:
# WALK_BASE_Z (MuJoCo, soles on the floor; 45 deg: rc.STAND_BASE_Z 0.3542).
WALK_KNEE = 0.4  # rad (23 deg)
WALK_BASE_Z = 0.3688
WALK_LEG_HOME = {r".*_hip_pitch": -WALK_KNEE / 2, r".*_knee": WALK_KNEE, r".*_ankle_pitch": -WALK_KNEE / 2}
WALK_KNEE_STD_WALKING = 0.25  # rad (the leg-only walking task: 0.4)


def _set_walk_legs(cfg: ManagerBasedRlEnvCfg) -> None:
    robot = cfg.scene.entities["robot"]
    pos = (robot.init_state.pos[0], robot.init_state.pos[1], WALK_BASE_Z + (robot.init_state.pos[2] - rc.STAND_BASE_Z))
    robot.init_state = replace(
        robot.init_state, pos=pos, joint_pos={**dict(robot.init_state.joint_pos), **WALK_LEG_HOME}
    )
    pose = cfg.rewards["pose"]
    for key in ("std_walking", "std_running"):
        if isinstance(pose.params.get(key), dict):
            pose.params[key] = {**pose.params[key], r".*knee.*": WALK_KNEE_STD_WALKING}


def _walk_leg_default() -> list[float]:
    out = []
    for n in rc.QDD_JOINT_NAMES:
        v = 0.0
        for k, x in WALK_LEG_HOME.items():
            if re.fullmatch(k, n):
                v = x
        out.append(v)
    return out


def _set_arm_home(cfg: ManagerBasedRlEnvCfg) -> None:
    """Arms' default pose (bent elbows): the action offset and the reset pose."""
    robot = cfg.scene.entities["robot"]
    robot.init_state = replace(robot.init_state, joint_pos={**dict(robot.init_state.joint_pos), **ARM_WALK_HOME})


def _add_arm_swing(cfg: ManagerBasedRlEnvCfg) -> None:
    """Walking: the arm swing reference replaces the arms' share of the pose term."""
    from mjlab.managers.reward_manager import RewardTermCfg

    from mjlab_microduck.tasks import mdp as microduck_mdp

    pose = cfg.rewards["pose"]
    for key in ("std_standing", "std_walking", "std_running"):
        if isinstance(pose.params.get(key), dict):
            pose.params[key] = {**pose.params[key], r".*_shoulder_pitch": 10.0, r".*_elbow": 10.0}
    cfg.rewards["arm_swing"] = RewardTermCfg(
        func=microduck_mdp.qdd_arm_swing_tracking,
        weight=ARM_SWING_WEIGHT,
        params={
            "std": ARM_SWING_STD,
            "shoulder0": ARM_WALK_SHOULDER0,
            "elbow0": ARM_WALK_ELBOW0,
            "amp_max": ARM_SWING_AMP,
            "v_full": ARM_SWING_V_FULL,
            "elbow_amp": ARM_SWING_ELBOW_AMP,
        },
    )


def make_qdd_biped_velocity_arms_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    from mjlab_microduck.tasks.qdd_biped_velocity_env_cfg import make_qdd_biped_velocity_env_cfg

    cfg = add_arms_to_env_cfg(make_qdd_biped_velocity_env_cfg(play=play), policy_arms=True)
    _set_arm_home(cfg)
    _set_walk_legs(cfg)
    _add_arm_swing(cfg)
    return cfg


def make_qdd_biped_rough_arms_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    from mjlab_microduck.tasks.qdd_biped_rough_env_cfg import make_qdd_biped_rough_env_cfg

    cfg = add_arms_to_env_cfg(make_qdd_biped_rough_env_cfg(play=play), policy_arms=True)
    _set_arm_home(cfg)
    _set_walk_legs(cfg)
    _add_arm_swing(cfg)
    return cfg


def make_qdd_biped_jump_arms_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    from mjlab_microduck.tasks.qdd_biped_skill_env_cfg import JUMP_SPEC, make_qdd_biped_skill_env_cfg

    from mjlab.managers.reward_manager import RewardTermCfg

    from mjlab_microduck.tasks import mdp as microduck_mdp

    # the jump already has envelope actuators (2.5 ms physics)
    cfg = add_arms_to_env_cfg(make_qdd_biped_skill_env_cfg(JUMP_SPEC, play=play), policy_arms=True)
    _set_arm_home(cfg)  # the jump's arm reference (the default pose) has bent elbows too
    cfg.rewards["arm_pose"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=ARM_POSE_WEIGHT,
        params={
            "std": ARM_POSE_STD,
            "joint_indices": [rc.QDD_ARMS_JOINT_NAMES.index(n) for n in rc.QDD_ARM_JOINT_NAMES],
        },
    )
    return cfg


def make_qdd_biped_getup_arms_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    from mjlab_microduck.tasks.qdd_biped_getup_env_cfg import make_qdd_biped_getup_env_cfg

    cfg = add_arms_to_env_cfg(make_qdd_biped_getup_env_cfg(play=play), policy_arms=True, getup=True)
    reset = cfg.events["reset_getup"]
    reset.params["states_path"] = str(rc.QDD_GETUP_ARMS_STATES_NPZ)
    reset.params["kind_probs"] = dict(GETUP_ARMS_KIND_PROBS)
    reset.params["frontiers"] = {k: v for k, v in reset.params["frontiers"].items() if k != "bridge_prone"}
    cfg.rewards["demo_tracking"].params["demo_path"] = str(rc.QDD_GETUP_ARMS_DEMO_NPZ)
    if "path_frontier" in cfg.curriculum:
        steps = cfg.curriculum["path_frontier"].params.get("steps", {})
        steps.pop("bridge_prone", None)
    return cfg


# =============================================================================
# Supine get-up as a residual on the found motion (OTL 2026-10-19: "keep training
# from supine; from prone, roll over to supine first"). arms4/5_getup (the demo as a
# reward only) never stood from a real fall; the found joint-target sequence alone
# (getup_demo_traj_arms.npz ref_*, scripts/qdd_getup_search/export_arms_demo.py) stands
# from 29/32 supine falls in the search model. Here it is the feed-forward and the
# policy adds a residual (mdp.GetupReferenceJointPositionAction), observing the
# reference clock (mdp.getup_ref_phase). Lying-on-the-front starts are left out.
# =============================================================================
GETUP_SUPINE_ARMS_TASK_ID = "Mjlab-Getup-Supine-QddBipedArms"
GETUP_SUPINE_RESIDUAL_SCALE = 0.5  # rad per action unit (the from-scratch get-up: 0.75)
GETUP_SUPINE_INIT_STD = 0.3  # start near the feed-forward alone
GETUP_SUPINE_EPISODE_S = 8.0  # the motion takes 4.9 s, then standing
GETUP_SUPINE_RESIDUAL_WEIGHT = -0.02  # |a|^2: stay on the feed-forward unless it pays
GETUP_SUPINE_KIND_PROBS = {
    "push": 0.1,
    "stride": 0.1,
    "drop": 0.1,
    "crouch": 0.1,
    "path": 0.25,
    "bridge_supine": 0.35,
}


def make_qdd_biped_getup_supine_arms_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    from mjlab.managers import ObservationTermCfg, RewardTermCfg

    from mjlab_microduck.tasks import mdp as microduck_mdp

    cfg = make_qdd_biped_getup_arms_env_cfg(play=play)
    old = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = microduck_mdp.GetupReferenceJointPositionActionCfg(
        entity_name=old.entity_name,
        actuator_names=old.actuator_names,
        scale=GETUP_SUPINE_RESIDUAL_SCALE,
        use_default_offset=False,
        demo_path=str(rc.QDD_GETUP_ARMS_DEMO_NPZ),
    )
    cfg.episode_length_s = GETUP_SUPINE_EPISODE_S
    reset = cfg.events["reset_getup"]
    reset.params["kind_probs"] = dict(GETUP_SUPINE_KIND_PROBS)
    reset.params["supine_only"] = True
    phase = ObservationTermCfg(func=microduck_mdp.getup_ref_phase)
    cfg.observations["actor"].terms["ref_phase"] = phase
    cfg.observations["critic"].terms["ref_phase"] = deepcopy(phase)
    cfg.rewards["residual"] = RewardTermCfg(
        func=microduck_mdp.skill_residual_l2, weight=GETUP_SUPINE_RESIDUAL_WEIGHT
    )
    return cfg


# =============================================================================
# Chained get-up (OTL 2026-10-19: "from prone, roll sideways onto the back, then get
# up"): the supine task with a roll stage in front of its reference for starts lying
# on the front (mdp.GetupReferenceJointPositionAction roll_path; the sideways roll
# alone ends supine in every front start of the training sim). Front starts come
# back; the reference is the feed-forward, so the demo-tracking reward (which has
# no roll) is dropped.
# =============================================================================
GETUP_CHAIN_ARMS_TASK_ID = "Mjlab-Getup-Chain-QddBipedArms"
GETUP_CHAIN_EPISODE_S = 14.0  # roll 5.7 s + get-up 4.9 s, then standing
GETUP_CHAIN_KIND_PROBS = {
    "push": 0.15,
    "stride": 0.15,
    "drop": 0.15,
    "crouch": 0.1,
    "path": 0.2,
    "bridge_supine": 0.25,
}


def make_qdd_biped_getup_chain_arms_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_qdd_biped_getup_supine_arms_env_cfg(play=play)
    cfg.actions["joint_pos"].roll_path = str(rc.QDD_GETUP_ARMS_ROLL_NPZ)
    cfg.episode_length_s = GETUP_CHAIN_EPISODE_S
    reset = cfg.events["reset_getup"]
    reset.params["kind_probs"] = dict(GETUP_CHAIN_KIND_PROBS)
    reset.params["supine_only"] = False
    cfg.rewards.pop("demo_tracking", None)
    for name in [k for k, v in cfg.curriculum.items() if v.params.get("reward_name") == "demo_tracking"]:
        cfg.curriculum.pop(name)
    return cfg


# =============================================================================
# Policy I/O of the get-up with arms (robot/qdd_biped/policy_io_getup_arms.json).
# Regenerate with  uv run python -m mjlab_microduck.tasks.qdd_biped_arms
# =============================================================================
ENVELOPE_NOTE = (
    "trained with the motors' torque-speed envelope on every joint (actuator/qdd_motor.py: "
    "GIM6010-8 FOC voltage limit, GIM3510-8 linear to 0 at 42.4 rad/s x V / 24) and with the "
    "motor housings colliding with the robot itself"
)


def extend_policy_io_with_arms(io: dict, task: str, arm_default: list | None = None) -> dict:
    """A leg-only policy_io (10 joints) with the 4 arm joints appended: action,
    actuators (kp / kd / effort / limits) and every per-joint observation term
    (joint_pos, joint_vel, actions); the term offsets and the ONNX shapes follow."""
    names = list(rc.QDD_ARMS_JOINT_NAMES)
    n, na = len(names), len(rc.QDD_ARM_JOINT_NAMES)
    io["task"] = task
    io["onnx"]["output"]["shape"] = [1, n]
    act = io["action"]
    act["dim"] = n
    act["joint_order"] = names
    if arm_default is None:
        arm_default = [0.0] * na
    act["default_joint_pos"] = list(act["default_joint_pos"]) + [float(x) for x in arm_default]
    if isinstance(act.get("scale"), list):
        act["scale"] = list(act["scale"]) + [act["scale"][-1]] * na
    acts = io["actuators"]
    acts["joint_order"] = names
    acts["kp_nm_per_rad"] = list(acts["kp_nm_per_rad"]) + [rc.ARM_KP] * na
    acts["kd_nm_s_per_rad"] = list(acts["kd_nm_s_per_rad"]) + [rc.ARM_KD] * na
    acts["effort_limit_nm"] = list(acts["effort_limit_nm"]) + [rc.ARM_PEAK] * na
    m = rc.get_qdd_biped_arms_spec().compile()
    acts["joint_pos_limits_rad"] = [[round(float(x), 6) for x in m.joint(j).range] for j in names]
    acts.setdefault("torque_speed_envelope", ENVELOPE_NOTE)
    start = 0
    for term in io["observation"]["terms"]:
        if term["name"] in ("joint_pos", "joint_vel", "actions"):
            term["size"] = n
        term["start"] = start
        start += term["size"]
    io["observation"]["dim"] = start
    io["onnx"]["input"]["shape"] = [1, start]
    ref = io.get("skill", {}).get("reference")
    if isinstance(ref, dict):
        ref["arms_note"] = "the table has the legs only: the arms' reference is their default pose (elbows bent), dq = tau_ff = 0"
    io["arms"] = {
        "note": "arms (robot/qdd_biped_constants.py add_arms, CAD v2): not in the robot repo's model yet",
        "motor": "GIM3510-8",
        "length_m": rc.ARM_LENGTH,
    }
    return io


def _arm_walk_default() -> list[float]:
    return [ARM_WALK_SHOULDER0 if n.endswith("shoulder_pitch") else ARM_WALK_ELBOW0 for n in rc.QDD_ARM_JOINT_NAMES]


def build_getup_arms_policy_io() -> dict:
    from mjlab_microduck.tasks.qdd_biped_getup_env_cfg import build_getup_policy_io

    return extend_policy_io_with_arms(build_getup_policy_io(), GETUP_ARMS_TASK_ID)


def build_walking_arms_policy_io(task: str = VELOCITY_ARMS_TASK_ID) -> dict:
    from mjlab_microduck.tasks.qdd_biped_velocity_env_cfg import build_policy_io

    io = extend_policy_io_with_arms(build_policy_io(), task, _arm_walk_default())
    act = io["action"]
    act["default_joint_pos"] = _walk_leg_default() + list(act["default_joint_pos"][len(rc.QDD_JOINT_NAMES) :])
    act["walk_knee_note"] = f"straighter knees (OTL 2026-10-19): default knee {WALK_KNEE} rad, not 45 deg"
    return io


def build_jump_arms_policy_io(spec: dict | None = None) -> dict:
    from mjlab_microduck.tasks import qdd_skill
    from mjlab_microduck.tasks.qdd_biped_skill_env_cfg import JUMP_SPEC, build_skill_policy_io

    if spec is None:
        spec = qdd_skill.load_spec(JUMP_SPEC)
    io = extend_policy_io_with_arms(build_skill_policy_io(spec), JUMP_ARMS_TASK_ID, _arm_walk_default())
    return io


def arms_policy_io_onnx_metadata(task: str, spec: dict | None = None) -> dict[str, str]:
    """ONNX metadata for the walking / rough / jump policies with arms (generated,
    not a committed file: the leg-only contract + the arms)."""
    import hashlib

    if task == JUMP_ARMS_TASK_ID:
        io = build_jump_arms_policy_io(spec)
    else:
        io = build_walking_arms_policy_io(task)
    text = json.dumps(io, indent=2) + "\n"
    out = {"policy_io": text, "policy_io_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    if spec is not None and "_text" in spec:
        out["skill_spec"] = spec["_text"]
    return out


def write_getup_arms_policy_io(path=None) -> str:
    from pathlib import Path

    path = Path(path) if path is not None else rc.QDD_GETUP_ARMS_POLICY_IO_JSON
    path.write_text(json.dumps(build_getup_arms_policy_io(), indent=2) + "\n", encoding="utf-8", newline="\n")
    return str(path)


def getup_arms_policy_io_onnx_metadata() -> dict[str, str]:
    import hashlib

    text = rc.QDD_GETUP_ARMS_POLICY_IO_JSON.read_text(encoding="utf-8")
    return {"policy_io": text, "policy_io_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def build_getup_supine_arms_policy_io() -> dict:
    """The get-up-with-arms contract + the reference clock obs and the open-loop
    reference (generated, not a committed file)."""
    import numpy as np

    io = build_getup_arms_policy_io()
    io["task"] = GETUP_SUPINE_ARMS_TASK_ID
    io["kind"] = "getup_ref"
    obs = io["observation"]
    obs["terms"].append(
        {
            "name": "ref_phase",
            "start": obs["dim"],
            "size": 2,
            "source": "getup_ref_phase",
            "units": "-",
            "description": "s = reference clock / reference.t_end_s (0..1, 1 after the end): [sin(pi s), cos(pi s)]",
        }
    )
    obs["dim"] += 2
    dd = np.load(rc.QDD_GETUP_ARMS_DEMO_NPZ)
    names = [str(n) for n in dd["joint_names"]]
    order = [names.index(n) for n in io["action"]["joint_order"]]
    durs = [float(x) for x in dd["ref_durs_s"]]
    io["action"]["scale"] = GETUP_SUPINE_RESIDUAL_SCALE
    io["action"]["target"] = "q_target = reference(clock) + scale * action"
    io["reference"] = {
        "prep_s": float(dd["ref_prep_s"]),
        "keys_rad": [[float(v) for v in k[order]] for k in dd["ref_keys"]],
        "durs_s": durs,
        "t_end_s": float(dd["ref_prep_s"]) + sum(durs),
        "rule": "clock 0 at the start: ramp linearly from the start joint positions to keys_rad[0] in "
        "prep_s, then from each key to the next in durs_s, then hold keys_rad[-1]; the clock advances "
        "every control step",
    }
    io["getup"]["episode_length_s"] = GETUP_SUPINE_EPISODE_S
    io["getup"]["note"] = (
        "supine only (from lying on the front, roll over to supine first); " + io["getup"]["note"]
    )
    return io


def getup_supine_arms_policy_io_onnx_metadata() -> dict[str, str]:
    import hashlib

    text = json.dumps(build_getup_supine_arms_policy_io(), indent=2) + "\n"
    return {"policy_io": text, "policy_io_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def build_getup_chain_arms_policy_io() -> dict:
    """The supine contract + the roll stage in front of the reference for starts
    lying on the front."""
    import numpy as np

    io = build_getup_supine_arms_policy_io()
    io["task"] = GETUP_CHAIN_ARMS_TASK_ID
    rr = np.load(rc.QDD_GETUP_ARMS_ROLL_NPZ)
    names = [str(n) for n in rr["joint_names"]]
    order = [names.index(n) for n in io["action"]["joint_order"]]
    durs = [float(x) for x in rr["ref_durs_s"]]
    ref = io["reference"]
    t_roll = float(rr["ref_prep_s"]) + sum(durs)
    ref["roll"] = {
        "prep_s": float(rr["ref_prep_s"]),
        "keys_rad": [[float(v) for v in k[order]] for k in rr["ref_keys"]],
        "durs_s": durs,
        "t_end_s": t_roll,
        "when": "the torso lies on its front at the start: torso x axis pointing down, i.e. "
        "2 (x z - w y) < 0 for the torso quaternion (w, x, y, z)",
        "rule": "clock starts at -t_end_s; while the clock is < 0 the target is this table played the "
        "same way (from the start joint positions) at clock + t_end_s; the get-up stage then ramps from "
        "keys_rad[-1] of this table instead of the start joint positions",
    }
    for t in io["observation"]["terms"]:
        if t["name"] == "ref_phase":
            t["description"] = (
                "s = (clock + reference.roll.t_end_s) / (reference.roll.t_end_s + reference.t_end_s), "
                "clipped to 0..1: [sin(pi s), cos(pi s)] (the same denominator for every start)"
            )
    io["getup"]["episode_length_s"] = GETUP_CHAIN_EPISODE_S
    io["getup"]["note"] = io["getup"]["note"].replace(
        "supine only (from lying on the front, roll over to supine first); ",
        "from the front: the roll stage first, then the supine get-up; ",
    )
    return io


def getup_chain_arms_policy_io_onnx_metadata() -> dict[str, str]:
    import hashlib

    text = json.dumps(build_getup_chain_arms_policy_io(), indent=2) + "\n"
    return {"policy_io": text, "policy_io_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def held_joints_onnx_metadata() -> dict[str, str]:
    return {"held_joints": json.dumps(held_joints_metadata())}


if __name__ == "__main__":
    print("written", write_getup_arms_policy_io())
