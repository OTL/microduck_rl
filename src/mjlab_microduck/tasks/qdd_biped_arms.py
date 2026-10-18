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


def make_qdd_biped_velocity_arms_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    from mjlab_microduck.tasks.qdd_biped_velocity_env_cfg import make_qdd_biped_velocity_env_cfg

    return add_arms_to_env_cfg(make_qdd_biped_velocity_env_cfg(play=play), policy_arms=True)


def make_qdd_biped_rough_arms_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    from mjlab_microduck.tasks.qdd_biped_rough_env_cfg import make_qdd_biped_rough_env_cfg

    return add_arms_to_env_cfg(make_qdd_biped_rough_env_cfg(play=play), policy_arms=True)


def make_qdd_biped_jump_arms_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    from mjlab_microduck.tasks.qdd_biped_skill_env_cfg import JUMP_SPEC, make_qdd_biped_skill_env_cfg

    from mjlab.managers.reward_manager import RewardTermCfg

    from mjlab_microduck.tasks import mdp as microduck_mdp

    # the jump already has envelope actuators (2.5 ms physics)
    cfg = add_arms_to_env_cfg(make_qdd_biped_skill_env_cfg(JUMP_SPEC, play=play), policy_arms=True)
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
# Policy I/O of the get-up with arms (robot/qdd_biped/policy_io_getup_arms.json).
# Regenerate with  uv run python -m mjlab_microduck.tasks.qdd_biped_arms
# =============================================================================
ENVELOPE_NOTE = (
    "trained with the motors' torque-speed envelope on every joint (actuator/qdd_motor.py: "
    "GIM6010-8 FOC voltage limit, GIM3510-8 linear to 0 at 42.4 rad/s x V / 24) and with the "
    "motor housings colliding with the robot itself"
)


def extend_policy_io_with_arms(io: dict, task: str) -> dict:
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
    act["default_joint_pos"] = list(act["default_joint_pos"]) + [0.0] * na
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
        ref["arms_note"] = "the table has the legs only: the arms' reference is 0 (hanging), dq = tau_ff = 0"
    io["arms"] = {
        "note": "arms (robot/qdd_biped_constants.py add_arms, CAD v2): not in the robot repo's model yet",
        "motor": "GIM3510-8",
        "length_m": rc.ARM_LENGTH,
    }
    return io


def build_getup_arms_policy_io() -> dict:
    from mjlab_microduck.tasks.qdd_biped_getup_env_cfg import build_getup_policy_io

    return extend_policy_io_with_arms(build_getup_policy_io(), GETUP_ARMS_TASK_ID)


def build_walking_arms_policy_io(task: str = VELOCITY_ARMS_TASK_ID) -> dict:
    from mjlab_microduck.tasks.qdd_biped_velocity_env_cfg import build_policy_io

    return extend_policy_io_with_arms(build_policy_io(), task)


def build_jump_arms_policy_io(spec: dict | None = None) -> dict:
    from mjlab_microduck.tasks import qdd_skill
    from mjlab_microduck.tasks.qdd_biped_skill_env_cfg import JUMP_SPEC, build_skill_policy_io

    if spec is None:
        spec = qdd_skill.load_spec(JUMP_SPEC)
    return extend_policy_io_with_arms(build_skill_policy_io(spec), JUMP_ARMS_TASK_ID)


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


def held_joints_onnx_metadata() -> dict[str, str]:
    return {"held_joints": json.dumps(held_joints_metadata())}


if __name__ == "__main__":
    print("written", write_getup_arms_policy_io())
