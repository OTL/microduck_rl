"""QDD biped with arms (2026-10-18): the task variants on the robot with arms.

The arms (robot/qdd_biped_constants.py add_arms: shoulder pitch + elbow each,
GIM3510-8, 0.64 kg per arm; 6.61 -> 7.89 kg) are there so the robot can get up
with both feet together (design study: scripts/qdd_getup_search/). Two kinds of
variant:

  - arms HELD (walking, rough walking, jump): the policy has the same 41-D obs /
    10-D action as without arms (legs only), so the controller runs it like the
    leg-only policy; the arm joints are held at 0 (hanging down) by their own PD
    (kp ARM_KP, kd ARM_KD), which the controller must do too (``held_joints`` in
    the ONNX metadata). Trained again because the arms change the mass (+1.28 kg)
    and its distribution.
  - arms IN THE POLICY (get-up): 14 joints in the obs and the action (48-D obs:
    the leg-only get-up's 36-D + 3 x 4 arm terms), its own policy_io
    (policy_io_getup_arms.json), fallen-state library and demo with arms
    (getup_states_arms.npz from scripts/qdd_getup_states.py --arms;
    getup_demo_traj_arms.npz from scripts/qdd_getup_search/export_arms_demo.py:
    the supine get-up found by the search, 31/32 open loop with the GIM4310-10
    parameters, 1/32 with the GIM3510-8 ones: a guide for the policy, not a motion
    to replay). No prone demo with arms: the prone get-up is left to the policy
    (the leg-only prone demo does not stand with arms).
"""

from __future__ import annotations

import json
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


def _arms_spec(inner, getup: bool):
    spec = inner()
    return rc.add_arms_getup(spec) if getup else rc.add_arms(spec)


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


def add_arms_to_env_cfg(cfg: ManagerBasedRlEnvCfg, policy_arms: bool, getup: bool = False) -> ManagerBasedRlEnvCfg:
    """Put the arms on the robot of a QDD biped env cfg (in place; returns it).

    The arm actuator copies the style (built-in PD or torque-speed envelope) and
    the delay of the legs' first actuator. ``policy_arms`` False: action, joint
    obs and the joint-set rewards stay on the legs; the arm targets stay at 0.
    ``getup``: the robot entity is the get-up collision model."""
    robot = cfg.scene.entities["robot"]
    robot.spec_fn = partial(_arms_spec, robot.spec_fn, getup)
    art = robot.articulation
    robot.articulation = replace(art, actuators=tuple(art.actuators) + (rc.arm_actuator_like(art.actuators[0]),))
    if policy_arms:
        return cfg
    legs = rc.QDD_LEG_JOINT_EXPR
    cfg.actions["joint_pos"].actuator_names = legs
    for group in cfg.observations.values():
        for name in ("joint_pos", "joint_vel"):
            if name in group.terms:
                group.terms[name].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=legs)
    for name in _LEG_REWARDS:
        if name in cfg.rewards:
            ac = cfg.rewards[name].params.get("asset_cfg")
            if ac is None or not ac.joint_names or tuple(ac.joint_names) == (".*",):
                cfg.rewards[name].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=legs)
    return cfg


def make_qdd_biped_velocity_arms_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    from mjlab_microduck.tasks.qdd_biped_velocity_env_cfg import make_qdd_biped_velocity_env_cfg

    return add_arms_to_env_cfg(make_qdd_biped_velocity_env_cfg(play=play), policy_arms=False)


def make_qdd_biped_rough_arms_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    from mjlab_microduck.tasks.qdd_biped_rough_env_cfg import make_qdd_biped_rough_env_cfg

    return add_arms_to_env_cfg(make_qdd_biped_rough_env_cfg(play=play), policy_arms=False)


def make_qdd_biped_jump_arms_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    from mjlab_microduck.tasks.qdd_biped_skill_env_cfg import JUMP_SPEC, make_qdd_biped_skill_env_cfg

    return add_arms_to_env_cfg(make_qdd_biped_skill_env_cfg(JUMP_SPEC, play=play), policy_arms=False)


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
def build_getup_arms_policy_io() -> dict:
    from mjlab_microduck.tasks.qdd_biped_getup_env_cfg import build_getup_policy_io

    io = build_getup_policy_io()
    names = list(rc.QDD_ARMS_JOINT_NAMES)
    n, na = len(names), len(rc.QDD_ARM_JOINT_NAMES)
    io["task"] = GETUP_ARMS_TASK_ID
    io["onnx"]["output"]["shape"] = [1, n]
    act = io["action"]
    act["dim"] = n
    act["joint_order"] = names
    act["default_joint_pos"] = list(act["default_joint_pos"]) + [0.0] * na
    acts = io["actuators"]
    acts["joint_order"] = names
    acts["kp_nm_per_rad"] = list(acts["kp_nm_per_rad"]) + [rc.ARM_KP] * na
    acts["kd_nm_s_per_rad"] = list(acts["kd_nm_s_per_rad"]) + [rc.ARM_KD] * na
    acts["effort_limit_nm"] = list(acts["effort_limit_nm"]) + [rc.ARM_PEAK] * na
    m = rc.get_qdd_biped_arms_spec().compile()
    acts["joint_pos_limits_rad"] = [[round(float(x), 6) for x in m.joint(j).range] for j in names]
    start = 0
    for term in io["observation"]["terms"]:
        if term["name"] in ("joint_pos", "joint_vel", "actions"):
            term["size"] = n
        term["start"] = start
        start += term["size"]
    io["observation"]["dim"] = start
    io["onnx"]["input"]["shape"] = [1, start]
    io["arms"] = {
        "note": "design-study arms (robot/qdd_biped_constants.py add_arms): not in the robot repo's model yet",
        "motor": "GIM3510-8",
    }
    return io


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
