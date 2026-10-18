"""QDD biped with arms (tasks/qdd_biped_arms.py): model, task cfgs, get-up data."""

import json

import numpy as np
import pytest

import mjlab_microduck.tasks  # noqa: F401  (register the tasks first, as the other tests do)
from mjlab_microduck.robot import qdd_biped_constants as rc


def test_arms_model_joints_and_mass():
    m = rc.get_qdd_biped_arms_spec().compile()
    names = tuple(m.joint(i).name for i in range(1, m.njnt))
    assert names == rc.QDD_ARMS_JOINT_NAMES  # legs first, then the arms
    base = rc.get_qdd_biped_spec().compile().body_subtreemass[1]
    added = m.body_subtreemass[1] - base
    per_arm = 2 * rc.ARM_MOTOR["mass"] + rc.ARM_UPPER_MASS + rc.ARM_FORE_MASS + rc.ARM_SHOULDER_MOUNT_MASS
    assert abs(added - 2 * per_arm) < 1e-6  # CAD v2: about 0.74 kg per arm
    assert abs(per_arm - 0.743) < 0.01
    for n in ("l_shoulder_pitch", "r_shoulder_pitch"):
        assert tuple(m.joint(n).range) == pytest.approx(rc.ARM_SHOULDER_RANGE)
    # arm links are collision shapes (falls, self-collision)
    geoms = {m.geom(i).name for i in range(m.ngeom)}
    for s in "lr":
        assert {f"{s}_upper_arm_collision", f"{s}_forearm_collision"} <= geoms


def test_getup_arms_model_arm_motors_collide():
    m = rc.get_qdd_biped_getup_arms_spec().compile()
    names = [m.geom(i).name for i in range(m.ngeom)]
    motors = [n for n in names if n.startswith(rc.QDD_ARM_BODIES) and "_motor" in n]
    assert len(motors) == 4
    for n in motors:
        g = m.geom(n)
        assert g.contype == rc.GETUP_SELF_CONTYPE and g.conaffinity == rc.GETUP_SELF_CONAFFINITY


def test_arm_actuator_follows_the_leg_style():
    from mjlab_microduck.actuator.qdd_motor import QddMotorActuatorCfg

    env = rc.arm_actuator_like(rc.qdd_envelope_actuators()[0])
    assert isinstance(env, QddMotorActuatorCfg) and env.motor == "3510"
    assert env.stiffness == rc.ARM_KP and env.effort_limit == rc.ARM_PEAK and env.damping == 0.0  # kd in the joint
    builtin = rc.arm_actuator_like(rc.QDD_ACTUATOR_HIP_YAW_ROLL)
    assert not isinstance(builtin, QddMotorActuatorCfg)
    assert builtin.target_names_expr == (r".*_shoulder_pitch", r".*_elbow")


@pytest.mark.parametrize(
    "factory",
    ["make_qdd_biped_velocity_arms_env_cfg", "make_qdd_biped_rough_arms_env_cfg", "make_qdd_biped_jump_arms_env_cfg"],
)
def test_arms_variants_have_the_arms_in_the_policy(factory):
    from mjlab_microduck.actuator.qdd_motor import QddMotorActuatorCfg
    from mjlab_microduck.tasks import qdd_biped_arms as A

    cfg = getattr(A, factory)()
    names = cfg.actions["joint_pos"].actuator_names
    assert tuple(names) != rc.QDD_LEG_JOINT_EXPR  # not restricted to the legs
    robot = cfg.scene.entities["robot"]
    acts = robot.articulation.actuators
    assert acts[-1].target_names_expr == (r".*_shoulder_pitch", r".*_elbow")
    # torque-speed envelope on every joint, the arms as GIM3510-8
    assert all(isinstance(a, QddMotorActuatorCfg) for a in acts) and acts[-1].motor == "3510"
    m = robot.spec_fn().compile()
    assert m.nq == 7 + 14
    # motor housings collide with the robot too
    motors = [m.geom(i) for i in range(m.ngeom) if "_motor" in m.geom(i).name and m.geom(i).name.endswith("_collision")]
    assert motors and all(g.contype[0] & 1 and g.conaffinity[0] & 1 for g in motors)
    # elbows bent forward by default (OTL 2026-10-18)
    assert dict(robot.init_state.joint_pos)[r".*_elbow"] == pytest.approx(A.ARM_WALK_ELBOW0) and A.ARM_WALK_ELBOW0 > 0.5
    if "pose" in cfg.rewards:
        assert cfg.rewards["arm_swing"].weight > 0  # walking: the swing reference
    else:
        assert cfg.rewards["arm_pose"].weight > 0


def test_arms_symmetry_round_trip():
    import torch
    from tensordict import TensorDict

    from mjlab_microduck.tasks import symmetry as S

    obs = torch.randn(5, len(S.QDD_ARMS_OBS_PERM))
    td = TensorDict({"actor": obs, "critic": torch.randn(5, 3)}, batch_size=[5])
    act = torch.randn(5, 14)
    o, a = S.qdd_biped_arms_symmetry(None, td, act)
    mo, ma = o["actor"][5:], a[5:]
    td2 = TensorDict({"actor": mo, "critic": torch.randn(5, 3)}, batch_size=[5])
    o2, a2 = S.qdd_biped_arms_symmetry(None, td2, ma)
    assert torch.allclose(o2["actor"][5:], obs) and torch.allclose(a2[5:], act)
    # the arm actions swap l <-> r with no sign change
    assert torch.equal(ma[:, 10:], act[:, [12, 13, 10, 11]])
    assert len(S.QDD_ARMS_OBS_PERM) == 53 and sorted(S.QDD_ARMS_OBS_PERM) == list(range(53))


@pytest.mark.parametrize("task", ["Mjlab-Velocity-Flat-QddBipedArms", "Mjlab-Skill-Hop-QddBipedArms"])
def test_arms_policy_io(task):
    from mjlab_microduck.tasks import qdd_biped_arms as A

    io = json.loads(A.arms_policy_io_onnx_metadata(task)["policy_io"])
    assert io["action"]["dim"] == 14 and io["action"]["joint_order"] == list(rc.QDD_ARMS_JOINT_NAMES)
    assert io["observation"]["dim"] == 53 == io["onnx"]["input"]["shape"][1]
    sizes = {t["name"]: t["size"] for t in io["observation"]["terms"]}
    assert sizes["joint_pos"] == sizes["joint_vel"] == sizes["actions"] == 14


def test_getup_arms_cfg_and_policy_io():
    from mjlab_microduck.tasks import qdd_biped_arms as A

    cfg = A.make_qdd_biped_getup_arms_env_cfg()
    reset = cfg.events["reset_getup"].params
    assert reset["states_path"] == str(rc.QDD_GETUP_ARMS_STATES_NPZ)
    assert "bridge_prone" not in reset["kind_probs"] and "bridge_prone" not in reset["frontiers"]
    assert abs(sum(reset["kind_probs"].values()) - 1.0) < 1e-9
    assert cfg.rewards["demo_tracking"].params["demo_path"] == str(rc.QDD_GETUP_ARMS_DEMO_NPZ)
    io = A.build_getup_arms_policy_io()
    assert io["observation"]["dim"] == 48 and io["action"]["dim"] == 14
    assert io["action"]["joint_order"] == list(rc.QDD_ARMS_JOINT_NAMES)
    assert json.loads(rc.QDD_GETUP_ARMS_POLICY_IO_JSON.read_text()) == io  # regenerate: python -m ...qdd_biped_arms


def test_getup_arms_data_files():
    lib = np.load(rc.QDD_GETUP_ARMS_STATES_NPZ)
    assert tuple(str(n) for n in lib["joint_names"]) == rc.QDD_ARMS_JOINT_NAMES
    assert lib["joint_pos"].shape[1] == 14 and np.isfinite(lib["joint_pos"]).all()
    kinds = [str(k) for k in lib["kinds"]]
    present = {kinds[k] for k in np.unique(lib["kind"])}
    assert {"push", "stride", "drop", "crouch", "path", "bridge_supine"} <= present
    demo = np.load(rc.QDD_GETUP_ARMS_DEMO_NPZ)
    assert tuple(str(n) for n in demo["joint_names"]) == rc.QDD_ARMS_JOINT_NAMES
    assert demo["qpos"].shape[1] == 21 and set(np.unique(demo["demo"])) == {1}
    # ends standing at the default height
    assert abs(demo["qpos"][-1, 2] - rc.STAND_BASE_Z) < 0.03


def test_states_mirror_swaps_the_arms():
    import importlib.util
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "scripts" / "qdd_getup_states.py"
    spec = importlib.util.spec_from_file_location("qdd_getup_states", p)
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    j = np.arange(1, 15, dtype=float)
    out = S._swap_lr(j)
    assert list(out[10:]) == [13, 14, 11, 12]  # l arm <- r arm, no sign change
    assert list(out[:5]) == [-6, -7, 8, 9, 10]  # l leg <- r leg, hip yaw / roll negated
    assert np.allclose(S._swap_lr(out), j)


def test_no_self_contact_when_standing():
    """With the motor housings colliding with the robot (2026-10-18), the standing
    pose and the arms hanging down must not touch anything of the robot itself."""
    import re

    import mujoco

    m = rc.get_qdd_biped_getup_arms_spec().compile()
    d = mujoco.MjData(m)
    d.qpos[2], d.qpos[3] = 1.0, 1.0  # in the air: robot-robot contacts only
    for j in range(1, m.njnt):
        for k, v in rc.DEFAULT_JOINT_POS.items():
            if re.fullmatch(k, m.joint(j).name):
                d.qpos[m.jnt_qposadr[j]] = v
    mujoco.mj_forward(m, d)
    touching = [(m.geom(c.geom1).name, m.geom(c.geom2).name) for c in d.contact[: d.ncon] if c.dist < 0.0]
    assert touching == []


def test_arm_swing_target_against_the_legs():
    """Left arm furthest back (shoulder +) when the left foot lands (phi = swing
    fraction), right arm forward; standing / no command: the default pose."""
    import types

    import torch

    from mjlab_microduck.tasks import mdp as M
    from mjlab_microduck.tasks import qdd_biped_arms as A

    gait = M.GaitClockCommand.__new__(M.GaitClockCommand)
    gait.cfg = types.SimpleNamespace(swing_fraction=0.4)
    gait.phase = torch.tensor([[0.4], [0.9], [0.4]])
    gait.active = torch.tensor([True, True, False])
    twist = torch.tensor([[0.5, 0, 0], [0.5, 0, 0], [0.0, 0, 0]])
    env = types.SimpleNamespace(
        command_manager=types.SimpleNamespace(get_term=lambda n: gait, get_command=lambda n: twist)
    )
    orig = M._qdd_gait
    M._qdd_gait = lambda e, n: gait
    try:
        t = M.qdd_arm_swing_target(env, A.ARM_WALK_SHOULDER0, A.ARM_WALK_ELBOW0, A.ARM_SWING_AMP, 0.5, 0.2)
    finally:
        M._qdd_gait = orig
    s0 = A.ARM_WALK_SHOULDER0
    assert t[0, 0] == pytest.approx(s0 + A.ARM_SWING_AMP) and t[0, 2] == pytest.approx(s0 - A.ARM_SWING_AMP)
    assert t[1, 0] == pytest.approx(s0 - A.ARM_SWING_AMP)
    assert t[0, 3] > t[0, 1]  # the forward (right) arm bends its elbow more
    assert torch.allclose(t[2], torch.tensor([s0, A.ARM_WALK_ELBOW0, s0, A.ARM_WALK_ELBOW0]))


def test_elbow_bends_forward_shoulder_minus_is_forward():
    """OTL 2026-10-18: elbow + bends the forearm forward (+x, the walking direction,
    where the knees point); shoulder - swings the arm forward."""
    import mujoco

    m = rc.get_qdd_biped_arms_spec().compile()
    d = mujoco.MjData(m)

    def hand_x(sh, el):
        d.qpos[:] = 0
        d.qpos[3] = 1
        for nm in ("l", "r"):
            d.qpos[m.joint(f"{nm}_shoulder_pitch").qposadr[0]] = sh
            d.qpos[m.joint(f"{nm}_elbow").qposadr[0]] = el
        mujoco.mj_forward(m, d)
        out = []
        for nm in ("l", "r"):
            b = m.body(f"{nm}_forearm").id
            out.append((d.xpos[b] + d.xmat[b].reshape(3, 3) @ [0, 0, -0.15])[0] - d.xpos[m.body(f"{nm}_upper_arm").id][0])
        return np.array(out)

    assert (hand_x(0.0, 1.0) > 0.1).all()
    assert (hand_x(-0.5, 0.0) > 0.1).all()
    assert (hand_x(0.5, 0.0) < -0.1).all()
