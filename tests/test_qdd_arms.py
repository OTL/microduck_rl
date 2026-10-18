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


def test_getup_arms_model_has_floor_only_arm_motors():
    m = rc.get_qdd_biped_getup_arms_spec().compile()
    names = [m.geom(i).name for i in range(m.ngeom)]
    motors = [n for n in names if n.startswith(rc.QDD_ARM_BODIES) and "_motor" in n]
    assert len(motors) == 4
    for n in motors:
        g = m.geom(n)
        assert g.contype == rc.GETUP_FLOOR_BIT and g.conaffinity == 0


def test_arm_actuator_follows_the_leg_style():
    from mjlab_microduck.actuator.qdd_motor import QddMotorActuatorCfg

    env = rc.arm_actuator_like(rc.qdd_envelope_actuators()[0])
    assert isinstance(env, QddMotorActuatorCfg) and env.motor == "3510"
    assert env.stiffness == rc.ARM_KP and env.effort_limit == rc.ARM_PEAK
    builtin = rc.arm_actuator_like(rc.QDD_ACTUATOR_HIP_YAW_ROLL)
    assert not isinstance(builtin, QddMotorActuatorCfg)
    assert builtin.target_names_expr == (r".*_shoulder_pitch", r".*_elbow")


@pytest.mark.parametrize(
    "factory",
    ["make_qdd_biped_velocity_arms_env_cfg", "make_qdd_biped_rough_arms_env_cfg", "make_qdd_biped_jump_arms_env_cfg"],
)
def test_held_variants_keep_the_leg_policy(factory):
    from mjlab_microduck.tasks import qdd_biped_arms as A

    cfg = getattr(A, factory)()
    assert tuple(cfg.actions["joint_pos"].actuator_names) == rc.QDD_LEG_JOINT_EXPR
    for group in cfg.observations.values():
        for name in ("joint_pos", "joint_vel"):
            if name in group.terms:
                assert tuple(group.terms[name].params["asset_cfg"].joint_names) == rc.QDD_LEG_JOINT_EXPR
    robot = cfg.scene.entities["robot"]
    assert robot.articulation.actuators[-1].target_names_expr == (r".*_shoulder_pitch", r".*_elbow")
    m = robot.spec_fn().compile()
    assert m.nq == 7 + 14


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
