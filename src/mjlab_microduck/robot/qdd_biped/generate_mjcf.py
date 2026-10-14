"""Generate qdd_biped.xml (training MJCF) from the QDD biped robot repo.

The robot repo's ``robot/model.py`` is the single source of truth for the
kinematics, masses, joint ranges and motor specs. This script walks the same
body tree as the repo's ``robot/build.py`` but writes an MJCF tailored for
mjlab training:

  - no <actuator> block: mjlab's actuator cfg (qdd_biped_constants.py) adds
    position-PD actuators with the per-joint gains / effort limits,
  - every collision geom is named ``*_collision`` (mjlab CollisionCfg regex),
  - cheap primitive self-collision proxies on the legs (thigh / shank capsules,
    knee-motor sphere) so self-collision can be sensed and penalised,
  - torso collision reduced to two boxes (shell + deck) instead of 8 overlapping
    component boxes,
  - IMU site on the torso and a site at the centre of each sole,
  - visual geoms (primitive cylinders / boxes) kept for videos, contype=0.

Usage (from the microduck_rl root):

    uv run python src/mjlab_microduck/robot/qdd_biped/generate_mjcf.py \
        --robot-repo /path/to/robot-repo

The generated XML is committed, so training machines do not need the robot repo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "qdd_biped.xml"


def _v(a) -> str:
    return " ".join(f"{x:.6g}" for x in a)


def _leg_proxies(body_name: str, s: int) -> list[str]:
    """Self-collision proxies for the thigh / shank (body frame: joint at origin, link along -z).

    thigh: hip-pitch axis -> knee axis is 0.13 m straight down. The knee motor
    (dia 80 x 44, centred at (0, s*0.038, -0.13) on the thigh) is approximated by a
    sphere. shank: knee axis -> ankle axis 0.13 m straight down. Radii are kept a
    little inside the real parts so nominal poses are contact-free.
    """
    side = body_name.split("_")[0]
    if body_name.endswith("_thigh"):
        return [
            f'<geom name="{side}_thigh_collision" class="collision" type="capsule" '
            f'fromto="0 0 -0.025 0 0 -0.095" size="0.02"/>',
            f'<geom name="{side}_knee_motor_collision" class="collision" type="sphere" '
            f'pos="0 {s * 0.038:.6g} -0.13" size="0.035"/>',
        ]
    if body_name.endswith("_shank"):
        return [
            f'<geom name="{side}_shank_collision" class="collision" type="capsule" '
            f'fromto="0 0 -0.035 0 0 -0.095" size="0.018"/>',
        ]
    return []


def _cyl_fromto(c) -> str:
    h = c.size[1] / 2
    d = np.array({"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[c.axis]) * h
    p = np.array(c.pos)
    return _v(np.concatenate([p - d, p + d]))


def _body(b, depth: int, p) -> list[str]:
    ind = "  " * depth
    I = b.inertia
    lines = [f'{ind}<body name="{b.name}" pos="{_v(b.pos)}">']
    if b.joint is None:
        lines.append(f'{ind}  <freejoint name="root"/>')
        lines.append(f'{ind}  <site name="imu" pos="0 0 0.1"/>')
    else:
        j = b.joint
        lines.append(
            f'{ind}  <joint name="{j.name}" axis="{_v(j.axis)}" range="{j.lower:.6g} {j.upper:.6g}" '
            f'armature="{j.spec["armature"]:.6g}"/>'
        )
    lines.append(
        f'{ind}  <inertial pos="{_v(b.com)}" mass="{b.mass:.6g}" '
        f'fullinertia="{I[0,0]:.6g} {I[1,1]:.6g} {I[2,2]:.6g} {I[0,1]:.6g} {I[0,2]:.6g} {I[1,2]:.6g}"/>'
    )
    for c in b.components:  # visual primitives (no collision)
        if c.shape == "box":
            lines.append(f'{ind}  <geom class="visual" type="box" pos="{_v(c.pos)}" '
                         f'size="{_v(np.array(c.size) / 2)}" rgba="{c.rgba}"/>')
        else:
            lines.append(f'{ind}  <geom class="visual" type="cylinder" fromto="{_cyl_fromto(c)}" '
                         f'size="{c.size[0]:.6g}" rgba="{c.rgba}"/>')
    if b.joint is None:  # torso: shell (incl. battery, Pi, boards) + wide deck plate
        lines.append(f'{ind}  <geom name="torso_collision" class="collision" type="box" '
                     f'pos="0 0 0.0985" size="0.05 0.08 0.0525"/>')
        lines.append(f'{ind}  <geom name="torso_deck_collision" class="collision" type="box" '
                     f'pos="0 0 0.04" size="0.05 0.111 0.006"/>')
    elif b.name.endswith("_foot"):
        side = b.name.split("_")[0]
        sole_z = -p.ankle_to_sole
        lines.append(
            f'{ind}  <geom name="{side}_foot_collision" class="collision" type="box" '
            f'pos="{p.foot_x_offset:.6g} 0 {sole_z + 0.01:.6g}" '
            f'size="{p.foot_length / 2:.6g} {p.foot_width / 2:.6g} 0.01" rgba="{PRINT}"/>'
        )
        lines.append(f'{ind}  <site name="{side}_foot_site" pos="{p.foot_x_offset:.6g} 0 {sole_z:.6g}"/>')
    else:
        s = 1 if b.name.startswith("l_") else -1
        lines += [f"{ind}  {g}" for g in _leg_proxies(b.name, s)]
    for ch in b.children:
        lines += _body(ch, depth + 1, p)
    lines.append(f"{ind}</body>")
    return lines


PRINT = "0.85 0.85 0.8 1"


def generate(robot_repo: Path) -> str:
    sys.path.insert(0, str(robot_repo))
    from robot.model import Params, build_robot  # type: ignore

    p = Params()
    root = build_robot(p)
    body = "\n".join(_body(root, 2, p))
    total = sum(b.mass for b in root.walk())
    return f"""<!-- GENERATED by generate_mjcf.py from the QDD biped robot repo (robot/model.py). Do not edit by hand. -->
<!-- total mass {total:.4f} kg -->
<mujoco model="qdd_biped">
  <compiler angle="radian" autolimits="true"/>
  <default>
    <joint damping="0.05" frictionloss="0.05"/>
    <default class="visual"><geom contype="0" conaffinity="0" group="2" mass="0"/></default>
    <default class="collision"><geom group="3" mass="0" friction="0.8 0.02 0.01"/></default>
  </default>
  <worldbody>
{body}
  </worldbody>
  <sensor>
    <!-- Names follow mjlab's velocity template (robot/imu_ang_vel, robot/imu_lin_vel,
         robot/root_angmom). imu_lin_vel is privileged (critic only). -->
    <framequat name="imu_quat" objtype="site" objname="imu"/>
    <gyro name="imu_ang_vel" site="imu"/>
    <velocimeter name="imu_lin_vel" site="imu"/>
    <accelerometer name="imu_acc" site="imu"/>
    <subtreeangmom name="root_angmom" body="torso"/>
  </sensor>
</mujoco>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    a.out.write_text(generate(a.robot_repo.resolve()), encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
