"""QDD biped skill spec: load, validate, build the reference (Mjlab-Skill-QddBiped).

A *skill* is a short whole-body motion (bow, squat, ...) that starts and ends in
the standing pose. It is described by a small JSON file (the only thing a
language model writes later -- never code), schema skills/skill_spec.schema.json:

    {
      "schema_version": 1,
      "name": "bow",
      "duration_s": 3.0,             # keyframes span [0, duration_s]
      "settle_s": 1.0,               # then hold the standing pose ("return to stand")
      "interpolation": "minjerk",    # or "linear"
      "support": "both_feet",        # both soles stay flat on the floor
      "keyframes": [
        {"t": 0.0},                                        # = standing pose
        {"t": 1.2, "joints_deg": {"hip_pitch": -50, ...}, "torso": {"pitch_deg": 25}},
        {"t": 3.0}
      ],
      "reward_weights": {"joint_tracking": 3.0}           # optional, bounded
    }

Joint targets are ABSOLUTE angles in degrees in the robot convention (robot/model.py,
policy_io.json joint_order); joints a keyframe does not list keep the standing value.
A side-less name ("hip_pitch") sets both legs mirror-symmetrically (yaw / roll
negate on the right leg, as symmetry.QDD_JOINT_SIGN). Torso pitch is positive
nose-DOWN (a bow), roll positive = left side up; both are computed from the joint
targets with the soles flat on the floor (forward kinematics), so ``torso`` is only
a cross-check of what the author meant.

Everything safety-related (terminations, joint / knee limits, tilt and speed
bounds, domain randomisation, observation / action format) is fixed by the task
template; a spec that tries to set it is rejected. ``validate_spec`` is the gate a
generated spec must pass before any training; it returns the normalised spec or
raises ``SkillSpecError`` listing every problem with its location.

Schema version 2 (backward compatible: version 1 specs are still accepted
unchanged) adds jumps -- skills with flight phases (``"hop"``):

    "schema_version": 2,
    "interpolation": "pchip",       # monotone cubic: velocity continues through the push
    "policy_rate_hz": 50,           # or 100 (fast pushes); the ONNX policy_io states dt
    "contact": [                    # contact schedule: both soles down / both in the air
      {"t": 0.0, "feet": "both"},
      {"t": 1.39, "feet": "none"},  # take-off
      {"t": 1.55, "feet": "both"}   # touchdown
    ],
    "flight": {"com_rise_m": 0.03, "max_takeoff_speed_mps": 0.85,
               "max_landing_load_mg": 4.0, "toe_first": true}

Per-phase gates: around a flight (``fast window`` = take-off - 0.15 s ..
touchdown + 0.25 s) joint speeds up to 20 rad/s and keyframes 0.02 s apart are
allowed; the flat-sole checks (torso height / pitch rate, coplanar soles, static
balance) apply only while the feet are scheduled on the floor and outside the
landing window; the torso pitch limit is 40 deg for a skill with flight. Take-off
speed (CoM rise <= 4 cm, <= 0.9 m/s: motor bus current and landing) and the
landing load per foot (<= 4.5 m g: ankle-motor radial load) are bounded by the
template (docs/jump-feasibility.md in the robot repo).

CLI:  python -m mjlab_microduck.tasks.qdd_skill <spec.json> [...]   (exit 1 on errors)
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SKILL_SCHEMA_VERSION = 2  # newest; 1 is still accepted (no flight, 50 Hz)
SUPPORTED_SCHEMA_VERSIONS = (1, 2)
SKILLS_DIR = Path(__file__).resolve().parents[1] / "robot" / "qdd_biped" / "skills"
SKILL_SCHEMA_JSON = SKILLS_DIR / "skill_spec.schema.json"
DEFAULT_SKILL_JSON = SKILLS_DIR / "bow.json"
# Path (or bare name in skills/) of the skill trained by Mjlab-Skill-QddBiped; read
# when the task package is imported (registration builds the env cfg).
SKILL_SPEC_ENV = "QDD_SKILL_SPEC"

POLICY_DT_S = 0.02  # 50 Hz (same as walking); default policy rate
POLICY_RATES_HZ = (50, 100)  # v2 "policy_rate_hz"
G = 9.81

# --- Template limits (fixed; not settable from a spec) -----------------------------
DURATION_RANGE_S = (1.0, 8.0)
SETTLE_RANGE_S = (0.5, 3.0)
SETTLE_DEFAULT_S = 1.0
MAX_KEYFRAMES = 16
MIN_KEYFRAME_GAP_S = 0.2
MAX_DESCRIPTION_CHARS = 500
INTERPOLATIONS = ("minjerk", "linear", "pchip")  # pchip: schema 2
SUPPORTS = ("both_feet",)
JOINT_LIMIT_MARGIN_DEG = 3.0  # keyframes stay this far inside the joint range
KNEE_LIMIT_MARGIN_RAD = 0.05  # same margin as the walking task's knee_limit cost
MAX_JOINT_SPEED_RAD_S = 2.5  # reference joint speed (motors: 44 rad/s; keep skills calm)
MAX_TORSO_RATE_RAD_S = 1.5  # reference torso pitch/roll rate (controller fall check: 3 rad/s)
TORSO_PITCH_RANGE_DEG = (-15.0, 30.0)  # controller fall_tilt 35 deg (+gyro), max_tilt 50 deg
TORSO_ROLL_MAX_DEG = 10.0
TORSO_HEIGHT_MIN_M = 0.24  # termination at 0.20 m
FEET_COPLANAR_DEG = 3.0  # both_feet: soles parallel within this
FEET_COPLANAR_M = 0.01  # ... and at the same height within this
FEET_SPACING_RANGE_M = (0.08, 0.25)  # lateral sole distance (hips 0.13 apart)
COM_SUPPORT_MARGIN_M = 0.015  # static balance: CoM this far inside the support polygon
TORSO_CHECK_TOL_DEG = 5.0  # spec torso pitch/roll vs. forward kinematics
TORSO_CHECK_TOL_M = 0.02  # spec torso height vs. forward kinematics
STAND_TOL_DEG = 0.5  # first / last keyframe vs. the standing pose
FOOT_HALF_LENGTH_M = 0.075  # 150 x 70 mm soles centred under the ankle (foot site)
FOOT_HALF_WIDTH_M = 0.035

# --- Flight (schema 2; robot repo docs/jump-feasibility.md) ---------------------------
CONTACT_STATES = ("both", "none")
MAX_FLIGHTS = 3
FLIGHT_TIME_RANGE_S = (0.06, 0.3)  # the controller faults after 0.3 s airborne
FLIGHT_TIME_TOL_S = 0.06  # scheduled flight vs. ballistic 2 v0 / g
FAST_PRE_S = 0.15  # fast window starts this long before take-off (push-off) ...
LANDING_WINDOW_S = 0.25  # ... and ends this long after touchdown (absorb)
FAST_JOINT_SPEED_RAD_S = 20.0  # push-off 12-19 rad/s (GIM6010-8 at 22 V: 4 Nm left at 20 rad/s)
FAST_TORSO_RATE_RAD_S = 6.0  # the push-off pitches the torso back 35 -> 23 deg in ~90 ms
FAST_MIN_KEYFRAME_GAP_S = 0.02
MIN_RECOVERY_S = 0.5  # last touchdown -> end of the motion
FLIGHT_TORSO_PITCH_RANGE_DEG = (-15.0, 40.0)  # crouch for a momentum-free push: ~35 deg
COM_RISE_RANGE_M = (0.005, 0.04)  # bus current (LM5069) and landing load
TAKEOFF_SPEED_RANGE_MPS = (0.3, 0.9)
LANDING_LOAD_RANGE_MG = (2.0, 4.5)  # per foot; GIM3510-8 radial limit 300 N = 4.7 m g
FLIGHT_KEYS = {"com_rise_m", "max_takeoff_speed_mps", "max_landing_load_mg", "toe_first"}
CONTACT_TOL_S = 0.04  # take-off / touchdown may be this early / late (2 steps at 50 Hz)
FLIGHT_DEFAULTS = {"max_takeoff_speed_mps": 0.85, "max_landing_load_mg": 4.0, "toe_first": True}

# Reward weights a spec may set: name -> (min, max, default). Penalties are given
# as magnitudes (the task applies them with a negative sign). Balance / contact /
# return-to-stand have a non-zero minimum: a spec cannot switch them off.
REWARD_WEIGHT_BOUNDS: dict[str, tuple[float, float, float]] = {
    "joint_tracking": (0.0, 6.0, 3.0),
    "torso_orientation": (0.0, 6.0, 2.0),
    "torso_height": (0.0, 3.0, 1.0),
    "balance": (0.5, 3.0, 1.0),
    "feet_contact": (0.5, 3.0, 1.0),
    "final_stand": (0.5, 5.0, 2.0),
    "action_rate": (0.02, 0.5, 0.1),
    "joint_torques": (5.0e-5, 1.0e-3, 2.0e-4),
    "foot_slip": (0.5, 3.0, 1.0),
    # flight skills only (unused without a flight phase); feet_contact then weights the
    # contact schedule (both soles down / both in the air as scheduled)
    "flight_height": (0.5, 6.0, 3.0),
    "flight_rotation": (0.1, 2.0, 0.5),
    "landing_load": (0.5, 5.0, 1.0),
    "toe_first": (0.0, 2.0, 0.5),
    "takeoff_speed": (1.0, 20.0, 10.0),
    "joint_power": (0.0, 5.0e-3, 1.0e-3),
}

TOP_KEYS = {
    "schema_version",
    "name",
    "description",
    "duration_s",
    "settle_s",
    "interpolation",
    "support",
    "keyframes",
    "reward_weights",
    "contact",
    "flight",
    "policy_rate_hz",
}
V2_ONLY_KEYS = ("contact", "flight", "policy_rate_hz")
REQUIRED_TOP_KEYS = ("schema_version", "name", "duration_s", "keyframes")
KEYFRAME_KEYS = {"t", "joints_deg", "torso", "note"}
TORSO_KEYS = {"pitch_deg", "roll_deg", "height_m"}
CONTACT_KEYS = {"t", "feet"}
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_FORBIDDEN_HINT = {
    "terminations": "termination / safety limits are fixed by the template",
    "safety": "termination / safety limits are fixed by the template",
    "observations": "the observation format is fixed (same as walking, policy_io.json)",
    "actions": "the action format is fixed (same as walking, policy_io.json)",
    "domain_randomization": "domain randomisation is fixed by the template",
}


class SkillSpecError(ValueError):
    """A skill spec failed validation. ``errors`` lists every problem."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("invalid skill spec:\n  - " + "\n  - ".join(self.errors))


def _joint_names() -> tuple[str, ...]:
    from mjlab_microduck.robot.qdd_biped_constants import QDD_JOINT_NAMES

    return QDD_JOINT_NAMES


def _model():
    from mjlab_microduck.robot.qdd_biped_constants import get_qdd_biped_spec

    return get_qdd_biped_spec().compile()


def stand_pose() -> np.ndarray:
    """Default (standing) joint pose [rad], policy_io joint order."""
    from mjlab_microduck.robot.qdd_biped_constants import DEFAULT_JOINT_POS

    out = []
    for n in _joint_names():
        vals = [v for pat, v in DEFAULT_JOINT_POS.items() if re.fullmatch(pat, n)]
        assert len(vals) == 1, n
        out.append(float(vals[0]))
    return np.array(out)


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# =============================================================================
# Forward kinematics with both soles flat on the floor
# =============================================================================
@dataclass
class FloorPose:
    """Robot pose with the soles on a flat floor (ground frame: midpoint of the two
    foot sites, z = left sole normal, x = torso heading on the floor)."""

    pitch: float  # torso pitch [rad], + = nose down
    roll: float  # torso roll [rad], + = left side up
    height: float  # torso origin above the floor [m]
    quat: np.ndarray  # torso orientation in the ground frame (w, x, y, z)
    torso_xy: np.ndarray  # torso origin in the ground frame [m]
    com_xy: np.ndarray  # whole-robot CoM in the ground frame [m]
    com_z: float  # whole-robot CoM height above the floor [m]
    feet_angle: float  # angle between the two soles [rad]
    feet_dz: float  # height difference of the soles [m]
    feet_spacing: float  # l - r sole offset along the ground y axis [m]
    support_margin: float  # signed distance of the CoM inside the support polygon [m]


def pitch_roll_from_gravity(g) -> tuple[float, float]:
    """Torso pitch (+ nose down) / roll (+ left up) from the gravity direction in the
    torso frame (projected_gravity, (0, 0, -1) upright). Same formula as the reward."""
    return math.atan2(g[0], -g[2]), math.atan2(-g[1], -g[2])


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    p = sorted(map(tuple, pts))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1])  # counter-clockwise


def _inside_margin(hull: np.ndarray, x: np.ndarray) -> float:
    """Signed distance of x inside a CCW convex polygon (negative = outside)."""
    d = np.inf
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        e = b - a
        n = np.array([-e[1], e[0]]) / np.linalg.norm(e)  # inward normal (CCW)
        d = min(d, float(np.dot(x - a, n)))
    return d


class FloorKinematics:
    def __init__(self):
        import mujoco

        self._mj = mujoco
        self.m = _model()
        self.d = mujoco.MjData(self.m)
        self.qadr = [self.m.joint(n).qposadr[0] for n in _joint_names()]
        self.sites = [self.m.site("l_foot_site").id, self.m.site("r_foot_site").id]
        self.torso = self.m.body("torso").id

    def __call__(self, q: np.ndarray) -> FloorPose:
        mj, m, d = self._mj, self.m, self.d
        d.qpos[:] = 0.0
        d.qpos[3] = 1.0  # root at the origin, identity
        d.qpos[self.qadr] = q
        d.qvel[:] = 0.0
        mj.mj_forward(m, d)
        pl, pr = d.site_xpos[self.sites[0]].copy(), d.site_xpos[self.sites[1]].copy()
        Rl = d.site_xmat[self.sites[0]].reshape(3, 3).copy()
        Rr = d.site_xmat[self.sites[1]].reshape(3, 3).copy()
        origin = 0.5 * (pl + pr)
        # ground frame: z = left sole normal, x = torso heading projected on the floor
        z = Rl[:, 2]
        x = d.xmat[self.torso].reshape(3, 3)[:, 0]
        x = x - np.dot(x, z) * z
        x = x / np.linalg.norm(x)
        Rg = np.stack((x, np.cross(z, x), z), axis=1)

        def to_g(p):
            return Rg.T @ (p - origin)

        R_torso = Rg.T @ d.xmat[self.torso].reshape(3, 3)
        torso_pos = to_g(d.xpos[self.torso])
        com = to_g(d.subtree_com[1])  # body 1 = torso = whole robot subtree
        g_torso = R_torso.T @ np.array([0.0, 0.0, -1.0])
        pitch, roll = pitch_roll_from_gravity(g_torso)
        quat = np.zeros(4)
        mj.mju_mat2Quat(quat, R_torso.reshape(-1))
        # sole tilt relative to each other (normals); toe-in / toe-out (yaw) is fine
        feet_angle = math.acos(max(-1.0, min(1.0, float(np.dot(Rl[:, 2], Rr[:, 2])))))
        pl_g, pr_g = to_g(pl), to_g(pr)
        corners = []
        for p, R in ((pl, Rl), (pr, Rr)):
            for sx in (-1, 1):
                for sy in (-1, 1):
                    c = p + R @ np.array([sx * FOOT_HALF_LENGTH_M, sy * FOOT_HALF_WIDTH_M, 0.0])
                    corners.append(to_g(c)[:2])
        hull = _convex_hull(np.array(corners))
        return FloorPose(
            pitch=pitch,
            roll=roll,
            height=float(torso_pos[2]),
            quat=quat,
            torso_xy=torso_pos[:2],
            com_xy=com[:2],
            com_z=float(com[2]),
            feet_angle=feet_angle,
            feet_dz=float(pl_g[2] - pr_g[2]),
            feet_spacing=float(pl_g[1] - pr_g[1]),
            support_margin=_inside_margin(hull, com[:2]),
        )


_FK: FloorKinematics | None = None


def lowest_sole_offset(q: np.ndarray, quat: np.ndarray) -> float:
    """Lowest sole corner below the torso origin [m] (negative) with the torso at
    orientation quat (w, x, y, z) and joints q (for spawning airborne states)."""
    fk = floor_kinematics()
    mj, m, d = fk._mj, fk.m, fk.d
    d.qpos[:] = 0.0
    d.qpos[3:7] = quat
    d.qpos[fk.qadr] = q
    mj.mj_forward(m, d)
    z = np.inf
    for sid in fk.sites:
        p = d.site_xpos[sid]
        R = d.site_xmat[sid].reshape(3, 3)
        for sx in (-1, 1):
            for sy in (-1, 1):
                z = min(z, float((p + R @ np.array([sx * FOOT_HALF_LENGTH_M, sy * FOOT_HALF_WIDTH_M, 0.0]))[2]))
    return z


def floor_kinematics() -> FloorKinematics:
    global _FK
    if _FK is None:
        _FK = FloorKinematics()
    return _FK


# =============================================================================
# Reference
# =============================================================================
def _interp_weight(s: float, kind: str) -> float:
    if kind == "linear":
        return s
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))  # min-jerk: zero vel/acc at keyframes


def _pchip_slopes(kf_t: np.ndarray, kf_q: np.ndarray) -> np.ndarray:
    """Monotone cubic (Fritsch-Carlson) slopes at the keyframes [K, J]: no overshoot,
    zero slope at holds / extrema and at the first and last keyframe (standing, at
    rest), continuous velocity through a monotone run (a push-off)."""
    h = np.diff(kf_t)
    d = np.diff(kf_q, axis=0) / h[:, None]
    m = np.zeros_like(kf_q)
    for k in range(1, len(kf_t) - 1):
        w1, w2 = 2.0 * h[k] + h[k - 1], h[k] + 2.0 * h[k - 1]
        a, b = d[k - 1], d[k]
        same = a * b > 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            hm = (w1 + w2) / (w1 / a + w2 / b)
        m[k] = np.where(same, hm, 0.0)
    return m


def sample_joint_reference(times: np.ndarray, kf_t: np.ndarray, kf_q: np.ndarray, kind: str) -> np.ndarray:
    """Joint reference [len(times), 10] from keyframes (held after the last one)."""
    out = np.empty((len(times), kf_q.shape[1]))
    slopes = _pchip_slopes(kf_t, kf_q) if kind == "pchip" else None
    for i, t in enumerate(times):
        if t <= kf_t[0]:
            out[i] = kf_q[0]
        elif t >= kf_t[-1]:
            out[i] = kf_q[-1]
        else:
            j = int(np.searchsorted(kf_t, t, side="right")) - 1
            h = kf_t[j + 1] - kf_t[j]
            s = (t - kf_t[j]) / h
            if slopes is None:
                out[i] = kf_q[j] + _interp_weight(float(s), kind) * (kf_q[j + 1] - kf_q[j])
            else:  # cubic Hermite
                s2, s3 = s * s, s * s * s
                out[i] = (
                    (2 * s3 - 3 * s2 + 1) * kf_q[j]
                    + (s3 - 2 * s2 + s) * h * slopes[j]
                    + (-2 * s3 + 3 * s2) * kf_q[j + 1]
                    + (s3 - s2) * h * slopes[j + 1]
                )
    return out


def flight_intervals(contact) -> list[tuple[float, float]]:
    """[(take-off, touchdown)] from a (validated) contact schedule."""
    out = []
    for i, c in enumerate(contact or []):
        if c["feet"] == "none" and i + 1 < len(contact):
            out.append((float(c["t"]), float(contact[i + 1]["t"])))
    return out


def takeoff_speed(com_rise_m: float) -> float:
    """CoM take-off speed for a CoM rise h: v0 = sqrt(2 g h)."""
    return math.sqrt(2.0 * G * com_rise_m)


def build_reference_arrays(spec: dict, kf_t: np.ndarray, kf_q: np.ndarray, dt: float = POLICY_DT_S) -> dict:
    """Dense reference at the policy rate: index k = time k * dt since the skill
    started, k = 0 .. n (n = round((duration + settle) / dt)). Keys: t, q [n+1, 10]
    (absolute, rad), pitch, roll, height, quat [n+1, 4] (torso in the ground frame),
    com_xy, com_z, feet_angle, feet_dz, feet_spacing, support_margin, in_skill
    (t < duration). Contact schedule (schema 2): in_air (scheduled flight),
    air_window (flight + landing window: the flat-sole quantities are not meaningful
    there; pitch / roll / height are held at their take-off values), fast_window
    (push-off .. landing), height_window (push-off + flight: CoM height reference
    com_z_ref = flat-sole CoM, ballistic in flight), near_switch (within
    CONTACT_TOL_S of a scheduled take-off / touchdown), rsi_ok (reference-state
    initialisation allowed: outside the fast window), flights [(t0, t1)]."""
    total = spec["duration_s"] + spec["settle_s"]
    n = int(round(total / dt))
    t = np.arange(n + 1) * dt
    q = sample_joint_reference(t, kf_t, kf_q, spec["interpolation"])
    fk = floor_kinematics()
    poses = [fk(x) for x in q]
    ref = {
        "t": t,
        "q": q,
        "pitch": np.array([p.pitch for p in poses]),
        "roll": np.array([p.roll for p in poses]),
        "height": np.array([p.height for p in poses]),
        "quat": np.array([p.quat for p in poses]),
        "com_xy": np.array([p.com_xy for p in poses]),
        "com_z": np.array([p.com_z for p in poses]),
        "feet_angle": np.array([p.feet_angle for p in poses]),
        "feet_dz": np.array([p.feet_dz for p in poses]),
        "feet_spacing": np.array([p.feet_spacing for p in poses]),
        "support_margin": np.array([p.support_margin for p in poses]),
        "in_skill": t < spec["duration_s"] - 1e-9,
    }
    eps = 1e-9
    flights = flight_intervals(spec.get("contact"))
    in_air = np.zeros(n + 1, dtype=bool)
    air = np.zeros(n + 1, dtype=bool)
    fast = np.zeros(n + 1, dtype=bool)
    hwin = np.zeros(n + 1, dtype=bool)
    near = np.zeros(n + 1, dtype=bool)
    com_ref = ref["com_z"].copy()
    rise = (spec.get("flight") or {}).get("com_rise_m", 0.0)
    # reference-state initialisation inside a jump (push-off + flight): root height /
    # orientation / vertical speed. On the ground = flat-sole kinematics.
    rsi_h = ref["height"].copy()
    rsi_quat = ref["quat"].copy()
    rsi_vz = np.gradient(ref["height"], dt)
    rsi_jump = np.zeros(n + 1, dtype=bool)
    for t0, t1 in flights:
        k0 = int(np.clip(math.ceil(t0 / dt - 1e-6) - 1, 0, n))  # last step on the ground
        m_air = (t >= t0 - eps) & (t < t1 - eps)
        m_land = (t >= t0 - eps) & (t < t1 + LANDING_WINDOW_S - eps)
        in_air |= m_air
        air |= m_land
        fast |= (t >= t0 - FAST_PRE_S - eps) & (t < t1 + LANDING_WINDOW_S - eps)
        hwin |= (t >= t0 - FAST_PRE_S - eps) & (t < t1 - eps)
        near |= (np.abs(t - t0) <= CONTACT_TOL_S + eps) | (np.abs(t - t1) <= CONTACT_TOL_S + eps)
        v0 = takeoff_speed(rise)
        tau = t[m_air] - t[k0]
        com_ref[m_air] = ref["com_z"][k0] + v0 * tau - 0.5 * G * tau * tau
        for key in ("pitch", "roll", "height"):
            ref[key][m_land] = ref[key][k0]
        push = (t >= t0 - FAST_PRE_S - eps) & (t < t0 - eps)
        rsi_jump |= push | m_air
        for k in np.flatnonzero(m_air):
            # torso as at take-off, raised by the ballistic CoM rise; soles >= 2 mm above the floor
            rsi_quat[k] = ref["quat"][k0]
            h = ref["height"][k0] + (com_ref[k] - ref["com_z"][k0])
            low = lowest_sole_offset(q[k], rsi_quat[k])
            rsi_h[k] = max(h, 0.002 - low)
            rsi_vz[k] = takeoff_speed(rise) - G * (t[k] - t[k0])
    ref.update(
        in_air=in_air,
        air_window=air,
        fast_window=fast,
        height_window=hwin,
        near_switch=near,
        com_z_ref=com_ref,
        rsi_ok=~fast,
        rsi_jump=rsi_jump,
        rsi_height=rsi_h,
        rsi_quat=rsi_quat,
        rsi_vz=rsi_vz,
        com_vz_ref=np.gradient(com_ref, dt),
        flights=flights,
    )
    return ref


def build_reference(spec: dict, dt: float | None = None) -> dict:
    """Dense reference for a VALIDATED spec (see build_reference_arrays); dt defaults
    to the spec's policy period."""
    kf_t = np.array([kf["t"] for kf in spec["keyframes"]], dtype=float)
    kf_q = np.array([kf["_q_rad"] for kf in spec["keyframes"]], dtype=float)
    return build_reference_arrays(spec, kf_t, kf_q, spec.get("policy_dt_s", POLICY_DT_S) if dt is None else dt)


# =============================================================================
# Validation
# =============================================================================
def _joint_limits_rad() -> np.ndarray:
    m = _model()
    return np.array([m.joint(n).range for n in _joint_names()], dtype=float)


def _resolve_joints(jd, where: str, errors: list[str]) -> np.ndarray:
    """joints_deg -> absolute joint vector [rad] (unlisted joints = standing)."""
    from mjlab_microduck.tasks.symmetry import QDD_JOINT_SIGN

    names = _joint_names()
    q = stand_pose()
    if not isinstance(jd, dict):
        errors.append(f"{where}: must be an object {{joint name: degrees}}")
        return q
    bilateral = sorted({n[2:] for n in names})
    for key, val in jd.items():
        if key not in names and key not in bilateral:
            errors.append(
                f"{where}.{key}: unknown joint (use {bilateral} for both legs, or one of {list(names)})"
            )
            continue
        if not _is_number(val):
            errors.append(f"{where}.{key}: must be a finite number of degrees, got {val!r}")
            continue
        v = math.radians(float(val))
        if key in names:
            q[names.index(key)] = v
        else:
            for side in ("l_", "r_"):
                n = side + key
                if n in jd:
                    errors.append(f"{where}: both '{key}' and '{n}' are given (ambiguous)")
                i = names.index(n)
                q[i] = v * (QDD_JOINT_SIGN[i] if side == "r_" else 1.0)
    return q


def _validate_contact(spec: dict, duration, errors: list[str]) -> list[dict]:
    """Contact schedule (schema 2). Returns the normalised list (default: both feet
    down the whole time)."""
    default = [{"t": 0.0, "feet": "both"}]
    if "contact" not in spec:
        return default
    c = spec["contact"]
    if not isinstance(c, list) or not 1 <= len(c) <= 2 * MAX_FLIGHTS + 1:
        errors.append(f"contact: must be a list of 1..{2 * MAX_FLIGHTS + 1} phases {{t, feet}}")
        return default
    out = []
    for i, ph in enumerate(c):
        where = f"contact[{i}]"
        if not isinstance(ph, dict):
            errors.append(f'{where}: must be an object {{"t": seconds, "feet": "both" | "none"}}')
            return default
        for k in ph:
            if k not in CONTACT_KEYS:
                errors.append(f"{where}.{k}: not allowed (allowed: {sorted(CONTACT_KEYS)})")
        t, feet = ph.get("t"), ph.get("feet")
        if not _is_number(t):
            errors.append(f"{where}.t: must be a number of seconds, got {t!r}")
            return default
        if feet not in CONTACT_STATES:
            errors.append(
                f"{where}.feet: {feet!r} must be one of {list(CONTACT_STATES)} (single-foot phases are not "
                f"supported)"
            )
            return default
        out.append({"t": float(t), "feet": feet})
    n_err = len(errors)
    if out[0]["t"] != 0.0 or out[0]["feet"] != "both":
        errors.append("contact[0]: the schedule must start at t = 0 with both feet down")
    if out[-1]["feet"] != "both":
        errors.append(f"contact[{len(out) - 1}]: the schedule must end with both feet down (touchdown)")
    for i in range(1, len(out)):
        if out[i]["t"] <= out[i - 1]["t"]:
            errors.append(f"contact[{i}].t: times must increase ({out[i]['t']} after {out[i - 1]['t']})")
        if out[i]["feet"] == out[i - 1]["feet"]:
            errors.append(f"contact[{i}]: phases must alternate both / none (repeated '{out[i]['feet']}')")
    if len(errors) > n_err:
        return out
    lo, hi = FLIGHT_TIME_RANGE_S
    for t0, t1 in flight_intervals(out):
        if not lo - 1e-9 <= t1 - t0 <= hi + 1e-9:
            errors.append(
                f"contact: flight {t0:.3f}..{t1:.3f} s lasts {t1 - t0:.3f} s (allowed {lo}..{hi} s; the "
                f"controller faults after {hi} s airborne)"
            )
        if t0 < 0.5:
            errors.append(f"contact: take-off at {t0} s is too early (crouch first; >= 0.5 s)")
        if duration is not None and t1 > duration - MIN_RECOVERY_S + 1e-9:
            errors.append(
                f"contact: touchdown at {t1} s leaves less than {MIN_RECOVERY_S} s to return to the standing "
                f"pose before duration_s = {duration}"
            )
    return out


def _validate_flight(spec: dict, contact: list[dict], errors: list[str]) -> dict | None:
    flights = flight_intervals(contact)
    fl = spec.get("flight")
    if not flights:
        if fl is not None:
            errors.append("flight: given, but the contact schedule has no 'none' (flight) phase")
        return None
    if not isinstance(fl, dict):
        errors.append(
            'flight: required for a contact schedule with a flight phase ({"com_rise_m": ..., '
            '"max_takeoff_speed_mps": ..., "max_landing_load_mg": ...})'
        )
        return None
    for k in fl:
        if k not in FLIGHT_KEYS:
            errors.append(f"flight.{k}: not allowed (allowed: {sorted(FLIGHT_KEYS)})")
    out = dict(FLIGHT_DEFAULTS)
    for key, rng in (
        ("com_rise_m", COM_RISE_RANGE_M),
        ("max_takeoff_speed_mps", TAKEOFF_SPEED_RANGE_MPS),
        ("max_landing_load_mg", LANDING_LOAD_RANGE_MG),
    ):
        if key not in fl:
            if key == "com_rise_m":
                errors.append("flight.com_rise_m: required (target CoM rise above the take-off height, m)")
            continue
        v = fl[key]
        if not _is_number(v) or not rng[0] <= float(v) <= rng[1]:
            errors.append(f"flight.{key}: {v!r} must be a number in [{rng[0]}, {rng[1]}]")
            continue
        out[key] = float(v)
    if "toe_first" in fl:
        if not isinstance(fl["toe_first"], bool):
            errors.append("flight.toe_first: must be true or false")
        else:
            out["toe_first"] = fl["toe_first"]
    if "com_rise_m" not in out:
        return None
    v0 = takeoff_speed(out["com_rise_m"])
    if v0 > out["max_takeoff_speed_mps"] + 1e-9:
        errors.append(
            f"flight: a CoM rise of {out['com_rise_m']} m needs a take-off speed of {v0:.2f} m/s, above "
            f"max_takeoff_speed_mps = {out['max_takeoff_speed_mps']}"
        )
    t_ball = 2.0 * v0 / G
    for t0, t1 in flights:
        if abs((t1 - t0) - t_ball) > FLIGHT_TIME_TOL_S + 1e-9:
            errors.append(
                f"contact: flight {t0:.3f}..{t1:.3f} s ({t1 - t0:.3f} s) does not match the ballistic flight "
                f"time {t_ball:.3f} s of a {out['com_rise_m']} m CoM rise (tolerance {FLIGHT_TIME_TOL_S} s)"
            )
    return out


def validate_spec(spec) -> dict:
    """Validate a skill spec (dict parsed from JSON). Returns the normalised spec
    (defaults filled in, ``_q_rad`` joint vectors added to the keyframes) or raises
    SkillSpecError with every problem found."""
    errors: list[str] = []
    if not isinstance(spec, dict):
        raise SkillSpecError(["the spec must be a JSON object"])
    for k in spec:
        if k not in TOP_KEYS:
            hint = _FORBIDDEN_HINT.get(k, f"allowed keys: {sorted(TOP_KEYS)}")
            errors.append(f"'{k}' is not allowed ({hint})")
    missing = [k for k in REQUIRED_TOP_KEYS if k not in spec]
    errors += [f"'{k}' is required" for k in missing]
    if missing:
        raise SkillSpecError(errors)

    version = spec["schema_version"]
    if isinstance(version, bool) or version not in SUPPORTED_SCHEMA_VERSIONS:
        errors.append(f"schema_version: must be one of {list(SUPPORTED_SCHEMA_VERSIONS)}, got {version!r}")
        version = SKILL_SCHEMA_VERSION
    if version < 2:
        for k in V2_ONLY_KEYS:
            if k in spec:
                errors.append(f"'{k}' needs schema_version 2")
    name = spec["name"]
    if not isinstance(name, str) or not NAME_RE.match(name):
        errors.append(f"name: {name!r} must match {NAME_RE.pattern} (lowercase ascii letters, digits, _)")
    desc = spec.get("description", "")
    if not isinstance(desc, str) or len(desc) > MAX_DESCRIPTION_CHARS:
        errors.append(f"description: must be a string of at most {MAX_DESCRIPTION_CHARS} characters")

    def num_in(key, rng, default=None):
        v = spec.get(key, default)
        if not _is_number(v):
            errors.append(f"{key}: must be a number, got {v!r}")
            return None
        if not rng[0] <= float(v) <= rng[1]:
            errors.append(f"{key}: {v} is outside the allowed range [{rng[0]}, {rng[1]}]")
            return None
        return float(v)

    duration = num_in("duration_s", DURATION_RANGE_S)
    settle = num_in("settle_s", SETTLE_RANGE_S, SETTLE_DEFAULT_S)
    interp = spec.get("interpolation", "minjerk")
    if interp not in INTERPOLATIONS:
        errors.append(f"interpolation: {interp!r} must be one of {list(INTERPOLATIONS)}")
        interp = "minjerk"
    elif interp == "pchip" and version < 2:
        errors.append("interpolation: 'pchip' needs schema_version 2")
    support = spec.get("support", "both_feet")
    if support not in SUPPORTS:
        errors.append(f"support: {support!r} must be one of {list(SUPPORTS)}")
    rate = spec.get("policy_rate_hz", 50)
    if isinstance(rate, bool) or rate not in POLICY_RATES_HZ:
        errors.append(f"policy_rate_hz: {rate!r} must be one of {list(POLICY_RATES_HZ)}")
        rate = 50
    dt = 1.0 / float(rate)
    contact = _validate_contact(spec, duration, errors)
    flight = _validate_flight(spec, contact, errors)
    flights = flight_intervals(contact)

    def in_fast(t):
        return any(t0 - FAST_PRE_S - 1e-9 <= t <= t1 + LANDING_WINDOW_S + 1e-9 for t0, t1 in flights)

    def in_air(t):
        return any(t0 - 1e-9 <= t <= t1 + LANDING_WINDOW_S + 1e-9 for t0, t1 in flights)

    # --- reward weights
    weights = {k: v[2] for k, v in REWARD_WEIGHT_BOUNDS.items()}
    rw = spec.get("reward_weights", {})
    if not isinstance(rw, dict):
        errors.append("reward_weights: must be an object {term: weight}")
        rw = {}
    for k, v in rw.items():
        if k not in REWARD_WEIGHT_BOUNDS:
            errors.append(f"reward_weights.{k}: unknown term (allowed: {sorted(REWARD_WEIGHT_BOUNDS)})")
            continue
        lo, hi, _ = REWARD_WEIGHT_BOUNDS[k]
        if not _is_number(v) or not lo <= float(v) <= hi:
            errors.append(f"reward_weights.{k}: {v!r} must be a number in [{lo}, {hi}]")
            continue
        weights[k] = float(v)

    # --- keyframes
    kfs = spec["keyframes"]
    if not isinstance(kfs, list) or not 2 <= len(kfs) <= MAX_KEYFRAMES:
        errors.append(f"keyframes: must be a list of 2..{MAX_KEYFRAMES} keyframes")
        raise SkillSpecError(errors)
    stand = stand_pose()
    times, qs, torsos = [], [], []
    for i, kf in enumerate(kfs):
        where = f"keyframes[{i}]"
        if not isinstance(kf, dict):
            errors.append(f"{where}: must be an object")
            times.append(np.nan)
            qs.append(stand.copy())
            torsos.append({})
            continue
        for k in kf:
            if k not in KEYFRAME_KEYS:
                errors.append(f"{where}.{k}: not allowed (allowed: {sorted(KEYFRAME_KEYS)})")
        t = kf.get("t")
        if not _is_number(t):
            errors.append(f"{where}.t: must be a number of seconds, got {t!r}")
            t = np.nan
        times.append(float(t))
        qs.append(_resolve_joints(kf.get("joints_deg", {}), f"{where}.joints_deg", errors))
        tor = kf.get("torso", {})
        if not isinstance(tor, dict):
            errors.append(f"{where}.torso: must be an object")
            tor = {}
        for k, v in tor.items():
            if k not in TORSO_KEYS:
                errors.append(f"{where}.torso.{k}: not allowed (allowed: {sorted(TORSO_KEYS)})")
            elif not _is_number(v):
                errors.append(f"{where}.torso.{k}: must be a number, got {v!r}")
        torsos.append({k: float(v) for k, v in tor.items() if k in TORSO_KEYS and _is_number(v)})
        if "note" in kf and (not isinstance(kf["note"], str) or len(kf["note"]) > 200):
            errors.append(f"{where}.note: must be a string of at most 200 characters")

    t_arr = np.array(times)
    if np.all(np.isfinite(t_arr)):
        if t_arr[0] != 0.0:
            errors.append(f"keyframes[0].t: the first keyframe must be at t = 0 (got {t_arr[0]})")
        if duration is not None and abs(t_arr[-1] - duration) > 1e-9:
            errors.append(
                f"keyframes[{len(t_arr) - 1}].t: the last keyframe must be at t = duration_s = {duration} "
                f"(got {t_arr[-1]})"
            )
        for i in range(1, len(t_arr)):
            fast = in_fast(t_arr[i]) and in_fast(t_arr[i - 1])
            gap = FAST_MIN_KEYFRAME_GAP_S if fast else MIN_KEYFRAME_GAP_S
            if t_arr[i] - t_arr[i - 1] < gap - 1e-9:
                errors.append(
                    f"keyframes[{i}].t: {t_arr[i]} must be at least {gap} s after the "
                    f"previous keyframe ({t_arr[i - 1]})"
                    + (f" ({FAST_MIN_KEYFRAME_GAP_S} s only around a flight phase)" if flights and not fast else "")
                )
    for i in (0, len(kfs) - 1):
        dev = float(np.degrees(np.abs(qs[i] - stand)).max())
        if dev > STAND_TOL_DEG or torsos[i]:
            errors.append(
                f"keyframes[{i}]: a skill must {'start' if i == 0 else 'end'} in the standing pose "
                f"(leave joints_deg and torso empty; max joint deviation {dev:.1f} deg)"
            )

    # --- joint limits on the keyframes
    from mjlab_microduck.robot.qdd_biped_constants import knee_max

    names = _joint_names()
    lim = _joint_limits_rad()
    margin = math.radians(JOINT_LIMIT_MARGIN_DEG)
    for i, q in enumerate(qs):
        for j, n in enumerate(names):
            lo, hi = lim[j, 0] + margin, lim[j, 1] - margin
            if n.endswith("knee"):
                lo = lim[j, 0]  # the knee may straighten fully
            if not lo - 1e-9 <= q[j] <= hi + 1e-9:
                errors.append(
                    f"keyframes[{i}].joints_deg.{n}: {math.degrees(q[j]):.1f} deg is outside the allowed "
                    f"range [{math.degrees(lo):.1f}, {math.degrees(hi):.1f}] deg"
                )
        for side in ("l", "r"):
            hp, kn = q[names.index(f"{side}_hip_pitch")], q[names.index(f"{side}_knee")]
            kmax = knee_max(hp) - KNEE_LIMIT_MARGIN_RAD
            if kn > kmax + 1e-9:
                errors.append(
                    f"keyframes[{i}].joints_deg.{side}_knee: {math.degrees(kn):.1f} deg exceeds the "
                    f"pose-dependent knee limit {math.degrees(kmax):.1f} deg at hip_pitch "
                    f"{math.degrees(hp):.1f} deg (robot/model.py knee_max minus 3 deg)"
                )
    if errors or duration is None or settle is None:
        raise SkillSpecError(errors)

    # --- kinematic / balance checks on the keyframes and the dense reference
    fk = floor_kinematics()
    for i, (q, tor) in enumerate(zip(qs, torsos)):
        if tor and in_air(t_arr[i]):
            errors.append(
                f"keyframes[{i}].torso: no torso cross-check in a flight / landing window (the soles are not "
                f"flat on the floor there); remove it"
            )
            continue
        fp = fk(q)
        if "pitch_deg" in tor and abs(math.degrees(fp.pitch) - tor["pitch_deg"]) > TORSO_CHECK_TOL_DEG:
            errors.append(
                f"keyframes[{i}].torso.pitch_deg: {tor['pitch_deg']} deg, but the joint targets give "
                f"{math.degrees(fp.pitch):.1f} deg with the soles flat (torso pitch = -(hip_pitch + knee + "
                f"ankle_pitch)); adjust the joints"
            )
        if "roll_deg" in tor and abs(math.degrees(fp.roll) - tor["roll_deg"]) > TORSO_CHECK_TOL_DEG:
            errors.append(
                f"keyframes[{i}].torso.roll_deg: {tor['roll_deg']} deg, but the joint targets give "
                f"{math.degrees(fp.roll):.1f} deg with the soles flat"
            )
        if "height_m" in tor and abs(fp.height - tor["height_m"]) > TORSO_CHECK_TOL_M:
            errors.append(
                f"keyframes[{i}].torso.height_m: {tor['height_m']} m, but the joint targets give "
                f"{fp.height:.3f} m with the soles flat"
            )
    ref = build_reference_arrays(
        {"duration_s": duration, "settle_s": settle, "interpolation": interp, "contact": contact, "flight": flight},
        t_arr,
        np.array(qs),
        dt,
    )
    check_reference(ref, errors, flight=bool(flights))
    if errors:
        raise SkillSpecError(errors)

    out = {
        "schema_version": int(version),
        "name": name,
        "description": desc,
        "duration_s": duration,
        "settle_s": settle,
        "interpolation": interp,
        "support": support,
        "policy_rate_hz": int(rate),
        "policy_dt_s": dt,
        "contact": contact,
        "flight": flight,
        "keyframes": [dict(kf) for kf in kfs],
        "reward_weights": weights,
    }
    for kf, q in zip(out["keyframes"], qs):
        kf["_q_rad"] = [float(x) for x in q]
    return out


def check_reference(ref: dict, errors: list[str], flight: bool = False) -> None:
    """Template safety / feasibility checks on a dense reference (appends to errors).
    Around a flight (ref fast_window / air_window) the joint-speed limit is relaxed and
    the flat-sole checks are skipped (see the module docstring)."""
    from mjlab_microduck.robot.qdd_biped_constants import knee_max

    names = _joint_names()
    t = ref["t"]
    dt = float(t[1] - t[0])
    n1 = len(t)
    fast = ref.get("fast_window", np.zeros(n1, dtype=bool))
    ground = ~ref.get("air_window", np.zeros(n1, dtype=bool))

    def at(mask) -> str:
        return f"t = {t[int(np.argmax(mask))]:.2f} s"

    dq = np.abs(np.diff(ref["q"], axis=0)) / dt
    fast_pair = fast[:-1] | fast[1:]
    lim = np.where(fast_pair, FAST_JOINT_SPEED_RAD_S, MAX_JOINT_SPEED_RAD_S)[:, None]
    over = dq - lim
    if over.max() > 1e-9:
        k, j = np.unravel_index(np.argmax(over), over.shape)
        errors.append(
            f"reference: {names[j]} moves at {dq[k, j]:.2f} rad/s around t = {t[k]:.2f} s "
            f"(limit {lim[k, 0]} rad/s; spread the keyframes further apart)"
        )
    ground_pair = ground[:-1] & ground[1:]
    for key in ("pitch", "roll"):
        rate = np.abs(np.diff(ref[key])) / dt
        rlim = np.where(fast_pair, FAST_TORSO_RATE_RAD_S, MAX_TORSO_RATE_RAD_S)
        bad = ground_pair & (rate > rlim + 1e-9)
        if bad.any():
            k = int(np.argmax(np.where(bad, rate - rlim, -np.inf)))
            errors.append(
                f"reference: torso {key} rate {rate[k]:.2f} rad/s at t = {t[k]:.2f} s (limit {rlim[k]} rad/s)"
            )
    pd = np.degrees(ref["pitch"][ground])
    lo, hi = FLIGHT_TORSO_PITCH_RANGE_DEG if flight else TORSO_PITCH_RANGE_DEG
    if pd.size and (pd.max() > hi + 1e-6 or pd.min() < lo - 1e-6):
        errors.append(
            f"reference: torso pitch reaches {pd.max() if pd.max() > hi else pd.min():.1f} deg "
            f"(allowed [{lo}, {hi}] deg, + = nose down; the controller trips at 35 deg"
            f"{', 50 deg during a skill with flight' if flight else ''})"
        )
    rd = np.degrees(np.abs(ref["roll"][ground]))
    if rd.size and rd.max() > TORSO_ROLL_MAX_DEG + 1e-6:
        errors.append(f"reference: torso roll reaches {rd.max():.1f} deg (limit {TORSO_ROLL_MAX_DEG} deg)")
    if ref["height"][ground].min() < TORSO_HEIGHT_MIN_M:
        errors.append(
            f"reference: torso height drops to {ref['height'][ground].min():.3f} m "
            f"(minimum {TORSO_HEIGHT_MIN_M} m)"
        )
    for side in ("l", "r"):
        hp, kn = ref["q"][:, names.index(f"{side}_hip_pitch")], ref["q"][:, names.index(f"{side}_knee")]
        over = kn - (np.array([knee_max(h) for h in hp]) - KNEE_LIMIT_MARGIN_RAD)
        if over.max() > 1e-9:
            errors.append(
                f"reference: {side}_knee passes the pose-dependent knee limit between keyframes at "
                f"{at(over > 1e-9)} (add a keyframe that bends the hip first)"
            )
    fa = np.where(ground, np.degrees(ref["feet_angle"]), 0.0)
    if fa.max() > FEET_COPLANAR_DEG:
        errors.append(
            f"reference: the soles are not parallel ({fa.max():.1f} deg at {at(fa > FEET_COPLANAR_DEG)}); "
            f"support 'both_feet' needs both soles flat on the floor (mirror the legs or compensate with "
            f"the other joints)"
        )
    dz = np.where(ground, np.abs(ref["feet_dz"]), 0.0)
    if dz.max() > FEET_COPLANAR_M:
        errors.append(
            f"reference: the soles are at different heights ({100 * dz.max():.1f} cm at "
            f"{at(dz > FEET_COPLANAR_M)}); support 'both_feet' needs both soles on the floor"
        )
    sp = ref["feet_spacing"][ground]
    if sp.min() < FEET_SPACING_RANGE_M[0] or sp.max() > FEET_SPACING_RANGE_M[1]:
        errors.append(
            f"reference: lateral sole spacing {sp.min():.3f}..{sp.max():.3f} m is outside "
            f"[{FEET_SPACING_RANGE_M[0]}, {FEET_SPACING_RANGE_M[1]}] m"
        )
    sm = np.where(ground, ref["support_margin"], np.inf)
    if sm.min() < COM_SUPPORT_MARGIN_M:
        k = int(np.argmin(sm))
        errors.append(
            f"reference: statically unbalanced at t = {t[k]:.2f} s: the centre of mass is only "
            f"{100 * sm[k]:.1f} cm inside the edge of the feet (minimum {100 * COM_SUPPORT_MARGIN_M:.1f} cm; "
            f"CoM x {100 * ref['com_xy'][k, 0]:+.1f} cm from the ankles; bend the knees / ankles to move "
            f"the hips back)"
        )


# =============================================================================
# Loading
# =============================================================================
def resolve_spec_path(path=None) -> Path:
    """path None -> $QDD_SKILL_SPEC or skills/bow.json; a bare name ("bow") means
    skills/<name>.json."""
    import os

    if path is None:
        path = os.environ.get(SKILL_SPEC_ENV) or DEFAULT_SKILL_JSON
    p = Path(path)
    if not p.exists() and not p.suffix:
        p = SKILLS_DIR / f"{p}.json"
    return p


def load_spec_text(text: str, source: str = "<string>") -> dict:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        raise SkillSpecError([f"{source}: not valid JSON ({e})"]) from None
    spec = validate_spec(raw)
    spec["_source"] = source
    spec["_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    spec["_text"] = text
    return spec


def load_spec(path=None) -> dict:
    """Read + validate a skill JSON (see resolve_spec_path)."""
    p = resolve_spec_path(path)
    return load_spec_text(p.read_text(encoding="utf-8"), str(p))


def total_s(spec: dict) -> float:
    return spec["duration_s"] + spec["settle_s"]


def main(argv=None) -> int:
    import sys

    paths = sys.argv[1:] if argv is None else argv
    if not paths:
        print("usage: python -m mjlab_microduck.tasks.qdd_skill <spec.json | name> [...]")
        return 2
    bad = 0
    for p in paths:
        try:
            spec = load_spec(p)
        except (SkillSpecError, OSError) as e:
            bad += 1
            print(f"NG {p}\n{e}")
            continue
        ref = build_reference(spec)
        print(
            f"OK {p}: '{spec['name']}' {spec['duration_s']} s + settle {spec['settle_s']} s, "
            f"{len(spec['keyframes'])} keyframes, torso pitch max {np.degrees(ref['pitch'].max()):.1f} deg, "
            f"height min {ref['height'].min():.3f} m, CoM margin min {100 * ref['support_margin'].min():.1f} cm"
        )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
