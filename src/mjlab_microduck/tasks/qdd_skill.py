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

SKILL_SCHEMA_VERSION = 1
SKILLS_DIR = Path(__file__).resolve().parents[1] / "robot" / "qdd_biped" / "skills"
SKILL_SCHEMA_JSON = SKILLS_DIR / "skill_spec.schema.json"
DEFAULT_SKILL_JSON = SKILLS_DIR / "bow.json"
# Path (or bare name in skills/) of the skill trained by Mjlab-Skill-QddBiped; read
# when the task package is imported (registration builds the env cfg).
SKILL_SPEC_ENV = "QDD_SKILL_SPEC"

POLICY_DT_S = 0.02  # 50 Hz (same as walking)

# --- Template limits (fixed; not settable from a spec) -----------------------------
DURATION_RANGE_S = (1.0, 8.0)
SETTLE_RANGE_S = (0.5, 3.0)
SETTLE_DEFAULT_S = 1.0
MAX_KEYFRAMES = 16
MIN_KEYFRAME_GAP_S = 0.2
MAX_DESCRIPTION_CHARS = 500
INTERPOLATIONS = ("minjerk", "linear")
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
}
REQUIRED_TOP_KEYS = ("schema_version", "name", "duration_s", "keyframes")
KEYFRAME_KEYS = {"t", "joints_deg", "torso", "note"}
TORSO_KEYS = {"pitch_deg", "roll_deg", "height_m"}
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
            feet_angle=feet_angle,
            feet_dz=float(pl_g[2] - pr_g[2]),
            feet_spacing=float(pl_g[1] - pr_g[1]),
            support_margin=_inside_margin(hull, com[:2]),
        )


_FK: FloorKinematics | None = None


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


def sample_joint_reference(times: np.ndarray, kf_t: np.ndarray, kf_q: np.ndarray, kind: str) -> np.ndarray:
    """Joint reference [len(times), 10] from keyframes (held after the last one)."""
    out = np.empty((len(times), kf_q.shape[1]))
    for i, t in enumerate(times):
        if t <= kf_t[0]:
            out[i] = kf_q[0]
        elif t >= kf_t[-1]:
            out[i] = kf_q[-1]
        else:
            j = int(np.searchsorted(kf_t, t, side="right")) - 1
            s = (t - kf_t[j]) / (kf_t[j + 1] - kf_t[j])
            out[i] = kf_q[j] + _interp_weight(float(s), kind) * (kf_q[j + 1] - kf_q[j])
    return out


def build_reference_arrays(spec: dict, kf_t: np.ndarray, kf_q: np.ndarray, dt: float = POLICY_DT_S) -> dict:
    """Dense reference at the policy rate: index k = time k * dt since the skill
    started, k = 0 .. n (n = round((duration + settle) / dt)). Keys: t, q [n+1, 10]
    (absolute, rad), pitch, roll, height, quat [n+1, 4] (torso in the ground frame),
    com_xy, feet_angle, feet_dz, feet_spacing, support_margin, in_skill (t < duration)."""
    total = spec["duration_s"] + spec["settle_s"]
    n = int(round(total / dt))
    t = np.arange(n + 1) * dt
    q = sample_joint_reference(t, kf_t, kf_q, spec["interpolation"])
    fk = floor_kinematics()
    poses = [fk(x) for x in q]
    return {
        "t": t,
        "q": q,
        "pitch": np.array([p.pitch for p in poses]),
        "roll": np.array([p.roll for p in poses]),
        "height": np.array([p.height for p in poses]),
        "quat": np.array([p.quat for p in poses]),
        "com_xy": np.array([p.com_xy for p in poses]),
        "feet_angle": np.array([p.feet_angle for p in poses]),
        "feet_dz": np.array([p.feet_dz for p in poses]),
        "feet_spacing": np.array([p.feet_spacing for p in poses]),
        "support_margin": np.array([p.support_margin for p in poses]),
        "in_skill": t < spec["duration_s"] - 1e-9,
    }


def build_reference(spec: dict, dt: float = POLICY_DT_S) -> dict:
    """Dense reference for a VALIDATED spec (see build_reference_arrays)."""
    kf_t = np.array([kf["t"] for kf in spec["keyframes"]], dtype=float)
    kf_q = np.array([kf["_q_rad"] for kf in spec["keyframes"]], dtype=float)
    return build_reference_arrays(spec, kf_t, kf_q, dt)


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

    if spec["schema_version"] != SKILL_SCHEMA_VERSION or isinstance(spec["schema_version"], bool):
        errors.append(f"schema_version: must be {SKILL_SCHEMA_VERSION}, got {spec['schema_version']!r}")
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
    support = spec.get("support", "both_feet")
    if support not in SUPPORTS:
        errors.append(f"support: {support!r} must be one of {list(SUPPORTS)}")

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
            if t_arr[i] - t_arr[i - 1] < MIN_KEYFRAME_GAP_S - 1e-9:
                errors.append(
                    f"keyframes[{i}].t: {t_arr[i]} must be at least {MIN_KEYFRAME_GAP_S} s after the "
                    f"previous keyframe ({t_arr[i - 1]})"
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
        {"duration_s": duration, "settle_s": settle, "interpolation": interp}, t_arr, np.array(qs)
    )
    check_reference(ref, errors)
    if errors:
        raise SkillSpecError(errors)

    out = {
        "schema_version": SKILL_SCHEMA_VERSION,
        "name": name,
        "description": desc,
        "duration_s": duration,
        "settle_s": settle,
        "interpolation": interp,
        "support": support,
        "keyframes": [dict(kf) for kf in kfs],
        "reward_weights": weights,
    }
    for kf, q in zip(out["keyframes"], qs):
        kf["_q_rad"] = [float(x) for x in q]
    return out


def check_reference(ref: dict, errors: list[str]) -> None:
    """Template safety / feasibility checks on a dense reference (appends to errors)."""
    from mjlab_microduck.robot.qdd_biped_constants import knee_max

    names = _joint_names()
    t = ref["t"]

    def at(mask) -> str:
        return f"t = {t[int(np.argmax(mask))]:.2f} s"

    dq = np.abs(np.diff(ref["q"], axis=0)) / POLICY_DT_S
    if dq.max() > MAX_JOINT_SPEED_RAD_S:
        k, j = np.unravel_index(np.argmax(dq), dq.shape)
        errors.append(
            f"reference: {names[j]} moves at {dq.max():.2f} rad/s around t = {t[k]:.2f} s "
            f"(limit {MAX_JOINT_SPEED_RAD_S} rad/s; spread the keyframes further apart)"
        )
    for key in ("pitch", "roll"):
        rate = np.abs(np.diff(ref[key])) / POLICY_DT_S
        if rate.max() > MAX_TORSO_RATE_RAD_S:
            errors.append(
                f"reference: torso {key} rate {rate.max():.2f} rad/s at {at(rate == rate.max())} "
                f"(limit {MAX_TORSO_RATE_RAD_S} rad/s)"
            )
    pd = np.degrees(ref["pitch"])
    lo, hi = TORSO_PITCH_RANGE_DEG
    if pd.max() > hi + 1e-6 or pd.min() < lo - 1e-6:
        errors.append(
            f"reference: torso pitch reaches {pd.max() if pd.max() > hi else pd.min():.1f} deg "
            f"(allowed [{lo}, {hi}] deg, + = nose down; the controller trips at 35 deg)"
        )
    rd = np.degrees(np.abs(ref["roll"]))
    if rd.max() > TORSO_ROLL_MAX_DEG + 1e-6:
        errors.append(f"reference: torso roll reaches {rd.max():.1f} deg (limit {TORSO_ROLL_MAX_DEG} deg)")
    if ref["height"].min() < TORSO_HEIGHT_MIN_M:
        errors.append(
            f"reference: torso height drops to {ref['height'].min():.3f} m (minimum {TORSO_HEIGHT_MIN_M} m)"
        )
    for side in ("l", "r"):
        hp, kn = ref["q"][:, names.index(f"{side}_hip_pitch")], ref["q"][:, names.index(f"{side}_knee")]
        over = kn - (np.array([knee_max(h) for h in hp]) - KNEE_LIMIT_MARGIN_RAD)
        if over.max() > 1e-9:
            errors.append(
                f"reference: {side}_knee passes the pose-dependent knee limit between keyframes at "
                f"{at(over > 1e-9)} (add a keyframe that bends the hip first)"
            )
    fa = np.degrees(ref["feet_angle"])
    if fa.max() > FEET_COPLANAR_DEG:
        errors.append(
            f"reference: the soles are not parallel ({fa.max():.1f} deg at {at(fa > FEET_COPLANAR_DEG)}); "
            f"support 'both_feet' needs both soles flat on the floor (mirror the legs or compensate with "
            f"the other joints)"
        )
    dz = np.abs(ref["feet_dz"])
    if dz.max() > FEET_COPLANAR_M:
        errors.append(
            f"reference: the soles are at different heights ({100 * dz.max():.1f} cm at "
            f"{at(dz > FEET_COPLANAR_M)}); support 'both_feet' needs both soles on the floor"
        )
    sp = ref["feet_spacing"]
    if sp.min() < FEET_SPACING_RANGE_M[0] or sp.max() > FEET_SPACING_RANGE_M[1]:
        errors.append(
            f"reference: lateral sole spacing {sp.min():.3f}..{sp.max():.3f} m is outside "
            f"[{FEET_SPACING_RANGE_M[0]}, {FEET_SPACING_RANGE_M[1]}] m"
        )
    sm = ref["support_margin"]
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
