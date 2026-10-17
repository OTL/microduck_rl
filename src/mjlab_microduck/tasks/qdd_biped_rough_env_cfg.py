"""QDD biped walking on rough terrain — Mjlab-Velocity-Rough-QddBiped.

The flat recipe (qdd_biped_velocity_env_cfg.py, gait4) on a terrain curriculum
of bumps, slopes, low steps and waves. The actor is UNCHANGED (blind, same
41-D obs, same gait clock, same policy_io.json), so the controller runs a
rough-terrain ONNX exactly like the flat one: only the walking ONNX in
controller/config/robot.toml changes.

What changes against the flat task:
  - terrain: QDD_ROUGH_TERRAINS_CFG (curriculum mode: one column per terrain
    type, 10 difficulty rows; a robot moves one row up after a whole episode
    without a fall and with good velocity tracking, one row down after a fall:
    mdp.qdd_terrain_levels_survival),
  - foot heights in the gait rewards are the clearance above the terrain under
    each foot (foot_height_scan), not world z,
  - the critic also gets a height scan around the torso (privileged, training
    only; the actor stays blind),
  - feet_flat is weaker: on a slope the stance sole must tilt with the ground,
  - solver: more iterations and contacts (box edges).
"""

import math
from copy import deepcopy

import mjlab.terrains as terrain_gen
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import CurriculumTermCfg, ObservationTermCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mjlab_microduck.robot.qdd_biped_constants import TORSO_BODY, get_qdd_biped_spec
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.qdd_biped_velocity_env_cfg import make_qdd_biped_velocity_env_cfg

# Terrain. The robot: 0.26 m legs, 5 cm swing apex (3.5 cm minimum clearance),
# 150 x 70 mm feet, no ankle roll. Hardest row = what it should handle indoors
# and outdoors: 5 cm steps (door sills, kerb-like), 15 deg slopes, 4 cm bumps.
# Patch 5 m: a robot moves up after 2.5 m (5 s at 0.5 m/s of a 20 s episode).
TERRAIN_PATCH_SIZE = 5.0
TERRAIN_ROWS = 10  # difficulty levels
MAX_INIT_TERRAIN_LEVEL = 2  # new robots start on the 3 easiest rows
STEP_HEIGHT_MAX = 0.05  # m (pyramid stairs, up and down)
STEP_WIDTH = 0.30  # m (2 feet long)
SLOPE_MAX = math.tan(math.radians(15.0))
BUMP_MAX = 0.04  # m (random heightfield noise, peak to peak)
GRID_MAX = 0.03  # m (random box grid: uneven paving / cobbles)
WAVE_AMPLITUDE_MAX = 0.04  # m

QDD_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(TERRAIN_PATCH_SIZE, TERRAIN_PATCH_SIZE),
    border_width=10.0,
    num_rows=TERRAIN_ROWS,
    num_cols=6,
    curriculum=True,
    sub_terrains={
        # proportion = share of robots spawned on that column (curriculum mode)
        "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.10),
        "bumps": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(0.0, BUMP_MAX),
            noise_step=0.005,
            downsampled_scale=0.1,  # 10 cm bumps (a sole is 15 x 7 cm)
            horizontal_scale=0.05,
            border_width=0.25,
        ),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.0, SLOPE_MAX),
            platform_width=1.5,
            border_width=0.25,
            vertical_scale=0.002,  # 2 mm height steps: a smooth slope, not a staircase
        ),
        "stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.0, STEP_HEIGHT_MAX),
            step_width=STEP_WIDTH,
            platform_width=1.5,
            border_width=0.5,
        ),
        "grid": terrain_gen.BoxRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.35,  # 5 m / 0.35 does not divide evenly (generator requirement)
            grid_height_range=(0.0, GRID_MAX),
            platform_width=1.5,
        ),
        "waves": terrain_gen.HfWaveTerrainCfg(
            proportion=0.10,
            amplitude_range=(0.0, WAVE_AMPLITUDE_MAX),
            num_waves=6,
            border_width=0.25,
            vertical_scale=0.002,
        ),
    },
    add_lights=False,
)

# Sole tilt cost on rough ground (flat: -2.0). A 15 deg slope tilts the stance
# sole by 15 deg: -2.0 would cost 0.13 per foot per step for walking correctly.
FEET_FLAT_WEIGHT_ROUGH = -0.5

# Privileged torso height scan for the critic (0.8 x 0.5 m around the torso).
CRITIC_SCAN_SIZE = (0.8, 0.5)
CRITIC_SCAN_RESOLUTION = 0.1

# Sole clearance rays start this far ABOVE the sole: a ray cast from the sole
# itself starts inside the ground whenever the sole sinks in a little (soft
# contact, slopes) and then misses the surface (seen on the slope patches).
SOLE_SCAN_LIFT = 0.05  # m
SOLE_SCAN_SITES = ("l_sole_scan_site", "r_sole_scan_site")

# Terrain curriculum (mdp.qdd_terrain_levels_survival): one row up after a whole
# episode without a fall and with a mean xy velocity error below this; one row
# down after a fall. (mjlab's distance rule demoted even the flat column: rough1.)
TERRAIN_UP_MAX_ERR_XY = 0.2  # m/s
# rough2: stairs / grid / flat climbed to level 5-6, slope / bumps / waves
# stayed at ~0 with a mean xy error of 0.42 m/s. Standing still at level 0 the
# robot already slides ~0.14 m/s on these height-field columns (and 0.03 on
# grid): the walking error there is larger at any level, so they get a looser
# bar. Falls still move them down.
TERRAIN_UP_MAX_ERR_XY_BY_TYPE = {"slope": 0.4, "bumps": 0.4, "waves": 0.4}

# Play (export / video): terrain levels for the fixed command schedule.
PLAY_TERRAIN_ROWS = 4


def make_qdd_biped_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Velocity tracking on rough terrain (blind actor, same obs as flat)."""
    cfg = make_qdd_biped_velocity_env_cfg(play=play)

    # --- Terrain -----------------------------------------------------------------
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = deepcopy(QDD_ROUGH_TERRAINS_CFG)
    cfg.scene.terrain.max_init_terrain_level = MAX_INIT_TERRAIN_LEVEL
    if play:
        gen = cfg.scene.terrain.terrain_generator
        gen.num_rows = PLAY_TERRAIN_ROWS
        gen.difficulty_range = (0.3, 1.0)

    # Spawn a little higher than on the plane: the bumps / waves under a foot
    # reach up to half their amplitude above the patch origin.
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.01, 0.03)

    # Box edges + falls onto terrain: more contacts and solver iterations than
    # the flat task (microduck rough: same fix for edge-contact NaNs).
    cfg.sim.nconmax = 120
    cfg.sim.mujoco.iterations = 20
    cfg.sim.mujoco.ls_iterations = 30

    # --- Sensors -------------------------------------------------------------------
    robot = cfg.scene.entities["robot"]
    robot.spec_fn = _spec_with_sole_scan_sites
    sole_scan = TerrainHeightSensorCfg(
        name="sole_clearance_scan",
        frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in SOLE_SCAN_SITES),
        pattern=RingPatternCfg.single_ring(radius=0.06, num_samples=4),
        ray_alignment="yaw",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=False,
    )
    terrain_scan = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name=TORSO_BODY, entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=CRITIC_SCAN_SIZE, resolution=CRITIC_SCAN_RESOLUTION),
        max_distance=2.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=False,
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (terrain_scan, sole_scan)

    # --- Gait rewards measure clearance above the terrain ----------------------------
    for name in ("gait_swing_height", "swing_clearance", "feet_flat"):
        cfg.rewards[name].params["height_sensor_name"] = sole_scan.name
        cfg.rewards[name].params["height_offset"] = SOLE_SCAN_LIFT
    cfg.rewards["feet_flat"].weight = FEET_FLAT_WEIGHT_ROUGH

    # --- Critic: privileged height scan -------------------------------------------
    cfg.observations["critic"].terms["height_scan"] = ObservationTermCfg(
        func=envs_mdp.height_scan,
        params={"sensor_name": terrain_scan.name},
        scale=1.0 / terrain_scan.max_distance,
    )

    # --- Curriculum: terrain levels ----------------------------------------------------
    if not play:
        cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
            func=microduck_mdp.qdd_terrain_levels_survival,
            params={
                "command_name": "twist",
                "max_err_xy": TERRAIN_UP_MAX_ERR_XY,
                "max_err_xy_by_type": TERRAIN_UP_MAX_ERR_XY_BY_TYPE,
            },
        )

    return cfg


def _spec_with_sole_scan_sites():
    """Robot spec + a site SOLE_SCAN_LIFT above each sole (foot_site)."""
    spec = get_qdd_biped_spec()
    for side, name in zip(("l", "r"), SOLE_SCAN_SITES):
        site = spec.site(f"{side}_foot_site")
        pos = list(site.pos)
        spec.body(f"{side}_foot").add_site(name=name, pos=[pos[0], pos[1], pos[2] + SOLE_SCAN_LIFT])
    return spec
