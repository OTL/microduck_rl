# Get-up keyframe search (QDD biped)

Plain-MuJoCo CEM search for open-loop PD keyframe motions that get the robot up,
used to check that a get-up is physically possible with the hardware limits and
to seed the RL task with states along a working motion (the "path" kind of
`scripts/qdd_getup_states.py`). Run from the repo root with `uv run python`.
Results go to `$GETUP_SEARCH_OUT` (default /tmp/qdd_getup_search).

1. `feas2.py prone` - 4 asymmetric keyframes from lying on the front
   (`--relax`: wider hip / knee / ankle ranges, no knee_max coupling).
   `GOAL=prone feas2.py supine` - roll from the back onto the front.
2. `lunge.py OUT/best2_prone_hw_0.npz 1.3` - from the upright split reached at
   1.3 s, 3 keyframes to standing.
3. `export_traj.py src/mjlab_microduck/robot/qdd_biped/getup_demo_traj.npz` -
   replay and save qpos / qvel every 20 ms.
4. `show.py <best.npz> out.png` (MUJOCO_GL=glfw xvfb-run ...) - picture strip.

2026-10-18: prone -> standing found within the current limits (one leg kicks
back, torso up into a split, legs together; ~4 s, |tau| <= 11 Nm, |qd| <= 12
rad/s, torso up to 6 rad/s). From the back: no direct get-up found; rolling onto
the front is.
