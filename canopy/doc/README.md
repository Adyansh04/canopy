# The apartment, explored

`apartment/` is the world grove-g1's acceptance test saved after a simulated Unitree G1 explored a
six-room flat from no map, with a head and a chest camera and the real models (YOLOE-26, SigLIP 2
and a Gemini describer), exported with the test's `G1_EXPLORE_TEST_EXPORT=1`:
- `map.pgm` and `map.yaml`: the floor plan.
- `semantic_map.png`: a picture of it with the rooms and objects drawn in.
- `world.yaml`: every room and object.
- `coverage.bin`, `objects.bin` and `wall_hits.png`: the layers the world model and the C++
  replays load.
- `run.yaml`: how the run was made and what the test scored, including where the map lies in the
  simulator's world (`map_to_world`), then what the corrected world scores.
- `edits.log`: every correction made by hand in the map editor, one JSON line each.

The models found 36 of the 44 objects the cameras can see. The rest were missed, mislabelled (the
wardrobe as a cabinet), or doubled by phantoms (the floor as a desk). canopy's `~/clean_up` then
took out the floor twice and a stool that was part of the coffee table. The rest was corrected in
the map editor against the simulator's truth: 16 labels, 11 merges of pieces into their object, 5
wall slivers and a phantom removed, 5 missed objects added, 3 room types, and 17 boxes. Most were
well off; a kettle and two mugs were turned square to what they stand on, as the robot should face
them. It now holds all 44 objects and nothing else. The object crops are left out. The pictures,
made with grove-g1's `g1_bringup` tools:
- `apartment_truth.png` draws the scene's walls (blue) and furniture (orange) over the saved floor
  plan (`compare_truth.py --overlay`).
- `apartment_wardrobe.png` is the corner behind the bedroom's wardrobe, from the same command with
  `--crop 4 2.5 10.5 7.2`. The gap is narrower than a spot the robot can stand in, so it looks in
  from the nearest one, and the LiDAR sees only part of it.
- `apartment_mujoco.jpg` is the flat itself, rendered offscreen from grove-g1's
  `g1_apartment_scene.xml` with MuJoCo's own renderer, from the south-west.
- `apartment_walk.png` is the robot's walk, blue early to red late, with each viewpoint
  (`run_summary.py --plot`), from an earlier run with the mock detector.
- `frontier_pass_end.png` is SLAM's map as the frontier pass ended, from an earlier run
  (`snapshot_map.py`). Free is white, occupied black, unknown grey, and frontiers red. Most of the
  red is speckle between LiDAR beams, which the camera pass fills in.

`viewer/` is RViz during another run of the same flat (grove-g1's run 57): `frontier_pass.png` at
4 minutes, `camera_pass.png` at 21, and `done.png` once every room was seen.
