# canopy

The world model. On top of the SLAM map it keeps:

- **rooms** and the doorways between them,
- **objects**, each with a label, a box and the room it is in,
- **camera coverage**: how much of each room the cameras have actually seen.

It also plans where the robot should stand and look next, and answers "where is the dustbin" and
"how do I get there", so missions name targets instead of carrying coordinates.

![The saved semantic map](doc/apartment/semantic_map.png)

## How it works

Everything below is plain C++ with no ROS in it (`src/`, `include/canopy/`), so each part is unit
tested on its own. The node only wires it to topics and services.

### Rooms (`room_segmentation`)

- A room is a place where the free space is widest; two rooms meet where it narrows to a doorway.
- Rooms are cut on a **walls-only** version of the map. Furniture is taken out first:
  - anything the LiDAR sees over at wall height (1.4 to 2.2 m up) is furniture, not wall;
  - so is any blob standing clear of the walls with nothing above it.
- Corridors too narrow for a 2.4 m disc are split off as their own rooms.
- A room keeps its id (`R3`) when SLAM redraws the map.
- A room's type comes from the objects in it (a bed makes a bedroom), or from the describer when
  they leave it in doubt. A long, narrow room is a hallway whatever stands in it.

### Camera coverage (`coverage_map`)

- Three kinds of things to see: **floor**, **wall faces**, and the **tops** of tables and shelves.
- Every depth pixel credits the cell it lands on. Close, head-on and near the image centre count
  for more.
- Only frames taken while the robot stands still count: a frame taken mid-turn can land degrees
  off.
- When SLAM redraws a wall a cell or two away, as after a loop closure, the new wall keeps what
  the old one had been credited with.
- A table's top is the highest layer dense enough to be the table, so a mug standing on it does
  not lift the table.

### Where to look next (`viewpoint_planner`)

The robot explores in two passes, then tidies up:

1. **Frontier pass**: walk to the edges of the known map until the LiDAR map is closed. Shadows
   behind furniture that the robot has already looked past are ignored. Once a few visits in a
   row add little map, only a frontier that leads on, far from anywhere the robot has stood, is
   still walked to.
2. **Camera pass**: sample spots the robot can stand on, predict what each heading would show,
   and go where the most still-unseen area per second of walking, turning and waiting is.
3. **Pockets**: look into the unknown corners left inside the building, such as the floor behind
   a wardrobe, so their walls close in the saved map.

Along the way:

- What no standing spot can see (a shelf above the camera) is written off and reported per room,
  rather than chased.
- A room whose viewpoints Nav2 keeps refusing is given up.

### Objects (`object_map`)

- Each detector mask, lifted with depth, becomes a small cloud of voxels.
- It joins an existing object when they overlap and agree on label or embedding, which tolerates
  a localiser's decimetre drift. An embedding agrees only when close (`objects.kin_similarity`):
  crops of unlike things still look alike to an image embedder, and a bowl must not join its
  table.
- **Two sightings confirm an object.** Two cameras catching the same instant count once; until the
  second sighting the object is hidden, and the camera pass goes back to look again.
- Parts of one object seen from different sides merge when they touch.
- An object the camera should see but does not collects misses: stale, then removed.
- Furniture boxes snap to the outline the map shows under them, so an approach pose is computed
  from where the sofa really is. It stands square to a side, as near its middle as a short walk
  allows, and at a corner only when no side is clear (`approach.*` in `config/canopy.yaml`).

## Node: `canopy` (executable `world_model`)

| | Name | Type |
|---|---|---|
| Sub | `map` | `nav_msgs/OccupancyGrid` (latched) |
| Sub | `<camera>/depth/image_raw`, `<camera>/depth/camera_info`, `<camera>/color/image_raw` | each camera's aligned depth and colour |
| Sub | `<camera>/instance_masks` | `canopy_msgs/InstanceMaskArray` from canopy_perception's `detector` or `mock_detector` |
| Sub | `cloud` | `sensor_msgs/PointCloud2` from a LiDAR, for the walls-only map |
| Sub | `odom` | `nav_msgs/Odometry`, to tell when the robot stands still |
| Sub | `descriptions` | `canopy_msgs/Description` from canopy_perception's `object_describer` |
| Pub | `~/rooms` | `canopy_msgs/RoomArray` (latched): outline, doorways, type, coverage |
| Pub | `~/objects` | `canopy_msgs/WorldObjectArray` (latched) |
| Pub | `~/coverage` | `nav_msgs/OccupancyGrid` for RViz: 90 still to see, 99 written off |
| Pub | `~/floor_plan`, `~/walls` | `nav_msgs/OccupancyGrid` (latched): the saved floor plan, and the walls-only map |
| Pub | `~/trail` | `nav_msgs/Path` (latched): where the robot has walked |
| Pub | `~/markers` | `visualization_msgs/MarkerArray` for RViz: rooms, doorways, objects, viewpoints, the cameras' view on the floor |
| Pub | `~/describe_requests` | `canopy_msgs/DescribeRequest`, with `describe:=true` |
| Pub | `~/<camera>/still_image` | `sensor_msgs/Image`: that camera's frames from a still base, at most one a `still_frame_period_s`, while a detector with `still_frames` reads them |
| Srv | `~/next_viewpoint`, `~/report_viewpoint` | the exploration loop |
| Srv | `~/find_objects`, `~/get_approach_pose` | finding things |
| Srv | `~/object_history` | `canopy_msgs/ObjectHistory`: what happened to the objects a query names, oldest first: when each appeared, moved, went missing (looked at and not found), was seen again, removed or merged, and the object it may have become after a move |
| Srv | `~/save` | `std_srvs/Trigger`: writes the world to `world_dir` (it also saves every minute), unless someone else changed it on disk since |
| Srv | `~/reload` | `std_srvs/Trigger`: takes the world in `world_dir` as it is on disk now, e.g. after the map editor |
| Srv | `~/clean_up` | `std_srvs/Trigger`: once exploring is over, marks removed what is surely not an object (the floor, a piece inside something far more seen), with the reason; the map editor can bring any back |

- **Parameters**, each with its reason: `config/canopy.yaml`.
- **Room types and synonyms**: `config/room_types.yaml`. "dustbin" finds what the detector calls a
  trash can.
- **Cameras** are configuration only. `cameras` lists them, and each one's height, pitch, heading
  and field of view are read from TF and `camera_info`. The camera pass waits until at least one
  mount is known.

## What is saved

`world_dir` holds everything needed to see and reuse what the robot learnt:

| File | What it is |
|---|---|
| `map.pgm`, `map.yaml` | The floor plan, as `map_saver` writes it: SLAM's map with every room closed and furniture solid. |
| `semantic_map.png` | That plan with rooms tinted and named, and every object's box and label. |
| `world.yaml` | Rooms and objects in plain text: ids, names, types, boxes, what rests on what, and the newest `max_events` of what happened to them. |
| `objects.bin`, `coverage.bin`, `crops/` | Object voxels, what the camera has seen with the map SLAM made, and each object's best view. |
| `wall_hits.png` | The LiDAR's returns at wall height, to rebuild the floor plan offline. |

A later run with the same `world_dir` on the same map picks all of it up and carries on. What an
operator fixed with the [map editor](../editor) stays fixed: their labels, boxes and room types
win over later votes, the room typer and the describer.

## Running it

```bash
ros2 launch canopy world_model.launch.py world_dir:=/data/worlds/home rviz:=true \
  cameras:=head=/camera,chest=/chest_camera odom_topic:=/odom cloud_topic:=/lidar/points \
  detector:=true describe:=true params_file:=my_robot_canopy.yaml
```

- `cameras`: each camera's name and its RealSense-style driver namespace.
- `detector:=true` starts canopy_perception's detector per camera, against the host model server
  in `servers/` (start that first). In a simulator, `mock_detector` can publish the same masks.
- `params_file`: the robot's own values over `config/canopy.yaml`, such as its footprint radius,
  walking and turning speeds, and how long to wait for the detector.
- The robot needs a latched `/map`, TF from `map` to its base and cameras, and something to walk
  it through the loop below.

### The exploration loop

Whatever drives the robot (a behavior tree, a state machine, an agent) repeats:

1. `~/next_viewpoint` in `frontier` mode until it answers done. Walk to each pose, face each
   heading, then `~/report_viewpoint` whether the pose was reached.
2. The same in `coverage` mode, waiting a couple of seconds at each heading for the detector.
3. `~/clean_up`, then `~/save`.

`~/next_viewpoint` may also answer *unavailable* (no map or pose yet, or SLAM still catching up
with the last viewpoint: ask again shortly) or *stuck* (the last few viewpoints all went unreached
from where the robot stands: move it off, say a step to more clearance, and ask again).

Nav2's `NavigateToPose` and `Spin` are enough for the walking. grove-g1's `explore.xml` is one
such tree.

## Tools

In `scripts/`; `ros2 run canopy <tool> --help` lists every option.

| Tool | What it gives |
|---|---|
| `track_run.py PREFIX` | A run's record: the robot's pose, each room's coverage over time, the rooms' outlines. |
| `snapshot_map.py OUT.png` | SLAM's map as it stands, with the frontier cells the frontier pass plans on. |

Both are Python, not C++: they sit beside a run at a few hertz or once, write CSV, YAML and PNG
with the standard library and Pillow, and nothing the robot does waits on them.

`doc/` holds a flat a simulated Unitree G1 explored, and pictures of it.

## Tests

No simulator:

- `test_room_segmentation`: hand-drawn plans and two saved maps (`test/maps`).
- `test_viewpoint_planner`: a ray-cast camera covering two rooms and a corridor.
- `test_object_map`, `test_world_parts`.
- `test_world_edit`: the node resumes the example world, and the map editor changes it on disk.
  The node must refuse to save over that, take it with `~/reload`, and keep it.

The end-to-end run lives with a robot: grove-g1 explores its simulated flat and scores rooms,
coverage and objects against ground truth.

## Credits

Adapted from published work, reimplemented here:

| Idea | Source |
|---|---|
| Rooms from clearance | Hydra, MIT SPARK: https://github.com/MIT-SPARK/Hydra |
| Detection-to-object association | ConceptGraphs: https://github.com/concept-graphs/concept-graphs |
| Duplicate merging | OVO: https://github.com/tberriel/OVO |
| Absence evidence, "not found" answers | DynaMem: https://github.com/hello-robot/stretch_ai |
| Image-edge weight | VLFM: https://github.com/bdaiinstitute/vlfm |
| Viewpoints with headings, room-level ordering | FUEL: https://github.com/HKUST-Aerial-Robotics/FUEL, TARE: https://github.com/caochao39/tare_planner |
| Merge rules for small regions | Bormann et al., ICRA 2016: https://github.com/ipa320/ipa_coverage_planning |
