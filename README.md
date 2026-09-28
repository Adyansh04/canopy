# canopy

A world model for ROS 2 robots that explore buildings. On top of the SLAM map it keeps:

- the **rooms** and the doorways between them,
- the **objects** in each room, found by an open-vocabulary detector,
- how much of every room the robot's **cameras** have actually seen.

It plans where to stand and look next, and answers "where is the dustbin" and "how do I get
there", so missions name targets instead of carrying coordinates. The models run on the host GPU,
and the ROS messages never name one.

```mermaid
flowchart LR
    CAM[/RGB-D cameras/] --> DET[canopy_perception<br/>detector]
    DET <-- ZMQ --> SRV[(servers/<br/>semantic_server.py)]
    DET -- InstanceMaskArray --> WM[canopy<br/>world model]
    CAM --> WM
    MAP[/SLAM map, LiDAR, TF/] --> WM
    WM -- DescribeRequest --> DES[canopy_perception<br/>object_describer]
    DES <-- ZMQ --> SRV
    DES -- Description --> WM
    WM -- rooms, objects,<br/>viewpoints --> BT[your behavior tree<br/>or agent]
```

| Part | What it is |
|---|---|
| [`canopy`](canopy) | The world model: rooms, camera coverage, objects, the viewpoint planner, and the ROS node around them. |
| [`canopy_msgs`](canopy_msgs) | Its interfaces: instance masks, rooms, objects, describer requests, and the search and exploration services. |
| [`canopy_perception`](canopy_perception) | The front end: a detector and a describer that ask the model servers, and a mock detector for simulators. |
| [`servers`](servers) | The models, on the host GPU: YOLOE-26 detection, SigLIP 2 embeddings, and Gemini or a local Qwen VLM for names and room types. Not a ROS package. |
| [`editor`](editor) | A web page for checking a saved world by hand after a run: relabel, rename, delete, merge, split, move or add objects, and retype rooms. Not a ROS package. How to use it: [editor/doc/guide.md](editor/doc/guide.md). |

Built for ROS 2 Jazzy (Ubuntu 24.04), C++20.

## In pictures

A simulated Unitree G1 exploring a six-room flat from no map, watched in RViz (`rviz:=true`):

- each room is tinted and labelled with how much of it the cameras have seen;
- **red** is still to see, **cyan** what no standing spot can see;
- big green discs are doorways, small dots the viewpoints, the blue line the robot's trail;
- objects are boxed in their room's colour, and the head camera's picture sits top left.

| Frontier pass, 4 min | Camera pass, 21 min | Done, 35 min |
|---|---|---|
| ![Early in the frontier pass](canopy/doc/viewer/frontier_pass.png) | ![Halfway through the camera pass](canopy/doc/viewer/camera_pass.png) | ![Exploration finished](canopy/doc/viewer/done.png) |
| The LiDAR map is still growing; the robot walks to its edges. | The map is closed; the robot goes where the cameras have seen least. | Every room 99 % seen. Asked for "the dustbin in the office", the robot walked to it. |

What the run leaves in `world_dir`:

| Semantic map | Against ground truth | The walk |
|---|---|---|
| ![Semantic map](canopy/doc/apartment/semantic_map.png) | ![Floor plan against the simulator's truth](canopy/doc/apartment_truth.png) | ![The robot's walk](canopy/doc/apartment_walk.png) |
| `semantic_map.png`: rooms named, objects boxed and labelled, after the clean-up and a check in the map editor. | The saved floor plan with the scene's real walls (blue) and furniture (orange) drawn over it. | The robot's path, blue early to red late, and each viewpoint. |

The [map editor](editor/doc/guide.md) is for checking that world by hand afterwards:

![The map editor with the explored flat open](editor/doc/overview.png)

## Building

In a colcon workspace:

```bash
cd ~/ws/src && git clone https://github.com/Adyansh04/canopy.git
cd ~/ws && rosdep install --from-paths src --ignore-src -y
pip install msgpack-numpy==0.4.8            # no rosdep key; the Python nodes need it
colcon build --symlink-install
```

The model servers run outside ROS, on the host's GPU:

```bash
cd ~/ws/src/canopy
./servers/setup.sh                            # a uv virtualenv under ~/.local/share/canopy
./servers/start-vlm.sh start                  # optional: the offline describer
~/.local/share/canopy/.venv/bin/python servers/semantic_server.py
```

## Running it on a robot

```bash
ros2 launch canopy world_model.launch.py world_dir:=/data/worlds/home rviz:=true \
  cameras:=head=/camera odom_topic:=/odom cloud_topic:=/lidar/points \
  detector:=true describe:=true params_file:=my_robot_canopy.yaml
```

The robot needs:

- a latched `/map` from SLAM or `map_server`, and TF from `map` to its base and each camera;
- RGB-D cameras that publish as a RealSense driver does, and a LiDAR point cloud;
- its own values in `params_file`: footprint radius, walking and turning speeds, how long to wait
  for the detector;
- something to walk it through the exploration loop that [`canopy`](canopy#the-exploration-loop)
  describes. Nav2's `NavigateToPose` and `Spin` are enough.

## How well it does

On the simulated G1 above, from no map, with a head and a chest camera
([grove-g1](https://github.com/Adyansh04/grove-g1), where canopy began). The flat has 43 objects a
standing robot can see.

| Detector | Rooms | Walls seen | Objects found | Boxes (median IoU) | Time |
|---|---|---|---|---|---|
| Ground-truth masks | 6 of 6 | 96-99 % | 100 % | 0.76-0.80 | 30-34 min |
| YOLOE-26 + SigLIP 2 + describer | 6 of 6 | 95-99 % | 72-79 % | 0.82-0.83 | 37-42 min |

With the real detector:

- YOLOE confuses furniture of one material on the simulator's renders: desk, cabinet, TV stand.
- It misses mugs and bowls on tables.
- The describer names most of them correctly.

## Credits

- The world model reimplements ideas from Hydra, ConceptGraphs, OVO, DynaMem, VLFM, FUEL, TARE and
  Bormann et al.; [`canopy`](canopy#credits) links each.
- The models the servers run are credited in [`servers`](servers#models-and-libraries).

BSD 3-Clause; see [LICENSE](LICENSE).
