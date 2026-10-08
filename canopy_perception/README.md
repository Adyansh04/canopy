# canopy_perception

The world model's perception front end. A detector asks a host model server what is in each
camera frame and publishes instance masks; a describer asks the same server to name objects and
rooms; a segmenter answers one text prompt about one camera's newest frame, on request; a mock
detector cuts the masks a detector would publish out of a simulator's ground truth.
Nothing here knows which model answered.

```mermaid
flowchart LR
    C[/color/image_raw/] --> D[detector]
    D <-- "segment (ZMQ)" --> S[(servers/semantic_server.py)]
    D -- InstanceMaskArray --> W[canopy]
    T[/ground-truth boxes + depth/] --> M[mock_detector]
    M -- InstanceMaskArray --> W
    W -- DescribeRequest --> O[object_describer]
    O <-- "describe (ZMQ)" --> S
    O -- Description --> W
```

```bash
colcon build --symlink-install --packages-select canopy_perception
```

## Nodes

| Node | Does |
|---|---|
| `detector` | Sends the newest camera frame and the phrase list to the model server and publishes the masks it answers with, one call per `detect_rate_hz`. With `embed` it also asks for an image embedding per instance. With `still_frames` it reads the world model's frames from a still base instead, the only ones the world model uses masks from: for a model slower than a frame, such as SAM 3.1. |
| `object_describer` | Sends each `DescribeRequest` to the server's `describe` endpoint (its own port, `tcp://127.0.0.1:5562` by default), one at a time, and publishes the answer. A refusal is logged and dropped: the world model asks again. |
| `segmenter` | A service: segments the newest frame of a named camera by a text prompt ("the floor", "every mug") and answers with the masks, stamped with that frame. Apart from the world model, so an agent's floor never becomes an object. |
| `mock_detector` | Cuts masks out of ground-truth boxes against the rendered depth: every pixel whose depth lands inside a box. No GPU, no server. Simulation only. A phrase matches every numbered body of its class: "chair" finds `chair_1` to `chair_5`. |

## Interfaces

| Direction | Name | Type |
|---|---|---|
| Sub | `color/image_raw`, or `still_image` with `still_frames` (detector) | `sensor_msgs/Image`, best effort, depth 1 |
| Sub | `object_poses`, `depth/image_raw`, `camera_info` (mock) | `vision_msgs/Detection3DArray` in the camera frame, `sensor_msgs/Image`, `sensor_msgs/CameraInfo` |
| Pub | `~/instance_masks` (both detectors) | `canopy_msgs/InstanceMaskArray`, reliable, stamped with the image's own stamp |
| Sub | `describe_requests` (describer) | `canopy_msgs/DescribeRequest` |
| Pub | `~/descriptions` (describer) | `canopy_msgs/Description` |
| Sub | each topic in `cameras` (segmenter) | `sensor_msgs/Image`, best effort, depth 1 |
| Srv | `~/segment` (segmenter) | `canopy_msgs/Segment`: a camera name and a prompt in, `InstanceMaskArray` out, or `success` false and why |

`phrases` is the detector's whole control interface: an empty list idles it, and
`ros2 param set /detector phrases "['chair', 'mug']"` changes it while it runs. An entry may be
`name=phrase`, asking for the wording that separates an object best and publishing the short name.

## Configuration

`config/detector.yaml` is keyed `/**`, so a detector per camera reads it whatever its name:

| Parameter | Default | Meaning |
|---|---|---|
| `server_address` | `tcp://127.0.0.1:5561` | The model server. |
| `detect_rate_hz` | 1.0 | How soon after an answer the next question is asked. |
| `still_frames` | true | Read `still_image`, the world model's frames from a still base, rather than `color/image_raw`: SAM 3.1's 1.6 s a frame are spent only on frames the world model keeps masks from. |
| `max_image_age_s` | 8.0 | Frames older than this are not asked about. A still frame may wait for the other camera's turn; the world model's `frame_history_s` covers it. |
| `box_threshold` | 0.6 | Detections under this score are dropped. SAM 3.1's score is its match times the word's presence in the frame. |
| `embed` | true | Ask for an image embedding per instance. |
| `phrases` | 34 whole-object words | One name per kind of object, which the world model's room table is keyed on; its synonyms map other words onto them. |

`config/detector_yoloe.yaml` goes over these for YOLOE-26: every frame of `color/image_raw`, 2.5 s,
0.30 and a 128-word list, which YOLOE answers in one pass.

The segmenter takes `server_address`, `max_image_age_s`, `box_threshold`, `text_threshold`,
`zmq_timeout_ms` and `cameras`, a list of `name=topic` such as
`["chest=/chest_camera/color/image_raw"]`.

`config/mock_detector.yaml` sets the mock's latency, rate, box margin and minimum mask size.

## Library

`canopy_perception_core`, for any node that pairs masks with depth:

- `DepthHistory` keeps recent frames, so a mask that arrives seconds late is paired with the frame
  it was cut from rather than the newest one.
- `DepthView` and `Intrinsics` read a 32FC1 depth image and its pinhole parameters.
- `slugify` turns a phrase into the id it is published under: "Red Block" is `red_block`.

## Why three nodes are Python

`detector`, `segmenter` and `object_describer` are ZMQ and msgpack clients. The models need torch and CUDA,
which the robot's image need not carry, so they run on the host (`servers/`) and these nodes speak
their wire protocol. The socket handling lives in `canopy_perception/host_clients.py`.

## Running

Start the server and the world model first (`servers/README.md`, `canopy/README.md`), then one
detector per camera, on the world model's still frames of it:

```bash
ros2 run canopy_perception detector --ros-args \
  --params-file $(ros2 pkg prefix canopy_perception)/share/canopy_perception/config/detector.yaml \
  -r still_image:=/canopy/camera/still_image
```

With `still_frames: false`, as in `config/detector_yoloe.yaml`, remap `color/image_raw` to the
camera's image instead. `canopy/launch/world_model.launch.py` starts the detectors with these
remaps.

## Tests

| Test | Covers |
|---|---|
| `test_depth_history` | Pairing a late mask with its own frame, refusing one outside the tolerance, and dropping frames past the window or the frame cap. |
| `test_phrase` | Phrases as ids. |
| `test_detector` | The detector against a stub server: the request encoding, the mask message, the image's own stamp, and a phrase list that is re-read rather than cached. |
| `test_segmenter` | The segmenter against the same stub: masks for the named camera's frame with its stamp, and a plain answer for an unknown camera or an empty prompt. |
