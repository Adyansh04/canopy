# editor

A web page for checking a saved world by hand once a run is over. Models make mistakes a person
fixes in seconds: a dining table labelled "bowl", the floor mapped as a desk, a storage room typed
"office". The page lists what deserves a second look and turns each fix into one edit.

```bash
python3 editor/canopy_editor.py /data/worlds/home      # then open http://127.0.0.1:8765/
ros2 service call /canopy/reload std_srvs/srv/Trigger  # if canopy is still running on that world
```

Python and PyYAML, nothing else, rather than C++: a human-paced tool that edits files and serves
one page. `--host 0.0.0.0` lets a tablet on the same network edit, and anyone else there too.

## What it does

- **Objects:** set what it is and its name; mark it checked; delete it; merge it with another; split
  a part off by drawing a box; move, turn or resize its box; add one the models missed.
- **Rooms:** set the type and the name; mark it checked. Clearing a type hands the room back to
  canopy's typer.
- **Review:** each object and room that looks wrong, and why:
  - a phantom, weakly seen;
  - the floor itself;
  - the describer disagreeing with the label;
  - a split vote;
  - a piece inside something much stronger;
  - a room typed with low confidence.

  Checking an item takes it off the list.
- **Undo and redo** back to the last save, with nothing written until **Save**. Saving appends
  every edit to `edits.log` in the world directory.

canopy keeps an operator's label while its votes go on counting underneath, keeps a box they placed,
keeps a name they gave from the describer, and never lets the typer or describer retype a room they
typed. It refuses to save over a world edited since it last wrote it, until `~/reload` takes the
edits.

Best edit a world canopy is done with. If it is still running, it may save while you edit, every
minute by default. The editor then refuses to save, and offers to **rebase**: it reads canopy's
newer world and makes your edits again on it. Call `~/reload` after saving. `semantic_map.png` is
drawn by canopy, so it shows the edits only once canopy has saved again.

## The API

The page is one client of a JSON API; anything else can drive the same edits.

| Call | Does |
|---|---|
| `GET /api/world` | The map (resolution, origin, size), rooms with outlines, objects with votes, boxes, heights and crops, the review list, and undo state. |
| `GET /api/map.png`, `GET /api/crops/O12.jpg` | The floor plan, and what the camera saw of an object. |
| `POST /api/edit` | One edit, e.g. `{"op": "label", "id": 12, "label": "dining table"}`. Answers the new world, or 400 and why. |
| `POST /api/undo`, `/api/redo`, `/api/save`, `/api/reload` | Save answers 409 if canopy wrote the world since. |
| `POST /api/rebase` | After a 409: reads canopy's world and makes this session's edits again on it. |

The `op`s are:
- objects: `label`, `name`, `check`, `delete`, `merge` (`ids`, `into`), `split` (`id`, `polygon`,
  `label`), `box` (`id`, `centre`, `size`, `yaw`) and `add` (`label`, `centre`, `size`, `yaw`,
  `z_min`, `z_max`);
- rooms: `room_type`, `room_name` and `room_check`, each with `room`.

Coordinates are metres in the map frame, and yaws are radians.

```bash
python3 -m unittest editor/test_canopy_editor.py
```
