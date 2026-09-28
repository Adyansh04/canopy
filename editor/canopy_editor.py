#!/usr/bin/env python3
"""A map editor for a saved canopy world: fix what the models got wrong, after the run.

    python3 editor/canopy_editor.py WORLD_DIR [--port 8765] [--host 127.0.0.1]

Serves one page at http://HOST:PORT/ and a JSON API under /api/ that the page, or any other
client, drives. Edits change the world in memory, with undo, until /api/save writes world.yaml
and objects.bin back. A running canopy takes them with its ~/reload service; until then it
refuses to save over them. Python, not C++: a human-paced tool whose work is file edits and a web
page, run beside the robot rather than on it.
"""

import argparse
import array
import copy
import json
import math
import os
import re
import struct
import sys
import tempfile
import threading
import time
import zlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parent / "canopy" / "config"
VOCABULARY = HERE.parent / "canopy_perception" / "config" / "detector.yaml"

# canopy's voxel keys: 21 bits an axis, offset so negative cells fit (object_map.cpp).
KEY_BITS = 21
KEY_MASK = (1 << KEY_BITS) - 1
KEY_OFFSET = 1 << 20
DEFAULT_VOXEL = 0.04
# Share of footprint points left off each box side, as canopy's objects.box_trim.
BOX_TRIM = 0.02
UNDO_DEPTH = 100

# Review rules, measured on four real-detector runs of the test flat: the weak and floor rules
# flagged 33 of 77 objects that matched nothing, and 2 of 132 true ones.
WEAK_WEIGHT = 1.2
WEAK_SIGHTINGS = 4
FLOOR_TOP = 0.14
FLAT_LABELS = {"rug", "mat", "carpet", "doormat"}
SPLIT_SHARE = 0.4
FLOOR_STANDING = 0.15
PIECE_SHARE = 0.5
PIECE_RATIO = 4.0
LOW_ROOM_CONFIDENCE = 0.6


class EditError(ValueError):
    """An edit the world cannot take; the API answers 400 with it."""


class Conflict(RuntimeError):
    """Someone else wrote the world since it was opened; the API answers 409."""


# --- voxels ---------------------------------------------------------------------------------


def key_of(x, y, z, voxel):
    def axis(value):
        cell = math.floor(value / voxel) + KEY_OFFSET
        return min(max(cell, 0), KEY_MASK)

    return (axis(x) << (2 * KEY_BITS)) | (axis(y) << KEY_BITS) | axis(z)


def centre_of(key, voxel):
    def axis(bits):
        return ((bits & KEY_MASK) - KEY_OFFSET + 0.5) * voxel

    return axis(key >> (2 * KEY_BITS)), axis(key >> KEY_BITS), axis(key)


def shifted(key, dx, dy, dz):
    def step(bits, delta):
        return ((bits & KEY_MASK) + delta) & KEY_MASK

    return (
        (step(key >> (2 * KEY_BITS), dx) << (2 * KEY_BITS))
        | (step(key >> KEY_BITS, dy) << KEY_BITS)
        | step(key, dz)
    )


def overlap(inner, outer):
    """Share of a sample of `inner`'s voxels within one voxel of `outer`'s, as canopy measures it."""
    if not inner or not outer:
        return 0.0
    keys = sorted(inner)
    sample = keys[:: max(1, len(keys) // 600)]
    near = sum(
        1
        for key in sample
        if any(
            shifted(key, dx, dy, dz) in outer
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        )
    )
    return near / len(sample)


def fit_box(keys, voxel, yaw):
    """The footprint box of `keys` along `yaw`, trimmed as canopy trims it: (centre, size)."""
    c, s = math.cos(yaw), math.sin(yaw)
    along, across = [], []
    for key in keys:
        x, y, _ = centre_of(key, voxel)
        along.append(x * c + y * s)
        across.append(y * c - x * s)
    along.sort()
    across.sort()
    trim = int(BOX_TRIM * len(along))
    low_u, high_u = along[trim], along[len(along) - 1 - trim]
    low_v, high_v = across[trim], across[len(across) - 1 - trim]
    mid_u, mid_v = (low_u + high_u) / 2, (low_v + high_v) / 2
    centre = [mid_u * c - mid_v * s, mid_u * s + mid_v * c]
    return centre, [high_u - low_u + voxel, high_v - low_v + voxel]


def z_range(keys, voxel):
    zs = [centre_of(key, voxel)[2] for key in keys]
    return min(zs) - voxel / 2, max(zs) + voxel / 2


def inside(polygon, x, y):
    """Even-odd test of a point against a polygon of [x, y] corners."""
    hit = False
    for i, (x0, y0) in enumerate(polygon):
        x1, y1 = polygon[(i + 1) % len(polygon)]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            hit = not hit
    return hit


def box_corners(centre, size, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return [
        [centre[0] + c * u - s * v, centre[1] + s * u + c * v]
        for u, v in (
            (size[0] / 2, size[1] / 2),
            (-size[0] / 2, size[1] / 2),
            (-size[0] / 2, -size[1] / 2),
            (size[0] / 2, -size[1] / 2),
        )
    ]


def shell(centre, size, yaw, z_min, z_max, voxel):
    """Voxel keys over a box's sides and top: what a camera would have measured of it."""
    keys = set()
    c, s = math.cos(yaw), math.sin(yaw)
    half_u, half_v = size[0] / 2, size[1] / 2
    step = voxel * 0.9
    us = [-half_u + i * step for i in range(int(size[0] / step) + 1)] + [half_u]
    vs = [-half_v + i * step for i in range(int(size[1] / step) + 1)] + [half_v]
    zs = [z_min + voxel / 2 + i * step for i in range(max(1, int((z_max - z_min) / step)))]

    def add(u, v, z):
        keys.add(key_of(centre[0] + c * u - s * v, centre[1] + s * u + c * v, z, voxel))

    for z in zs:
        for u in us:
            add(u, -half_v, z)
            add(u, half_v, z)
        for v in vs:
            add(-half_u, v, z)
            add(half_u, v, z)
    top = z_max - voxel / 2
    for u in us:
        for v in vs:
            add(u, v, top)
    return keys


# --- the world on disk ----------------------------------------------------------------------


class World:
    """A canopy world directory: world.yaml, objects.bin, the map and the crops."""

    def __init__(self, directory, config=CONFIG):
        self.directory = Path(directory)
        self.lock = threading.Lock()
        self.history = []  # (before, after) images of what each edit touched.
        self.future = []
        self.edits = []  # This session's ops, appended to edits.log on save.
        self.load()
        self.labels, self.synonyms, self.room_types = read_config(Path(config))

    # -- loading and saving --

    def load(self):
        self.stamp = os.stat(self.directory / "world.yaml").st_mtime_ns
        with open(self.directory / "world.yaml") as stream:
            self.yaml = yaml.safe_load(stream)
        if self.yaml.get("version") != 1:
            raise ValueError("world.yaml has an unknown version")
        self.voxel = float(self.yaml.get("voxel") or DEFAULT_VOXEL)
        self.rooms = {room["id"]: room for room in self.yaml.get("rooms") or []}
        self.objects = {}
        records = {int(record["id"]): record for record in self.yaml.get("objects") or []}
        data = (self.directory / "objects.bin").read_bytes()
        self.magic, version, count = struct.unpack_from("<III", data, 0)
        at = 12
        for _ in range(count):
            object_id, voxels = struct.unpack_from("<iI", data, at)
            at += 8
            keys = array.array("Q", data[at : at + 8 * voxels])
            at += 8 * voxels
            dimensions, embedded = struct.unpack_from("<Ii", data, at)
            at += 8
            embedding = list(struct.unpack_from(f"<{dimensions}f", data, at))
            at += 4 * dimensions
            record = records.get(object_id)
            if record is None:
                continue  # canopy drops these too: voxels with no record are a torn save.
            record["_voxels"] = set(keys)
            record["_embedding"] = embedding
            record["_embedded"] = embedded
            self.objects[object_id] = record
        self.map = read_map(self.directory)
        self.history.clear()
        self.future.clear()
        self.edits.clear()

    def save(self):
        with self.lock:
            if os.stat(self.directory / "world.yaml").st_mtime_ns != self.stamp:
                raise Conflict(
                    "world.yaml changed on disk since the editor opened it: canopy saved it. "
                    "Reload the world here, or your edits would undo canopy's."
                )
            body = copy.copy(self.yaml)
            body["rooms"] = list(self.rooms.values())
            body["objects"] = [
                {k: v for k, v in record.items() if not k.startswith("_")}
                for record in self.objects.values()
            ]
            blob = [struct.pack("<III", self.magic, 1, len(self.objects))]
            for object_id, record in self.objects.items():
                keys = array.array("Q", sorted(record["_voxels"]))
                embedding = record["_embedding"]
                blob.append(struct.pack("<iI", object_id, len(keys)))
                blob.append(keys.tobytes())
                blob.append(struct.pack("<Ii", len(embedding), record["_embedded"]))
                blob.append(struct.pack(f"<{len(embedding)}f", *embedding))
            # objects.bin first and world.yaml last, as canopy writes them: a torn save leaves a
            # world.yaml that still names what objects.bin holds.
            write_atomically(self.directory / "objects.bin", b"".join(blob))
            text = yaml.safe_dump(body, sort_keys=False, default_flow_style=None, width=100)
            write_atomically(self.directory / "world.yaml", text.encode())
            crops = self.directory / "crops"
            if crops.is_dir():
                for crop in crops.glob("O*.jpg"):
                    number = crop.stem[1:]
                    if number.isdigit() and int(number) not in self.objects:
                        crop.unlink()
            if self.edits:
                with open(self.directory / "edits.log", "a") as log:
                    for edit in self.edits:
                        log.write(json.dumps(edit) + "\n")
            self.stamp = os.stat(self.directory / "world.yaml").st_mtime_ns
            saved = len(self.edits)
            self.edits.clear()
            return f"saved {saved} edits to {self.directory}"

    # -- edits --

    def apply(self, op):
        with self.lock:
            name = op.get("op")
            handler = getattr(self, f"_op_{name}", None) if isinstance(name, str) else None
            if handler is None:
                raise EditError(f"unknown op {name!r}")
            objects, rooms = self._touched(op)
            before = self._image(objects, rooms)
            created = handler(op) or []
            # What an edit made did not exist before it: undo takes it away again.
            before["objects"].update(dict.fromkeys(created))
            after = self._image(objects | set(created), rooms)
            self.history.append((before, after))
            del self.history[:-UNDO_DEPTH]
            self.future.clear()
            self.edits.append({"time": time.time(), **op})
            return created

    def undo(self):
        with self.lock:
            if not self.history:
                raise EditError("nothing to undo")
            before, after = self.history.pop()
            self._restore(before)
            self.future.append((before, after))
            self.edits.append({"time": time.time(), "op": "undo"})

    def redo(self):
        with self.lock:
            if not self.future:
                raise EditError("nothing to redo")
            before, after = self.future.pop()
            self._restore(after)
            self.history.append((before, after))
            self.edits.append({"time": time.time(), "op": "redo"})

    def _touched(self, op):
        ids = set()
        for key in ("id", "into"):
            if key in op:
                ids.add(self._object(op[key]))
        for value in op.get("ids", []):
            ids.add(self._object(value))
        rooms = {self._room(op["room"])} if "room" in op else set()
        return ids, rooms

    def _image(self, objects, rooms):
        return {
            "objects": {i: copy.deepcopy(self.objects.get(i)) for i in objects},
            "rooms": {r: copy.deepcopy(self.rooms[r]) for r in rooms},
        }

    def _restore(self, image):
        for object_id, record in image["objects"].items():
            if record is None:
                self.objects.pop(object_id, None)
            else:
                self.objects[object_id] = copy.deepcopy(record)
        for room_id, record in image["rooms"].items():
            self.rooms[room_id] = copy.deepcopy(record)

    def _object(self, value):
        object_id = int(str(value).lstrip("O"))
        if object_id not in self.objects:
            raise EditError(f"no object O{object_id}")
        return object_id

    def _room(self, value):
        if value not in self.rooms:
            raise EditError(f"no room {value}")
        return value

    def _next_id(self):
        return max(self.objects, default=0) + 1

    def _refit(self, record, yaw=None):
        """Recompute a record's box from its voxels, as canopy will when it loads the world."""
        if not record["_voxels"]:
            raise EditError(f"O{record['id']} would have no voxels left")
        yaw = record["yaw"] if yaw is None else yaw
        centre, size = fit_box(record["_voxels"], self.voxel, yaw)
        z_low, z_high = z_range(record["_voxels"], self.voxel)
        record.update(centre=centre, size=[size[0], size[1], z_high - z_low], yaw=yaw)

    def _op_label(self, op):
        record = self.objects[self._object(op["id"])]
        votes = record.get("votes") or {}
        record["operator_label"] = str(op.get("label", "")).strip().lower()
        # As canopy writes it: the operator's label, or the votes' when that is cleared.
        record["label"] = record["operator_label"] or (max(votes, key=votes.get) if votes else "")

    def _op_name(self, op):
        self.objects[self._object(op["id"])]["name"] = str(op.get("name", "")).strip()

    def _op_check(self, op):
        self.objects[self._object(op["id"])]["checked"] = bool(op.get("checked", True))

    def _op_delete(self, op):
        del self.objects[self._object(op["id"])]

    def _op_merge(self, op):
        into = self.objects[self._object(op["into"])]
        others = [self.objects[self._object(i)] for i in op["ids"] if self._object(i) != into["id"]]
        if not others:
            raise EditError("merge needs a second object")
        # The embedding as canopy keeps it: a mean of the views, weighted by how many made each.
        vectors = [
            (r["_embedding"], max(r["_embedded"], 1)) for r in [into, *others] if r["_embedding"]
        ]
        for other in others:
            into["_voxels"] |= other["_voxels"]
            for label, votes in other["votes"].items():
                into["votes"][label] = into["votes"].get(label, 0.0) + votes
            into["observations"] += other["observations"]
            into["first_seen"] = min(into["first_seen"], other["first_seen"])
            into["last_seen"] = max(into["last_seen"], other["last_seen"])
            into["misses"] = min(into["misses"], other["misses"])
            into["top_seen"] = into["top_seen"] or other["top_seen"]
            into["best_view_score"] = max(into["best_view_score"], other["best_view_score"])
            into["name"] = into["name"] or other["name"]
            into["caption"] = into["caption"] or other["caption"]
            del self.objects[other["id"]]
        if vectors and all(len(v) == len(vectors[0][0]) for v, _ in vectors):
            total = sum(n for _, n in vectors)
            mean = [sum(v[i] * n for v, n in vectors) / total for i in range(len(vectors[0][0]))]
            norm = math.sqrt(sum(x * x for x in mean)) or 1.0
            into["_embedding"] = [x / norm for x in mean]
            into["_embedded"] = total
        votes = into["votes"]
        into["label"] = into.get("operator_label") or max(votes, key=votes.get)
        into["state"] = "active"
        into["box_pinned"] = False
        self._refit(into)

    def _op_box(self, op):
        record = self.objects[self._object(op["id"])]
        centre, size, yaw = op["centre"], op["size"], float(op["yaw"])
        if min(size) <= 0:
            raise EditError("a box needs a positive size")
        old_c, old_s, old_yaw = record["centre"], record["size"], record["yaw"]
        c0, s0 = math.cos(old_yaw), math.sin(old_yaw)
        c1, s1 = math.cos(yaw), math.sin(yaw)
        moved = set()
        # Each voxel keeps its place in the box, so association still finds the object there.
        for key in record["_voxels"]:
            x, y, z = centre_of(key, self.voxel)
            u = (x - old_c[0]) * c0 + (y - old_c[1]) * s0
            v = -(x - old_c[0]) * s0 + (y - old_c[1]) * c0
            u *= size[0] / old_s[0] if old_s[0] > 0 else 1.0
            v *= size[1] / old_s[1] if old_s[1] > 0 else 1.0
            moved.add(
                key_of(centre[0] + c1 * u - s1 * v, centre[1] + s1 * u + c1 * v, z, self.voxel)
            )
        record["_voxels"] = moved
        record.update(
            centre=list(centre), size=[size[0], size[1], old_s[2]], yaw=yaw, box_pinned=True
        )

    def _op_split(self, op):
        parent = self.objects[self._object(op["id"])]
        label = str(op.get("label", "")).strip().lower()
        if not label:
            raise EditError("a split needs a label for the part")
        polygon = op["polygon"]
        part = {k for k in parent["_voxels"] if inside(polygon, *centre_of(k, self.voxel)[:2])}
        if not part:
            raise EditError("the drawn area holds none of its voxels")
        if part == parent["_voxels"]:
            raise EditError("the drawn area holds all of it: relabel it instead")
        new_id = self._next_id()
        child = copy.deepcopy({k: v for k, v in parent.items() if k != "_voxels"})
        child.update(
            id=new_id,
            label=label,
            operator_label=label,
            votes={label: 1.0},
            name=str(op.get("name", "")).strip(),
            caption="",
            box_pinned=False,
            checked=False,
            _voxels=part,
        )
        parent["_voxels"] = parent["_voxels"] - part
        parent["box_pinned"] = False
        self._refit(parent)
        self._refit(child)
        self.objects[new_id] = child
        return [new_id]

    def _op_add(self, op):
        label = str(op.get("label", "")).strip().lower()
        if not label:
            raise EditError("a new object needs a label")
        centre, size, yaw = op["centre"], op["size"], float(op.get("yaw", 0.0))
        z_min, z_max = float(op.get("z_min", 0.0)), float(op.get("z_max", 0.8))
        if min(size) <= 0 or z_max <= z_min:
            raise EditError("a new object needs a positive size and height")
        new_id = self._next_id()
        now = time.time()
        self.objects[new_id] = {
            "id": new_id,
            "label": label,
            "votes": {label: 1.0},
            "name": str(op.get("name", "")).strip(),
            "caption": "",
            "centre": list(centre),
            "size": [size[0], size[1], z_max - z_min],
            "yaw": yaw,
            # Confirmed as far as canopy is concerned: an operator saw it.
            "observations": 2,
            "first_seen": now,
            "last_seen": now,
            "misses": 0,
            "top_seen": True,
            "state": "active",
            "best_view_score": 0.0,
            "operator_label": label,
            "box_pinned": True,
            "checked": True,
            "_voxels": shell(centre, size, yaw, z_min, z_max, self.voxel),
            "_embedding": [],
            "_embedded": 0,
        }
        return [new_id]

    def _op_room_name(self, op):
        self.rooms[self._room(op["room"])]["name"] = str(op.get("name", "")).strip()

    def _op_room_type(self, op):
        room = self.rooms[self._room(op["room"])]
        room.update(
            type=str(op.get("type", "")).strip().lower(),
            type_confidence=1.0,
            type_source="operator",
        )

    def _op_room_check(self, op):
        self.rooms[self._room(op["room"])]["checked"] = bool(op.get("checked", True))

    # -- what the page shows --

    def view(self):
        with self.lock:
            objects = [self._object_view(r) for r in self.objects.values()]
            rooms = [
                {
                    **room,
                    "outline": room.get("outline") or [],
                    "checked": room.get("checked", False),
                }
                for room in self.rooms.values()
            ]
            return {
                "directory": str(self.directory),
                "map": {k: v for k, v in self.map.items() if k != "image"},
                "voxel": self.voxel,
                "rooms": rooms,
                "objects": objects,
                "suggestions": self.suggestions(),
                "labels": sorted(self.labels | {o["label"] for o in objects}),
                "room_types": self.room_types,
                "can_undo": bool(self.history),
                "can_redo": bool(self.future),
                "unsaved": len(self.edits),
            }

    def _object_view(self, record):
        z_low, z_high = z_range(record["_voxels"], self.voxel) if record["_voxels"] else (0.0, 0.0)
        votes = record.get("votes") or {}
        return {
            "id": record["id"],
            "label": record.get("operator_label") or record.get("label", ""),
            "operator_label": record.get("operator_label", ""),
            "voted_label": max(votes, key=votes.get) if votes else "",
            "name": record.get("name", ""),
            "caption": record.get("caption", ""),
            "votes": dict(sorted(votes.items(), key=lambda kv: -kv[1])),
            "weight": sum(votes.values()),
            "observations": record.get("observations", 0),
            "state": record.get("state", "active"),
            "centre": record["centre"][:2],
            "size": record["size"][:2],
            "yaw": record["yaw"],
            "z_min": z_low,
            "z_max": z_high,
            "voxels": len(record["_voxels"]),
            "box_pinned": record.get("box_pinned", False),
            "checked": record.get("checked", False),
            "crop": (self.directory / "crops" / f"O{record['id']}.jpg").exists(),
            # What canopy shows: removed objects and one-off glimpses stay in the file unseen.
            "shown": record.get("state") != "removed" and record.get("observations", 0) >= 2,
        }

    def suggestions(self):
        """What deserves a second look, most doubtful first; nothing checked, nothing removed."""
        out = []
        live = [
            r
            for r in self.objects.values()
            if r.get("state") != "removed"
            and r.get("observations", 0) >= 2
            and not r.get("checked")
            and r["_voxels"]
        ]
        weight = {r["id"]: sum((r.get("votes") or {}).values()) for r in live}
        ranges = {r["id"]: z_range(r["_voxels"], self.voxel) for r in live}
        for record in live:
            reasons = []
            object_id = record["id"]
            label = record.get("operator_label") or record.get("label", "")
            name = (record.get("name") or "").lower()
            votes = record.get("votes") or {}
            if weight[object_id] < WEAK_WEIGHT and record.get("observations", 0) <= WEAK_SIGHTINGS:
                reasons.append(
                    f"seen {record.get('observations', 0)} times with little confidence: a phantom?"
                )
            if ranges[object_id][1] < FLOOR_TOP and (
                label not in FLAT_LABELS or name.endswith("floor")
            ):
                reasons.append("flat on the floor: the floor itself?")
            if not record.get("operator_label"):
                if name and not self._agrees(label, name):
                    reasons.append(f"the describer called it '{record['name']}'")
                if votes and max(votes.values()) < SPLIT_SHARE * sum(votes.values()):
                    top = sorted(votes.items(), key=lambda kv: -kv[1])[:3]
                    reasons.append(
                        "the detector was split: "
                        + ", ".join(f"{label} {votes:.0f}" for label, votes in top)
                    )
            if ranges[object_id][0] < FLOOR_STANDING:
                for other in live:
                    if (
                        other is not record
                        and ranges[other["id"]][0] < FLOOR_STANDING
                        and weight[object_id] * PIECE_RATIO <= weight[other["id"]]
                        and math.dist(record["centre"][:2], other["centre"][:2])
                        < (math.hypot(*record["size"][:2]) + math.hypot(*other["size"][:2])) / 2
                        and overlap(record["_voxels"], other["_voxels"]) >= PIECE_SHARE
                    ):
                        reasons.append(f"mostly inside O{other['id']} ({other['label']})")
                        break
            if reasons:
                out.append({"kind": "object", "id": object_id, "reasons": reasons})
        for room in self.rooms.values():
            if room.get("checked") or room.get("type_source") == "operator":
                continue
            if (
                room.get("type_source") == "objects"
                and room.get("type_confidence", 0.0) < LOW_ROOM_CONFIDENCE
            ):
                out.append(
                    {
                        "kind": "room",
                        "id": room["id"],
                        "reasons": [
                            f"typed {room.get('type')} from its objects at "
                            f"{room.get('type_confidence', 0.0):.2f}"
                        ],
                    }
                )
            elif not room.get("type"):
                out.append({"kind": "room", "id": room["id"], "reasons": ["no type yet"]})
        out.sort(key=lambda s: -len(s["reasons"]))
        return out

    def _agrees(self, label, name):
        """Whether a describer's name is the label or one of its synonyms, word for word."""
        canonical = self.synonyms.get(label, label)
        words = set(re.findall(r"[a-z]+", name))
        forms = {label, canonical} | {w for w, c in self.synonyms.items() if c == canonical}
        return any(set(form.split()) <= words for form in forms) or canonical in {
            self.synonyms.get(word, word) for word in words
        }


def read_config(config):
    """Known labels, synonyms (other word -> the table's) and room types, from canopy's config and
    the detector's word list."""
    labels, synonyms, types = set(), {}, []
    if VOCABULARY.exists():
        with open(VOCABULARY) as stream:
            body = yaml.safe_load(stream) or {}
        for node in body.values():
            labels.update((node or {}).get("ros__parameters", {}).get("phrases", []))
    table = config / "room_types.yaml"
    if table.exists():
        with open(table) as stream:
            body = yaml.safe_load(stream) or {}
        for room_type, entry in (body.get("room_types") or {}).items():
            types.append(room_type)
            labels.update((entry or {}).get("objects", {}))
        for word, others in (body.get("synonyms") or {}).items():
            for other in others:
                synonyms[other.lower()] = word
    return labels, synonyms, sorted(set(types) | {"hallway", "study", "bathroom"})


def read_map(directory):
    """map.yaml and map.pgm: resolution, origin, size and the PNG the page draws."""
    with open(directory / "map.yaml") as stream:
        meta = yaml.safe_load(stream)
    data = (directory / meta.get("image", "map.pgm")).read_bytes()
    fields = []
    at = 0
    while len(fields) < 4:
        # P5, width, height and maxval, whitespace separated, comments skipped.
        while data[at : at + 1].isspace():
            at += 1
        if data[at : at + 1] == b"#":
            at = data.index(b"\n", at) + 1
            continue
        end = at
        while not data[end : end + 1].isspace():
            end += 1
        fields.append(data[at:end])
        at = end
    if fields[0] != b"P5":
        raise ValueError("map image is not a binary PGM")
    width, height = int(fields[1]), int(fields[2])
    pixels = data[at + 1 : at + 1 + width * height]
    return {
        "resolution": float(meta["resolution"]),
        "origin": [float(v) for v in meta["origin"][:2]],
        "width": width,
        "height": height,
        "image": png(width, height, pixels),
    }


def png(width, height, grey):
    """An 8-bit greyscale PNG, with the standard library alone."""

    def chunk(kind, body):
        return (
            struct.pack(">I", len(body))
            + kind
            + body
            + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
        )

    rows = b"".join(b"\x00" + grey[y * width : (y + 1) * width] for y in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows, 6))
        + chunk(b"IEND", b"")
    )


def write_atomically(path, data):
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
        # Owned and moded like the file it replaces, so canopy can overwrite it later.
        if path.exists():
            info = path.stat()
            os.chmod(temporary, info.st_mode & 0o7777)
            if hasattr(os, "chown"):
                try:
                    os.chown(temporary, info.st_uid, info.st_gid)
                except PermissionError:
                    pass
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


# --- HTTP -----------------------------------------------------------------------------------


STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}
LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def handler_for(world):
    class Handler(BaseHTTPRequestHandler):
        server_version = "canopy-editor"

        def log_message(self, fmt, *args):
            sys.stderr.write("%s\n" % (fmt % args))

        def _trusted(self):
            """Whether a request can be the page's own. Bound to loopback, only a loopback Host is
            (a rebound DNS name is not); a POST must be JSON from this origin, which no other web
            page can send without a preflight this server never answers."""
            host = self.headers.get("Host") or ""
            if self.server.loopback and host.rsplit(":", 1)[0].strip("[]") not in LOOPBACK:
                return False
            if self.command != "POST":
                return True
            origin = self.headers.get("Origin")
            kind = (self.headers.get("Content-Type") or "").split(";")[0].strip()
            return kind == "application/json" and origin in (None, f"http://{host}")

        def do_GET(self):
            if not self._trusted():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "not from this editor"})
            if self.path in STATIC:
                name, kind = STATIC[self.path]
                return self._send(HTTPStatus.OK, (HERE / "static" / name).read_bytes(), kind)
            if self.path == "/api/world":
                return self._json(HTTPStatus.OK, world.view())
            if self.path == "/api/map.png":
                return self._send(HTTPStatus.OK, world.map["image"], "image/png")
            match = re.fullmatch(r"/api/crops/O(\d+)\.jpg", self.path)
            if match:
                crop = world.directory / "crops" / f"O{match.group(1)}.jpg"
                if crop.exists():
                    return self._send(HTTPStatus.OK, crop.read_bytes(), "image/jpeg")
            return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self):
            if not self._trusted():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "not from this editor"})
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/edit":
                    created = world.apply(body)
                    return self._json(HTTPStatus.OK, {"created": created, "world": world.view()})
                if self.path == "/api/undo":
                    world.undo()
                elif self.path == "/api/redo":
                    world.redo()
                elif self.path == "/api/save":
                    message = world.save()
                    return self._json(HTTPStatus.OK, {"message": message, "world": world.view()})
                elif self.path == "/api/reload":
                    with world.lock:
                        world.load()
                else:
                    return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return self._json(HTTPStatus.OK, {"world": world.view()})
            except Conflict as error:
                return self._json(HTTPStatus.CONFLICT, {"error": str(error)})
            except (EditError, KeyError, TypeError, ValueError) as error:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def _json(self, status, body):
            self._send(status, json.dumps(body).encode(), "application/json")

        def _send(self, status, body, kind):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve(world, host="127.0.0.1", port=8765):
    server = ThreadingHTTPServer((host, port), handler_for(world))
    server.loopback = host in LOOPBACK
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("world", help="a saved world directory (world.yaml, objects.bin, map)")
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to edit from a tablet")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--config", default=str(CONFIG), help="canopy's config, for labels")
    args = parser.parse_args()
    world = World(args.world, args.config)
    server = serve(world, args.host, args.port)
    print(f"editing {args.world} at http://{args.host}:{server.server_address[1]}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    if world.edits:
        print(f"{len(world.edits)} edits were not saved", file=sys.stderr)


if __name__ == "__main__":
    main()
