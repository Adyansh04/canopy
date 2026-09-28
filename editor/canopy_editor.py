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
import contextlib
import copy
import fcntl
import hmac
import json
import math
import os
import re
import secrets
import struct
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
import zlib
from http import HTTPStatus
from http.cookies import SimpleCookie
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
# objects.bin's header, as world_store.cpp writes it.
MAGIC = 0x4D573147
BIN_VERSION = 1
# Share of footprint points left off each box side, as canopy's ObjectParams::box_trim.
BOX_TRIM = 0.02
UNDO_DEPTH = 100

# What canopy's loadWorld takes for a field an older world.yaml lacks (world_store.cpp).
OBJECT_DEFAULTS = {
    "label": "",
    "votes": None,
    "name": "",
    "caption": "",
    "yaw": 0.0,
    "observations": 0,
    "first_seen": 0.0,
    "last_seen": 0.0,
    "misses": 0,
    "top_seen": False,
    "state": "active",
    "best_view_score": 0.0,
    "operator_label": "",
    "operator_named": False,
    "removed_by": "",
    "box_pinned": False,
    "checked": False,
}
OBJECT_TEXT = ("label", "name", "caption", "state", "operator_label", "removed_by")
ROOM_DEFAULTS = {
    "name": "",
    "type": "",
    "type_confidence": 0.0,
    "type_source": "",
    "x": 0.0,
    "y": 0.0,
    "checked": False,
}
ROOM_TEXT = ("id", "name", "type", "type_source")

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


def text(value):
    """A free-text field as text: YAML reads an unquoted 3 as a number, and JSON has null."""
    return "" if value is None else str(value)


def number(value, what):
    """A finite number from an edit; JSON's true is no number, and Python's JSON reads NaN."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise EditError(f"{what} must be a finite number")
    return float(value)


def pair(value, what):
    if not isinstance(value, list) or len(value) != 2:
        raise EditError(f"{what} must be two numbers")
    return [number(v, what) for v in value]


def top_vote(votes):
    return max(votes, key=votes.get) if votes else ""


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
    sample = keys[:: -(-len(keys) // 600)]
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


@contextlib.contextmanager
def world_lock(directory, exclusive):
    """canopy's WorldLock on world_dir/.lock: no save interleaves with another save or a load,
    whichever program makes it."""
    handle = None
    # Read-only when another user made the file: flock needs no write access.
    for flags in (os.O_RDWR | os.O_CREAT, os.O_RDONLY):
        with contextlib.suppress(OSError):
            handle = os.open(directory / ".lock", flags, 0o666)
            break
    try:
        if handle is not None:
            fcntl.flock(handle, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        if handle is not None:
            os.close(handle)


def stamp_of(path):
    """Which save of world.yaml is on disk: canopy replaces the file, so a save within one mtime
    tick still has a new inode."""
    info = os.stat(path)
    return info.st_ino, info.st_mtime_ns


def standing(edits):
    """The edits that undo and redo left standing, in order."""
    done, undone = [], []
    for entry in edits:
        if entry["op"] == "undo":
            undone.append(done.pop())
        elif entry["op"] == "redo":
            done.append(undone.pop())
        else:
            done.append(entry)
            undone.clear()
    return done


class World:
    """A canopy world directory: world.yaml, objects.bin, the map and the crops."""

    def __init__(self, directory, config=CONFIG):
        self.directory = Path(directory)
        self.lock = threading.RLock()
        self.history = []  # (before, after) images of what each edit touched, since the save.
        self.future = []
        self.edits = []  # This session's ops since the save, appended to edits.log by it.
        self.labels, self.synonyms, self.room_types, self.default_voxel = read_config(Path(config))
        self.load()

    # -- loading and saving --

    def load(self):
        """Reads the world as it is on disk; nothing here changes unless all of it reads."""
        with world_lock(self.directory, exclusive=False):
            stamp = stamp_of(self.directory / "world.yaml")
            with open(self.directory / "world.yaml") as stream:
                body = yaml.safe_load(stream)
            data = (self.directory / "objects.bin").read_bytes()
        if not isinstance(body, dict) or body.get("version") != 1:
            raise ValueError("world.yaml is not a version 1 canopy world")
        magic, version, count = struct.unpack_from("<III", data, 0)
        if magic != MAGIC or version != BIN_VERSION:
            raise ValueError("objects.bin is not a version 1 canopy object file")
        records = {}
        for entry in body.get("objects") or []:
            record = {**OBJECT_DEFAULTS, **entry}
            for key in OBJECT_TEXT:
                record[key] = text(record[key])
            record["votes"] = {text(k): float(v) for k, v in (record["votes"] or {}).items()}
            record["id"] = int(record["id"])
            records[record["id"]] = record
        objects = {}
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
            record.update(_voxels=set(keys), _embedding=embedding, _embedded=embedded)
            objects[object_id] = record
        rooms = {}
        for entry in body.get("rooms") or []:
            room = {**ROOM_DEFAULTS, **entry}
            for key in ROOM_TEXT:
                room[key] = text(room[key])
            rooms[room["id"]] = room
        world_map = read_map(self.directory)
        with self.lock:
            self.stamp, self.yaml, self.map = stamp, body, world_map
            self.voxel = float(body.get("voxel") or self.default_voxel)
            self.rooms, self.objects = rooms, objects
            # Past every id canopy handed out, torn ones too: crops and edits.log name objects by
            # id, so none is used twice.
            self.next_object = max(int(body.get("next_object") or 1), max(records, default=0) + 1)
            self.history.clear()
            self.future.clear()
            self.edits.clear()

    def reload(self):
        self.load()
        return "read the world from disk again"

    def save(self):
        with self.lock, world_lock(self.directory, exclusive=True):
            if stamp_of(self.directory / "world.yaml") != self.stamp:
                raise Conflict(
                    "canopy saved the world since the editor read it: rebase to make your edits "
                    "again on its version, or reload to drop them."
                )
            body = {**self.yaml, "next_object": self.next_object}
            body["rooms"] = list(self.rooms.values())
            body["objects"] = [
                {k: v for k, v in record.items() if not k.startswith("_")}
                for record in self.objects.values()
            ]
            blob = [struct.pack("<III", MAGIC, BIN_VERSION, len(self.objects))]
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
            content = yaml.safe_dump(body, sort_keys=False, default_flow_style=None, width=100)
            write_atomically(self.directory / "world.yaml", content.encode())
            if self.edits:
                with open(self.directory / "edits.log", "a") as log:
                    for edit in self.edits:
                        log.write(json.dumps(edit) + "\n")
            self.stamp = stamp_of(self.directory / "world.yaml")
            saved = len(self.edits)
            # Undo reaches back to the save: what is on disk is what a rebase starts from.
            self.edits.clear()
            self.history.clear()
            self.future.clear()
            return f"saved {saved} edits to {self.directory}"

    def rebase(self):
        """Reads the world canopy saved since, and makes this session's edits again on it."""
        with self.lock:
            ops = standing(self.edits)
            self.load()
            remade = {}  # An object an edit made then, by the one it makes now.
            failed = []
            for entry in ops:
                op = {k: v for k, v in entry.items() if k not in ("time", "created")}
                for key in ("id", "into"):
                    if key in op:
                        op[key] = remade.get(op[key], op[key])
                if isinstance(op.get("ids"), list):
                    op["ids"] = [remade.get(i, i) for i in op["ids"]]
                try:
                    created = self.apply(op)
                except (EditError, KeyError, TypeError, ValueError) as error:
                    failed.append(f"{op['op']}: {error}")
                    continue
                remade.update(zip(entry.get("created", []), created, strict=False))
            message = f"made {len(ops) - len(failed)} of {len(ops)} edits again on canopy's world"
            return message + (f"; not {', '.join(failed)}" if failed else "")

    # -- edits --

    def apply(self, op):
        with self.lock:
            if not isinstance(op, dict):
                raise EditError("an edit is a JSON object")
            name = op.get("op")
            handler = getattr(self, f"_op_{name}", None) if isinstance(name, str) else None
            if handler is None:
                raise EditError(f"unknown op {name!r}")
            objects, rooms = self._touched(op)
            before = self._image(objects, rooms)
            known = set(self.objects)
            try:
                created = handler(op) or []
            except BaseException:
                # Half an edit is worse than none: put back what it touched, drop what it made.
                self._restore(before)
                for object_id in set(self.objects) - known:
                    del self.objects[object_id]
                raise
            # What an edit made did not exist before it: undo takes it away again.
            before["objects"].update(dict.fromkeys(created))
            after = self._image(objects | set(created), rooms)
            self.history.append((before, after))
            del self.history[:-UNDO_DEPTH]
            self.future.clear()
            entry = {"time": time.time(), **op}
            if created:
                entry["created"] = created  # For a rebase, whose made objects get new ids.
            self.edits.append(entry)
            return created

    def undo(self):
        with self.lock:
            if not self.history:
                raise EditError("nothing to undo since the save")
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
        if isinstance(op.get("ids"), list):
            ids.update(self._object(value) for value in op["ids"])
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

    def _ids(self, op):
        if not isinstance(op.get("ids"), list) or not op["ids"]:
            raise EditError(f"{op['op']} needs ids, a list")
        return list(dict.fromkeys(self._object(value) for value in op["ids"]))

    def _room(self, value):
        if value not in self.rooms:
            raise EditError(f"no room {value}")
        return value

    def _next_id(self):
        new_id = self.next_object
        self.next_object += 1
        return new_id

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
        record["operator_label"] = text(op.get("label")).strip().lower()
        # As canopy writes it: the operator's label, or the votes' when that is cleared.
        record["label"] = record["operator_label"] or top_vote(record["votes"])

    def _op_name(self, op):
        record = self.objects[self._object(op["id"])]
        record["name"] = text(op.get("name")).strip()
        # The describer leaves a name an operator gave alone; clearing it hands it back.
        record["operator_named"] = bool(record["name"])

    def _op_check(self, op):
        self.objects[self._object(op["id"])]["checked"] = bool(op.get("checked", True))

    def _op_delete(self, op):
        del self.objects[self._object(op["id"])]

    def _op_remove(self, op):
        """Out of the map but kept in the file, with why, as canopy's clean-up leaves what it
        takes: restore brings it back."""
        reason = text(op.get("reason")).strip() or "removed by hand"
        for object_id in self._ids(op):
            self.objects[object_id].update(state="removed", removed_by=reason)

    def _op_restore(self, op):
        for object_id in self._ids(op):
            record = self.objects[object_id]
            # Vouched for: canopy's clean-up leaves a checked object be, and misses start again.
            record.update(
                state="active",
                removed_by="",
                misses=0,
                checked=True,
                observations=max(record["observations"], 2),
            )

    def _op_merge(self, op):
        into = self.objects[self._object(op["into"])]
        others = [self.objects[i] for i in self._ids(op) if i != into["id"]]
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
            # What an operator said about either survives, as in canopy's own merge.
            if not into["name"] or (other["operator_named"] and not into["operator_named"]):
                into.update(
                    name=other["name"],
                    caption=other["caption"],
                    operator_named=other["operator_named"],
                )
            into["operator_label"] = into["operator_label"] or other["operator_label"]
            into["checked"] = into["checked"] or other["checked"]
            del self.objects[other["id"]]
        if vectors and all(len(v) == len(vectors[0][0]) for v, _ in vectors):
            total = sum(n for _, n in vectors)
            mean = [sum(v[i] * n for v, n in vectors) / total for i in range(len(vectors[0][0]))]
            norm = math.sqrt(sum(x * x for x in mean)) or 1.0
            into["_embedding"] = [x / norm for x in mean]
            into["_embedded"] = total
        into["label"] = into["operator_label"] or top_vote(into["votes"])
        into.update(state="active", removed_by="", box_pinned=False)
        self._refit(into)

    def _op_box(self, op):
        record = self.objects[self._object(op["id"])]
        centre, size = pair(op.get("centre"), "centre"), pair(op.get("size"), "size")
        yaw = number(op.get("yaw"), "yaw")
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
        record.update(centre=centre, size=[size[0], size[1], old_s[2]], yaw=yaw, box_pinned=True)

    def _op_split(self, op):
        parent = self.objects[self._object(op["id"])]
        label = text(op.get("label")).strip().lower()
        if not label:
            raise EditError("a split needs a label for the part")
        polygon = op.get("polygon")
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise EditError("a split needs an area of three corners or more")
        polygon = [pair(corner, "a corner") for corner in polygon]
        part = {k for k in parent["_voxels"] if inside(polygon, *centre_of(k, self.voxel)[:2])}
        if not part:
            raise EditError("the drawn area holds none of its voxels")
        if part == parent["_voxels"]:
            raise EditError("the drawn area holds all of it: relabel it instead")
        new_id = self._next_id()
        name = text(op.get("name")).strip()
        child = copy.deepcopy({k: v for k, v in parent.items() if k != "_voxels"})
        # A different thing from here on: none of the parent's look, name or best view.
        child.update(
            id=new_id,
            label=label,
            operator_label=label,
            votes={label: 1.0},
            name=name,
            operator_named=bool(name),
            caption="",
            best_view_score=0.0,
            box_pinned=False,
            checked=False,
            _voxels=part,
            _embedding=[],
            _embedded=0,
        )
        # Both labelled by hand, or canopy's merge pass joins two touching chairs again.
        parent["operator_label"] = parent["operator_label"] or parent["label"]
        parent["_voxels"] = parent["_voxels"] - part
        parent["box_pinned"] = False
        self._refit(parent)
        self._refit(child)
        self.objects[new_id] = child
        return [new_id]

    def _op_add(self, op):
        label = text(op.get("label")).strip().lower()
        if not label:
            raise EditError("a new object needs a label")
        centre, size = pair(op.get("centre"), "centre"), pair(op.get("size"), "size")
        yaw = number(op.get("yaw", 0.0), "yaw")
        z_min = number(op.get("z_min", 0.0), "z_min")
        z_max = number(op.get("z_max", 0.8), "z_max")
        if min(size) <= 0 or z_max <= z_min:
            raise EditError("a new object needs a positive size and height")
        new_id = self._next_id()
        name = text(op.get("name")).strip()
        now = time.time()
        self.objects[new_id] = {
            **OBJECT_DEFAULTS,
            "id": new_id,
            "label": label,
            "votes": {label: 1.0},
            "name": name,
            "centre": centre,
            "size": [size[0], size[1], z_max - z_min],
            "yaw": yaw,
            # Confirmed as far as canopy is concerned: an operator saw it.
            "observations": 2,
            "first_seen": now,
            "last_seen": now,
            "top_seen": True,
            "operator_label": label,
            "operator_named": bool(name),
            "box_pinned": True,
            "checked": True,
            "_voxels": shell(centre, size, yaw, z_min, z_max, self.voxel),
            "_embedding": [],
            "_embedded": 0,
        }
        return [new_id]

    def _op_room_name(self, op):
        self.rooms[self._room(op["room"])]["name"] = text(op.get("name")).strip()

    def _op_room_type(self, op):
        room = self.rooms[self._room(op["room"])]
        kind = text(op.get("type")).strip().lower()
        # Cleared, the room goes back to canopy's typer.
        room.update(
            type=kind,
            type_confidence=1.0 if kind else 0.0,
            type_source="operator" if kind else "",
        )

    def _op_room_check(self, op):
        self.rooms[self._room(op["room"])]["checked"] = bool(op.get("checked", True))

    # -- what the page shows --

    def view(self):
        with self.lock:
            objects = [self._object_view(r) for r in self.objects.values()]
            rooms = [{**room, "outline": room.get("outline") or []} for room in self.rooms.values()]
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
        votes = record["votes"]
        return {
            "id": record["id"],
            "label": record["operator_label"] or record["label"],
            "operator_label": record["operator_label"],
            "voted_label": top_vote(votes),
            "name": record["name"],
            "caption": record["caption"],
            "votes": dict(sorted(votes.items(), key=lambda kv: -kv[1])),
            "weight": sum(votes.values()),
            "observations": record["observations"],
            "state": record["state"],
            "removed_by": record["removed_by"],
            "centre": record["centre"][:2],
            "size": record["size"][:2],
            "yaw": record["yaw"],
            "z_min": z_low,
            "z_max": z_high,
            "voxels": len(record["_voxels"]),
            "box_pinned": record["box_pinned"],
            "checked": record["checked"],
            "crop": (self.directory / "crops" / f"O{record['id']}.jpg").exists(),
            # What canopy shows: removed objects and one-off glimpses stay in the file unseen.
            "shown": record["state"] != "removed" and record["observations"] >= 2,
        }

    def suggestions(self):
        """What deserves a second look, most doubtful first; nothing checked, nothing removed.
        `phantom` marks what is likely no object at all, for removing in one go."""
        out = []
        live = [
            r
            for r in self.objects.values()
            if r["state"] != "removed"
            and r["observations"] >= 2
            and not r["checked"]
            and r["_voxels"]
        ]
        weight = {r["id"]: sum(r["votes"].values()) for r in live}
        ranges = {r["id"]: z_range(r["_voxels"], self.voxel) for r in live}
        for record in live:
            reasons = []
            phantom = False
            object_id = record["id"]
            label = record["operator_label"] or record["label"]
            name = record["name"].lower()
            votes = record["votes"]
            if weight[object_id] < WEAK_WEIGHT and record["observations"] <= WEAK_SIGHTINGS:
                reasons.append(
                    f"seen {record['observations']} times with little confidence: a phantom?"
                )
                phantom = True
            if ranges[object_id][1] < FLOOR_TOP and (
                label not in FLAT_LABELS or name.endswith("floor")
            ):
                reasons.append("flat on the floor: the floor itself?")
                phantom = True
            if not record["operator_label"]:
                if name and not self._agrees(label, name):
                    reasons.append(f"the describer called it '{record['name']}'")
                if votes and max(votes.values()) < SPLIT_SHARE * sum(votes.values()):
                    top = sorted(votes.items(), key=lambda kv: -kv[1])[:3]
                    reasons.append(
                        "the detector was split: "
                        + ", ".join(f"{word} {count:.0f}" for word, count in top)
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
                        phantom = True
                        break
            if reasons:
                out.append(
                    {"kind": "object", "id": object_id, "reasons": reasons, "phantom": phantom}
                )
        for room in self.rooms.values():
            if room["checked"] or room["type_source"] == "operator":
                continue
            if room["type_source"] == "objects" and room["type_confidence"] < LOW_ROOM_CONFIDENCE:
                out.append(
                    {
                        "kind": "room",
                        "id": room["id"],
                        "reasons": [
                            f"typed {room['type']} from its objects at {room['type_confidence']:.2f}"
                        ],
                    }
                )
            elif not room["type"]:
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
    """Known labels, synonyms (other word -> the table's), room types and the voxel size, from
    canopy's config and the detector's word list."""
    labels, synonyms, types, voxel = set(), {}, [], 0.04
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
    # For worlds saved before world.yaml carried it: the size canopy maps objects at.
    params = config / "canopy.yaml"
    if params.exists():
        with open(params) as stream:
            body = yaml.safe_load(stream) or {}
        for node in body.values():
            objects = (node or {}).get("ros__parameters", {}).get("objects") or {}
            voxel = float(objects.get("voxel", voxel))
    # A hallway is typed by its shape, not the table.
    return labels, synonyms, sorted(set(types) | {"hallway"}), voxel


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
LOOPBACK = {"127.0.0.1", "localhost"}


def handler_for(world):
    class Handler(BaseHTTPRequestHandler):
        server_version = "canopy-editor"

        def log_message(self, fmt, *args):
            sys.stderr.write("%s\n" % (fmt % args))

        def _token(self):
            """The token a request shows: the cookie the page keeps, or the printed URL's."""
            cookie = SimpleCookie(self.headers.get("Cookie") or "")
            if "canopy_token" in cookie:
                return cookie["canopy_token"].value
            query = urllib.parse.urlsplit(self.path).query
            return urllib.parse.parse_qs(query).get("token", [""])[0]

        def _trusted(self):
            """Whether a request can be the page's own. Off loopback anyone on the network can
            reach the port, so only the printed URL's token gets in. On loopback only a loopback
            Host does (a rebound DNS name does not). A POST must be JSON from this origin, which
            no other web page can send without a preflight this server never answers."""
            host = self.headers.get("Host") or ""
            if self.server.token:
                if not hmac.compare_digest(self._token().encode(), self.server.token.encode()):
                    return False
            elif host.rsplit(":", 1)[0] not in LOOPBACK:
                return False
            if self.command != "POST":
                return True
            origin = self.headers.get("Origin")
            kind = (self.headers.get("Content-Type") or "").split(";")[0].strip()
            return kind == "application/json" and origin in (None, f"http://{host}")

        def do_GET(self):
            self._answer(self._get)

        def do_POST(self):
            self._answer(self._post)

        def _answer(self, route):
            try:
                if not self._trusted():
                    return self._json(
                        HTTPStatus.FORBIDDEN, {"error": "not from this editor's printed address"}
                    )
                return route()
            except Conflict as error:
                return self._json(HTTPStatus.CONFLICT, {"error": str(error)})
            except KeyError as error:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": f"missing {error}"})
            except (EditError, TypeError, ValueError) as error:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except Exception as error:
                # Answered, so the page can say what went wrong instead of losing the connection.
                traceback.print_exc()
                message = f"{type(error).__name__}: {error}"
                return self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": message})

        def _get(self):
            url = urllib.parse.urlsplit(self.path)
            if url.path == "/" and url.query and self.server.token:
                # The printed URL: its token moves to a cookie, out of the address bar.
                self.send_response(HTTPStatus.SEE_OTHER)
                self.send_header(
                    "Set-Cookie",
                    f"canopy_token={self.server.token}; HttpOnly; SameSite=Strict; Path=/",
                )
                self.send_header("Location", "/")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            if url.path in STATIC:
                name, kind = STATIC[url.path]
                return self._send(HTTPStatus.OK, (HERE / "static" / name).read_bytes(), kind)
            if url.path == "/api/world":
                return self._json(HTTPStatus.OK, world.view())
            if url.path == "/api/map.png":
                return self._send(HTTPStatus.OK, world.map["image"], "image/png")
            match = re.fullmatch(r"/api/crops/O(\d+)\.jpg", url.path)
            if match:
                crop = world.directory / "crops" / f"O{match.group(1)}.jpg"
                if crop.exists():
                    return self._send(HTTPStatus.OK, crop.read_bytes(), "image/jpeg")
            return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def _post(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            path = urllib.parse.urlsplit(self.path).path
            if path == "/api/edit":
                created = world.apply(body)
                return self._json(HTTPStatus.OK, {"created": created, "world": world.view()})
            actions = {
                "/api/undo": world.undo,
                "/api/redo": world.redo,
                "/api/save": world.save,
                "/api/reload": world.reload,
                "/api/rebase": world.rebase,
            }
            if path not in actions:
                return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            message = actions[path]()
            return self._json(HTTPStatus.OK, {"message": message, "world": world.view()})

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
    # Bound off loopback, anyone on the network can reach it: a token keeps them out.
    server.token = None if host in LOOPBACK else secrets.token_urlsafe(16)
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("world", help="a saved world directory (world.yaml, objects.bin, map)")
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to edit from a tablet")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--config", default=str(CONFIG), help="canopy's config, for labels")
    args = parser.parse_args()
    try:
        world = World(args.world, args.config)
    except (OSError, ValueError, struct.error, yaml.YAMLError) as error:
        sys.exit(f"cannot edit {args.world}: {error}")
    server = serve(world, args.host, args.port)
    address = f"http://{args.host}:{server.server_address[1]}/"
    if server.token:
        # Open once from the device that edits (this host's address rather than 0.0.0.0).
        address += f"?token={server.token}"
    print(f"editing {args.world} at {address}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    if world.edits:
        print(f"{len(world.edits)} edits were not saved", file=sys.stderr)


if __name__ == "__main__":
    main()
