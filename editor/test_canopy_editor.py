"""The map editor against canopy's example world: file round trip, every edit and its undo, the
review queue, the save guard and the HTTP API.

    python3 -m unittest editor/test_canopy_editor.py
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canopy_editor as editor  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[1] / "canopy" / "doc" / "apartment"


class EditorTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="canopy_editor_test_"))
        for name in ("world.yaml", "objects.bin", "map.pgm", "map.yaml"):
            shutil.copy(EXAMPLE / name, self.directory / name)
        self.world = editor.World(self.directory)

    def tearDown(self):
        shutil.rmtree(self.directory)

    def reopened(self):
        self.world.save()
        return editor.World(self.directory)

    def an_object(self, label=None):
        for record in self.world.objects.values():
            if label is None or record["label"] == label:
                return record
        self.fail(f"no {label} in the example world")

    def add(self, label, **fields):
        op = {"op": "add", "label": label, "centre": [5.0, 5.0], "size": [1.2, 0.6], **fields}
        return self.world.apply(op)[0]

    def test_keeps_every_object_and_voxel_across_a_save(self):
        before = {i: set(r["_voxels"]) for i, r in self.world.objects.items()}
        again = self.reopened()
        self.assertEqual({i: set(r["_voxels"]) for i, r in again.objects.items()}, before)
        self.assertEqual(again.rooms.keys(), self.world.rooms.keys())
        self.assertEqual(again.map["width"], self.world.map["width"])

    def test_keys_round_trip_the_way_canopy_makes_them(self):
        key = editor.key_of(-3.21, 7.5, 0.83, 0.04)
        x, y, z = editor.centre_of(key, 0.04)
        self.assertLess(max(abs(x + 3.21), abs(y - 7.5), abs(z - 0.83)), 0.04)

    def test_relabels_renames_and_checks_an_object(self):
        record = self.an_object("bowl")
        self.world.apply({"op": "label", "id": record["id"], "label": "Dining Table"})
        self.world.apply({"op": "name", "id": record["id"], "name": "the long table"})
        self.world.apply({"op": "check", "id": record["id"]})
        again = self.reopened().objects[record["id"]]
        self.assertEqual(again["operator_label"], "dining table")
        self.assertEqual(again["label"], "dining table")
        self.assertEqual(again["name"], "the long table")
        self.assertTrue(again["operator_named"])  # The describer leaves it alone.
        self.assertTrue(again["checked"])
        self.assertIn("bowl", again["votes"])  # The votes are kept underneath.
        self.world.apply({"op": "name", "id": record["id"], "name": ""})
        self.assertFalse(self.world.objects[record["id"]]["operator_named"])

    def test_takes_a_null_label_for_none(self):
        record = self.an_object("bowl")
        self.world.apply({"op": "label", "id": record["id"], "label": None})
        self.assertEqual(record["label"], editor.top_vote(record["votes"]))
        with self.assertRaises(editor.EditError):
            self.add(None)

    def test_undoes_and_redoes_a_delete(self):
        record = self.an_object()
        self.world.apply({"op": "delete", "id": record["id"]})
        self.assertNotIn(record["id"], self.world.objects)
        self.world.undo()
        self.assertEqual(self.world.objects[record["id"]]["_voxels"], record["_voxels"])
        self.world.redo()
        self.assertNotIn(record["id"], self.world.objects)

    def test_merges_two_objects_into_one(self):
        a, b = [r for r in self.world.objects.values() if r["label"] == "chair"][:2]
        union = a["_voxels"] | b["_voxels"]
        first = a["observations"]
        sightings = first + b["observations"]
        # An id twice is one object, not two sets of its votes and sightings.
        self.world.apply({"op": "merge", "ids": [a["id"], b["id"], b["id"]], "into": a["id"]})
        merged = self.world.objects[a["id"]]
        self.assertNotIn(b["id"], self.world.objects)
        self.assertEqual(merged["_voxels"], union)
        self.assertEqual(merged["observations"], sightings)
        self.world.undo()
        self.assertIn(b["id"], self.world.objects)
        self.assertEqual(self.world.objects[a["id"]]["observations"], first)

    def test_leaves_the_world_as_it_was_when_an_edit_fails(self):
        a, b = [r for r in self.world.objects.values() if r["label"] == "chair"][:2]
        voxels, count = set(a["_voxels"]), len(self.world.objects)
        with (
            mock.patch.object(editor.World, "_refit", side_effect=editor.EditError("no")),
            self.assertRaises(editor.EditError),
        ):
            self.world.apply({"op": "merge", "ids": [a["id"], b["id"]], "into": a["id"]})
        self.assertEqual(len(self.world.objects), count)
        self.assertEqual(self.world.objects[a["id"]]["_voxels"], voxels)
        self.assertFalse(self.world.edits)

    def test_splits_a_part_out_by_a_drawn_area(self):
        record = max(self.world.objects.values(), key=lambda r: len(r["_voxels"]))
        cx, cy = record["centre"][:2]
        # The half of it on one side of its centre.
        polygon = [[cx - 5, cy - 5], [cx, cy - 5], [cx, cy + 5], [cx - 5, cy + 5]]
        total = len(record["_voxels"])
        created = self.world.apply(
            {"op": "split", "id": record["id"], "polygon": polygon, "label": "chair"}
        )
        part = self.world.objects[created[0]]
        self.assertEqual(part["operator_label"], "chair")
        self.assertEqual((part["_embedding"], part["_embedded"]), ([], 0))  # Not the parent's look.
        # Labelled by hand too, so canopy's merge pass leaves the two apart.
        self.assertEqual(self.world.objects[record["id"]]["operator_label"], record["label"])
        self.assertEqual(len(part["_voxels"]) + len(record["_voxels"]), total)
        self.assertTrue(all(editor.centre_of(k, 0.04)[0] < cx for k in part["_voxels"]))
        self.world.undo()
        self.assertNotIn(created[0], self.world.objects)
        self.assertEqual(len(self.world.objects[record["id"]]["_voxels"]), total)

    def test_moves_a_box_and_its_voxels_together(self):
        record = self.an_object("sofa")
        centre = [record["centre"][0] + 1.0, record["centre"][1]]
        self.world.apply(
            {
                "op": "box",
                "id": record["id"],
                "centre": centre,
                "size": record["size"][:2],
                "yaw": record["yaw"],
            }
        )
        moved = self.reopened().objects[record["id"]]
        self.assertTrue(moved["box_pinned"])
        self.assertAlmostEqual(moved["centre"][0], centre[0])
        xs = [editor.centre_of(k, 0.04)[0] for k in moved["_voxels"]]
        self.assertAlmostEqual(sum(xs) / len(xs), centre[0], delta=0.3)

    def test_refuses_numbers_that_are_none(self):
        record = self.an_object("sofa")
        box = {
            "op": "box",
            "id": record["id"],
            "centre": [1.0, 1.0],
            "size": [1.0, 1.0],
            "yaw": 0.0,
        }
        for bad in (
            {"yaw": float("nan")},
            {"centre": [True, 1.0]},
            {"size": [1.0, float("inf")]},
            {"centre": "1,2"},
        ):
            with self.assertRaises(editor.EditError, msg=bad):
                self.world.apply({**box, **bad})

    def test_adds_a_missed_object_as_a_confirmed_shell(self):
        created = self.add("wardrobe", yaw=0.0, z_min=0.0, z_max=2.0)
        added = self.reopened().objects[created]
        self.assertEqual(added["operator_label"], "wardrobe")
        self.assertGreaterEqual(added["observations"], 2)
        low, high = editor.z_range(added["_voxels"], 0.04)
        self.assertLess(low, 0.1)
        self.assertGreater(high, 1.9)

    def test_never_hands_out_an_id_twice(self):
        highest = max(self.world.objects)
        self.world.apply({"op": "delete", "id": highest})
        created = self.add("lamp")
        self.assertGreater(created, highest)
        self.assertGreater(self.reopened().next_object, created)

    def test_types_and_names_a_room_for_good(self):
        room = next(iter(self.world.rooms))
        self.world.apply({"op": "room_type", "room": room, "type": "Storage Room"})
        self.world.apply({"op": "room_name", "room": room, "name": "the store"})
        again = self.reopened().rooms[room]
        self.assertEqual(again["type"], "storage room")
        self.assertEqual(again["type_source"], "operator")
        self.assertEqual(again["name"], "the store")
        # Cleared, it is canopy's typer's again.
        self.world.apply({"op": "room_type", "room": room, "type": ""})
        cleared = self.world.rooms[room]
        self.assertEqual((cleared["type_source"], cleared["type_confidence"]), ("", 0.0))

    def test_flags_the_floor_and_takes_checked_things_off_the_list(self):
        slab = self.add("table", z_min=0.0, z_max=0.08)
        self.world.apply({"op": "check", "id": slab, "checked": False})
        flagged = {s["id"]: s for s in self.world.suggestions() if s["kind"] == "object"}
        self.assertIn("flat on the floor: the floor itself?", flagged[slab]["reasons"])
        self.world.apply({"op": "check", "id": slab})
        self.assertNotIn(slab, {s["id"] for s in self.world.suggestions()})

    def test_refuses_to_save_over_a_newer_world(self):
        self.world.apply({"op": "check", "id": self.an_object()["id"]})
        stamp = os.stat(self.directory / "world.yaml").st_mtime_ns
        os.utime(self.directory / "world.yaml", ns=(stamp + 10**9, stamp + 10**9))
        with self.assertRaises(editor.Conflict):
            self.world.save()

    def test_makes_its_edits_again_on_a_world_canopy_saved_since(self):
        sofa = self.an_object("sofa")["id"]
        self.world.apply({"op": "label", "id": sofa, "label": "couch"})
        wardrobe = self.add("wardrobe")
        self.world.apply({"op": "name", "id": wardrobe, "name": "the tall one"})
        self.world.undo()
        self.world.redo()
        # canopy saves meanwhile, renaming a room and handing out the id the wardrobe took.
        canopy = editor.World(self.directory)
        room = next(iter(canopy.rooms))
        canopy.apply({"op": "room_name", "room": room, "name": "renamed by canopy"})
        lamp = canopy.apply({"op": "add", "label": "lamp", "centre": [1, 1], "size": [0.3, 0.3]})[0]
        canopy.save()
        with self.assertRaises(editor.Conflict):
            self.world.save()
        self.assertIn("made 3 of 3", self.world.rebase())
        again = self.reopened()
        self.assertEqual(again.rooms[room]["name"], "renamed by canopy")
        self.assertEqual(again.objects[sofa]["operator_label"], "couch")
        self.assertEqual(again.objects[lamp]["label"], "lamp")
        [made] = [r for r in again.objects.values() if r["label"] == "wardrobe"]
        self.assertNotEqual(made["id"], lamp)
        self.assertEqual(made["name"], "the tall one")

    def test_refuses_a_file_that_is_no_canopy_world(self):
        data = bytearray((self.directory / "objects.bin").read_bytes())
        data[0] ^= 0xFF
        (self.directory / "objects.bin").write_bytes(bytes(data))
        with self.assertRaises(ValueError):
            editor.World(self.directory)

    def test_refuses_an_unknown_edit(self):
        with self.assertRaises(editor.EditError):
            self.world.apply({"op": "paint"})
        with self.assertRaises(editor.EditError):
            self.world.apply({"op": "delete", "id": 99999})

    def serving(self, host):
        server = editor.serve(self.world, host, 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server, f"http://127.0.0.1:{server.server_address[1]}"

    def assertRefused(self, request, code):
        with self.assertRaises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(request)
        self.assertEqual(refused.exception.code, code)
        return json.load(refused.exception)

    def test_serves_the_page_and_the_api(self):
        _, base = self.serving("127.0.0.1")
        with urllib.request.urlopen(base + "/") as page:
            self.assertIn(b"<canvas", page.read())
        with urllib.request.urlopen(base + "/api/map.png") as image:
            self.assertEqual(image.read(8), b"\x89PNG\r\n\x1a\n")
        record = self.an_object()
        json_type = {"Content-Type": "application/json"}
        request = urllib.request.Request(
            base + "/api/edit",
            data=json.dumps({"op": "label", "id": record["id"], "label": "lamp"}).encode(),
            headers=json_type,
        )
        with urllib.request.urlopen(request) as answer:
            body = json.load(answer)
        shown = {o["id"]: o for o in body["world"]["objects"]}
        self.assertEqual(shown[record["id"]]["label"], "lamp")
        self.assertEqual(body["world"]["unsaved"], 1)
        bad = urllib.request.Request(base + "/api/edit", data=b'{"op": "paint"}', headers=json_type)
        self.assertRefused(bad, 400)
        # What another web page could send: a plain form post, or JSON from its own origin.
        for headers in (
            {"Content-Type": "text/plain"},
            {**json_type, "Origin": "http://evil.example"},
        ):
            forged = urllib.request.Request(
                base + "/api/save", data=b"{}", headers=headers, method="POST"
            )
            self.assertRefused(forged, 403)
        rebound = urllib.request.Request(base + "/api/world", headers={"Host": "evil.example"})
        self.assertRefused(rebound, 403)
        # Whatever breaks is answered, not a dropped connection.
        with (
            mock.patch.object(self.world, "view", side_effect=RuntimeError("broken")),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            answer = self.assertRefused(base + "/api/world", 500)
        self.assertEqual(answer["error"], "RuntimeError: broken")


if __name__ == "__main__":
    unittest.main()
