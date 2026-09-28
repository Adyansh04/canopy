#!/usr/bin/env python3
"""A world edited by hand after a run, taken by a running world model; no simulator.

The node resumes canopy's example world on its own map. The map editor then relabels an object,
deletes another and retypes a room on disk: the node must refuse to save over that, take it with
~/reload, publish it, and keep it through its own next save.
"""

import os
import pathlib
import shutil
import sys
import tempfile
import time
import unittest

import launch_testing
import pytest
import rclpy
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node as RclpyNode
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger

from canopy_msgs.msg import RoomArray, WorldObjectArray

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE = os.path.dirname(HERE)
EXAMPLE = os.path.join(PACKAGE, "doc", "apartment")
sys.path.insert(0, os.path.join(os.path.dirname(PACKAGE), "editor"))
import canopy_editor as editor  # noqa: E402

WORLD_DIR = tempfile.mkdtemp(prefix="canopy_world_edit_")
for name in ("world.yaml", "objects.bin", "coverage.bin", "map.pgm", "map.yaml"):
    shutil.copy(os.path.join(EXAMPLE, name), WORLD_DIR)
LATCHED = QoSProfile(
    depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
)


@pytest.mark.launch_test
def generate_test_description():
    world_model = Node(
        package="canopy",
        executable="world_model",
        name="canopy",
        output="screen",
        parameters=[
            os.path.join(PACKAGE, "config", "canopy.yaml"),
            # Only the saves the test asks for.
            {"world_dir": WORLD_DIR, "autosave_period_s": 0.0},
        ],
    )
    return LaunchDescription([world_model, launch_testing.actions.ReadyToTest()])


def occupancy_grid(directory):
    """The saved map as map_server would serve it: PGM rows run from the top, the grid's from the
    bottom, and map_saver's greys mean free, occupied and unknown."""
    world = editor.read_map(pathlib.Path(directory))
    with open(os.path.join(directory, "map.pgm"), "rb") as stream:
        raw = stream.read()
    pixels = raw[len(raw) - world["width"] * world["height"] :]
    grid = OccupancyGrid()
    grid.header.frame_id = "map"
    grid.info.resolution = world["resolution"]
    grid.info.width = world["width"]
    grid.info.height = world["height"]
    grid.info.origin.position.x, grid.info.origin.position.y = world["origin"]
    grid.info.origin.orientation.w = 1.0
    value = {0: 100, 254: 0}
    data = []
    for row in range(world["height"] - 1, -1, -1):
        start = row * world["width"]
        data.extend(value.get(p, -1) for p in pixels[start : start + world["width"]])
    grid.data = data
    return grid


class WorldEditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = RclpyNode("world_edit_test")
        cls.objects = None
        cls.rooms = None
        cls.node.create_subscription(WorldObjectArray, "/canopy/objects", cls._objects, LATCHED)
        cls.node.create_subscription(RoomArray, "/canopy/rooms", cls._rooms, LATCHED)
        cls.map_pub = cls.node.create_publisher(OccupancyGrid, "/map", LATCHED)
        cls.map_pub.publish(occupancy_grid(WORLD_DIR))
        cls.save = cls.node.create_client(Trigger, "/canopy/save")
        cls.reload = cls.node.create_client(Trigger, "/canopy/reload")

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()
        shutil.rmtree(WORLD_DIR, ignore_errors=True)

    @classmethod
    def _objects(cls, msg):
        cls.objects = msg

    @classmethod
    def _rooms(cls, msg):
        cls.rooms = msg

    def spin_until(self, condition, timeout_s):
        deadline = time.time() + timeout_s
        while time.time() < deadline and not condition():
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return condition()

    def call(self, client):
        self.assertTrue(client.wait_for_service(timeout_sec=20.0), client.srv_name)
        future = client.call_async(Trigger.Request())
        self.assertTrue(self.spin_until(future.done, 20.0), client.srv_name)
        return future.result()

    def test_takes_an_edited_world_and_keeps_it(self):
        self.assertTrue(
            self.spin_until(lambda: self.objects is not None and self.objects.objects, 60.0),
            "the world model never resumed the example world",
        )
        world = editor.World(WORLD_DIR)
        shown = [
            o for o in world.objects.values() if o["state"] == "active" and o["observations"] >= 2
        ]
        relabelled, deleted = shown[0]["id"], shown[1]["id"]
        room = next(iter(world.rooms))
        world.apply({"op": "label", "id": relabelled, "label": "hand checked"})
        world.apply({"op": "delete", "id": deleted})
        world.apply({"op": "room_type", "room": room, "type": "study"})
        world.save()

        refused = self.call(self.save)
        self.assertFalse(refused.success, "the node saved over an edited world")
        self.assertIn("changed on disk", refused.message)

        self.assertTrue(self.call(self.reload).success)
        self.assertTrue(
            self.spin_until(
                lambda: (
                    any(o.label == "hand checked" for o in self.objects.objects)
                    and self.rooms is not None
                    and any(r.id == room and r.type == "study" for r in self.rooms.rooms)
                ),
                30.0,
            ),
            "the reloaded world was never published",
        )
        self.assertNotIn(f"O{deleted}", {o.id for o in self.objects.objects})

        self.assertTrue(self.call(self.save).success)
        with open(os.path.join(WORLD_DIR, "world.yaml")) as stream:
            saved = yaml.safe_load(stream)
        objects = {o["id"]: o for o in saved["objects"]}
        self.assertEqual(objects[relabelled]["operator_label"], "hand checked")
        self.assertNotIn(deleted, objects)
        rooms = {r["id"]: r for r in saved["rooms"]}
        self.assertEqual(rooms[room]["type_source"], "operator")
        self.assertTrue(rooms[room]["outline"])
