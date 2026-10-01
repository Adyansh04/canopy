#!/usr/bin/env python3
"""Runs the segmenter against a stub vision server; no simulator or GPU.

Covers what an agent relies on: masks for the named camera's newest frame, stamped with that
frame, and a plain answer when the camera or the prompt is wrong.
"""

import os
import sys
import unittest

import launch_testing
import pytest
import rclpy
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from rclpy.node import Node as RclpyNode
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from canopy_msgs.srv import Segment

PORT = 5592
STUB = os.path.join(os.path.dirname(__file__), "segment_server_stub.py")
TOPIC = "/test_chest/color/image_raw"


@pytest.mark.launch_test
def generate_test_description():
    segmenter = Node(
        package="canopy_perception",
        executable="segmenter",
        name="segmenter",
        output="screen",
        parameters=[
            {
                "server_address": f"tcp://127.0.0.1:{PORT}",
                "zmq_timeout_ms": 5000,
                "cameras": [f"chest={TOPIC}"],
                "max_image_age_s": 5.0,
            }
        ],
    )
    return LaunchDescription(
        [
            ExecuteProcess(cmd=[sys.executable, STUB, str(PORT)], output="screen"),
            TimerAction(period=3.0, actions=[segmenter]),
            launch_testing.actions.ReadyToTest(),
        ]
    )


class TestSegmenter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = RclpyNode("segmenter_probe")
        cls.camera = cls.node.create_publisher(Image, TOPIC, qos_profile_sensor_data)
        cls.client = cls.node.create_client(Segment, "/segmenter/segment")
        cls.sent = set()

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def _frame(self):
        image = Image()
        image.header.stamp = self.node.get_clock().now().to_msg()
        image.header.frame_id = "chest_camera_color_optical_frame"
        image.height, image.width = 480, 640
        image.encoding = "rgb8"
        image.step = image.width * 3
        image.data = bytes(image.height * image.step)
        self.camera.publish(image)
        self.sent.add((image.header.stamp.sec, image.header.stamp.nanosec))

    def _ask(self, camera, prompt):
        self.assertTrue(self.client.wait_for_service(timeout_sec=30.0))
        for _ in range(10):
            self._frame()
            rclpy.spin_once(self.node, timeout_sec=0.1)
        request = Segment.Request(camera=camera, prompt=prompt)
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=20.0)
        self.assertIsNotNone(future.result(), "the segmenter did not answer")
        return future.result()

    def test_masks_come_from_the_named_cameras_newest_frame(self):
        out = self._ask("chest", "red block")
        self.assertTrue(out.success, out.message)
        self.assertEqual(out.masks.model, "stub")
        self.assertEqual(len(out.masks.instances), 1)
        self.assertEqual(out.masks.instances[0].label, "red block")
        stamp = (out.masks.header.stamp.sec, out.masks.header.stamp.nanosec)
        self.assertIn(stamp, self.sent, "the masks are stamped with something other than a frame")

    def test_an_unknown_camera_or_an_empty_prompt_is_answered_plainly(self):
        out = self._ask("head", "the floor")
        self.assertFalse(out.success)
        self.assertIn("chest", out.message)
        out = self._ask("chest", "  ")
        self.assertFalse(out.success)
        self.assertIn("empty", out.message)
