#!/usr/bin/env python3
"""Runs the detector and the segmenter against a stub segmentation service; no model server,
simulator or GPU.

Skipped where sam3_inference_msgs is not built, as in a workspace without sam3_inference.
"""

import unittest

import launch_testing
import pytest
import rclpy
from launch import LaunchDescription
from launch_ros.actions import Node
from rclpy.node import Node as RclpyNode
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from canopy_msgs.msg import InstanceMaskArray
from canopy_msgs.srv import Segment as SegmentByPrompt

try:
    from sam3_inference_msgs.msg import Segment
    from sam3_inference_msgs.srv import SegmentObjects
except ImportError:
    SegmentObjects = None

CAMERA_TOPIC = "/camera/color/image_raw"
MASK_TOPIC = "/detector/instance_masks"
SERVICE = "/stub/segment_objects"
WIDTH, HEIGHT = 320, 240


@pytest.mark.launch_test
def generate_test_description():
    actions = []
    if SegmentObjects is not None:
        actions.append(
            Node(
                package="canopy_perception",
                executable="detector",
                name="detector",
                output="screen",
                remappings=[("color/image_raw", CAMERA_TOPIC)],
                parameters=[
                    {
                        "segment_service": SERVICE,
                        "detect_rate_hz": 4.0,
                        "phrases": ["red block", "block"],
                        "max_image_age_s": 5.0,
                        "box_threshold": 0.4,
                    }
                ],
            )
        )
        actions.append(
            Node(
                package="canopy_perception",
                executable="segmenter",
                name="segmenter",
                output="screen",
                parameters=[
                    {
                        "segment_service": SERVICE,
                        "cameras": [f"main={CAMERA_TOPIC}"],
                        "max_image_age_s": 5.0,
                        "zmq_timeout_ms": 5000,
                    }
                ],
            )
        )
    return LaunchDescription([*actions, launch_testing.actions.ReadyToTest()])


def _segment(label, score, x, y, w, h):
    segment = Segment(label=label, score=score)
    segment.roi.x_offset, segment.roi.y_offset = x, y
    segment.roi.width, segment.roi.height = w, h
    segment.mask.height, segment.mask.width = h, w
    segment.mask.encoding = "mono8"
    segment.mask.step = w
    segment.mask.data = bytes([255]) * (w * h)
    return segment


@unittest.skipIf(SegmentObjects is None, "sam3_inference_msgs is not built")
class TestDetectorService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = RclpyNode("detector_service_probe")
        cls.requests = []
        cls.node.create_service(SegmentObjects, SERVICE, cls._answer)
        cls.camera = cls.node.create_publisher(Image, CAMERA_TOPIC, qos_profile_sensor_data)
        cls.received = []
        cls.node.create_subscription(
            InstanceMaskArray, MASK_TOPIC, lambda msg: cls.received.append(msg), 2
        )

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    @classmethod
    def _answer(cls, request, response):
        cls.requests.append(request)
        response.success = True
        response.header = request.image.header
        # The same block under both phrases, the better score first, and one elsewhere.
        response.segments = [
            _segment("red block", 0.9, 100, 60, 40, 30),
            _segment("block", 0.6, 101, 61, 40, 30),
            _segment("block", 0.5, 10, 10, 20, 20),
        ]
        return response

    def _await_mask(self, timeout_s=30.0):
        self.received.clear()
        deadline = self.node.get_clock().now().nanoseconds + int(timeout_s * 1e9)
        while self.node.get_clock().now().nanoseconds < deadline:
            image = Image(height=HEIGHT, width=WIDTH, encoding="rgb8", step=3 * WIDTH)
            image.header.stamp = self.node.get_clock().now().to_msg()
            image.header.frame_id = "camera_color_optical_frame"
            image.data = bytes(HEIGHT * image.step)
            self.camera.publish(image)
            rclpy.spin_once(self.node, timeout_sec=0.2)
            if self.received:
                return self.received[-1]
        return None

    def test_01_the_service_masks_are_published(self):
        masks = self._await_mask()

        self.assertIsNotNone(masks, "no masks from the service's answer")
        self.assertEqual(masks.model, "sam3.1")
        self.assertEqual(masks.header.frame_id, "camera_color_optical_frame")
        request = self.requests[-1]
        self.assertEqual(list(request.prompts), ["red block", "block"])
        self.assertAlmostEqual(request.min_score, 0.4, places=5)

    def test_02_a_box_repeated_under_another_phrase_is_kept_once(self):
        masks = self._await_mask()

        self.assertIsNotNone(masks)
        labels = sorted((i.label, i.roi.x_offset) for i in masks.instances)
        self.assertEqual(labels, [("block", 10), ("red block", 100)])
        kept = next(i for i in masks.instances if i.label == "red block")
        self.assertEqual(len(kept.data), 40 * 30)

    def test_03_the_segmenter_answers_through_the_service(self):
        self._await_mask()  # frames on the camera, which the segmenter holds the newest of
        client = self.node.create_client(SegmentByPrompt, "/segmenter/segment")
        self.assertTrue(client.wait_for_service(timeout_sec=20.0))

        future = client.call_async(SegmentByPrompt.Request(camera="main", prompt="block"))
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=20.0)
        result = future.result()

        self.assertIsNotNone(result, "the segmenter never answered")
        self.assertTrue(result.success, result.message)
        self.assertEqual(list(self.requests[-1].prompts), ["block"])
        self.assertEqual(len(result.masks.instances), 2)


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):
    def test_processes_exit_cleanly(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -15])
