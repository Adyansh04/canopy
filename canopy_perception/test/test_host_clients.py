"""image_to_array against the layouts drivers publish, and the decoder's refusal of pickles."""

import pickle

import msgpack
import numpy as np
import pytest
from sensor_msgs.msg import Image

from canopy_perception.host_clients import _decode, image_to_array


def image(encoding, pixels, step):
    height, width, _ = pixels.shape
    rows = np.zeros((height, step), dtype=np.uint8)
    rows[:, : 3 * width] = pixels.reshape(height, 3 * width)
    return Image(height=height, width=width, encoding=encoding, step=step, data=rows.tobytes())


PIXELS = np.arange(2 * 5 * 3, dtype=np.uint8).reshape(2, 5, 3)


def test_rows_padded_to_four_bytes():
    np.testing.assert_array_equal(image_to_array(image("rgb8", PIXELS, 16)), PIXELS)


def test_bgr8_comes_back_as_rgb():
    np.testing.assert_array_equal(image_to_array(image("bgr8", PIXELS[:, :, ::-1], 15)), PIXELS)


def test_other_encodings_are_refused():
    with pytest.raises(ValueError, match="rgb8 or bgr8"):
        image_to_array(image("mono8", PIXELS, 15))


def test_object_arrays_are_refused():
    array = {b"nd": True, b"type": "|O", b"kind": b"O", b"shape": [1], b"data": pickle.dumps(1)}
    with pytest.raises(ValueError, match="object arrays"):
        msgpack.unpackb(msgpack.packb(array), object_hook=_decode, raw=False)
