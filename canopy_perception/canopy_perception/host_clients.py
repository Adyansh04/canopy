"""ZMQ client for the host model servers, which run outside the container for torch and CUDA.

REQ/REP with msgpack_numpy bodies. ReqClient is the transport; VisionClient names a call by
"endpoint", as servers/semantic_server.py and any server speaking its protocol expect.
"""

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq


def _decode(obj):
    """msgpack_numpy's hook, minus object arrays: those are unpickled, which runs a peer's code."""
    if isinstance(obj, dict) and obj.get(b"nd") and obj.get(b"kind") == b"O":
        raise ValueError("object arrays are refused")
    return mnp.decode(obj)


class ReqClient:
    """One REQ socket, rebuilt after a failed request: a timed-out REQ socket cannot send again."""

    def __init__(self, address, timeout_ms, server_name):
        self._address = address
        self._timeout_ms = timeout_ms
        self._server_name = server_name
        self._context = zmq.Context()
        self._socket = None
        self._connect()

    def _connect(self):
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._socket.connect(self._address)

    def _send(self, name, request):
        try:
            self._socket.send(msgpack.packb(request, default=mnp.encode))
            reply = msgpack.unpackb(self._socket.recv(), object_hook=_decode, raw=False)
        except zmq.ZMQError as error:
            self._connect()
            raise RuntimeError(f"{name} did not complete: {error}") from error
        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError(f"{self._server_name} refused {name}: {reply['error']}")
        return reply

    def close(self):
        if self._socket is not None:
            self._socket.close(linger=0)
        self._context.term()


class VisionClient(ReqClient):
    """The call is named by "endpoint", its body sits under "data"."""

    def __init__(self, address, timeout_ms):
        super().__init__(address, timeout_ms, "the model server")

    def call(self, endpoint, data=None):
        request = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        return self._send(endpoint, request)


def image_to_array(msg):
    """An rgb8 or bgr8 Image as (H, W, 3) RGB uint8, without cv_bridge for one reshape."""
    if msg.encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"expected rgb8 or bgr8, got {msg.encoding}")
    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    frame = rows[:, : 3 * msg.width].reshape(msg.height, msg.width, 3)
    return frame[:, :, ::-1] if msg.encoding == "bgr8" else frame
