"""The request handlers, the backend registries, and the ZMQ loop that answers on two ports."""

import sys
import threading
import time

import numpy as np

from .config import VENV_HINT
from .describers import DescriberChain, NoDescriber, OpenAIDescriber
from .detection import build_sam3, build_yoloe
from .embedding import Siglip2Embedder, masked_crop
from .gemini import GeminiDescriber, GeminiLimiter

# Eight 512 px images are about 2k tokens, half of the local VLM's context.
MAX_DESCRIBE_IMAGES = 8


# Registries: one entry per backend, built from its section of the config.

DETECTORS = {
    "sam3.1": lambda config, device: build_sam3(config["sam3.1"], device),
    "yoloe": lambda config, device: build_yoloe(config["yoloe"], device, prompt_free=False),
    "yoloe-pf": lambda config, device: build_yoloe(config["yoloe-pf"], device, prompt_free=True),
}
EMBEDDERS = {
    "siglip2": lambda config, device: Siglip2Embedder(
        device, config["siglip2"]["model"], config["siglip2"]["revision"]
    ),
    "none": lambda config, device: None,
}
DESCRIBERS = {
    "gemini": lambda config: GeminiDescriber(
        {task: config["gemini"][task] for task in ("object", "room")},
        config["gemini"]["models"],
        config["gemini"]["key_file"],
        GeminiLimiter(config["gemini"]["usage_file"], config["gemini"]["models"]),
        config["gemini"]["timeout_s"],
        config["gemini"]["min_attempt_s"],
    ),
    "openai": lambda config: OpenAIDescriber(
        config["openai"]["base_url"], config["openai"]["model"], config["openai"]["timeout_s"]
    ),
    "none": lambda config: NoDescriber(),
}


def _image(value, what="image"):
    if not isinstance(value, np.ndarray) or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"{what} must be an (H, W, 3) uint8 array")
    if value.dtype != np.uint8:
        raise ValueError(f"{what} must be uint8, got {value.dtype}")
    return value


class SemanticServer:
    """The request handlers, apart from the socket so they can be driven directly."""

    def __init__(self, detector, embedder, describer, device="cpu", box_threshold=0.25):
        self.detector = detector
        self.embedder = embedder
        self.describer = describer
        self.device = device
        self.box_threshold = box_threshold

    def handle(self, request, endpoints=None):
        """A reply for every request, failures included: a dropped reply strands a REQ socket."""
        endpoint = request.get("endpoint") if isinstance(request, dict) else None
        handlers = {
            "ping": self._ping,
            "segment": self._segment,
            "embed_text": self._embed_text,
            "embed_image": self._embed_image,
            "describe": self._describe,
        }
        try:
            if endpoint not in handlers:
                raise ValueError(f"unknown endpoint {endpoint!r}")
            if endpoints is not None and endpoint not in endpoints:
                raise ValueError(f"{endpoint} is served on the other port")
            data = request.get("data") or {}
            if not isinstance(data, dict):
                raise ValueError("data must be a map")
            return handlers[endpoint](data)
        except Exception as error:  # noqa: BLE001 - the client gets every failure as a reply
            reply = {"error": f"{type(error).__name__}: {error}"}
            print(f"  {endpoint}: {reply['error']}", file=sys.stderr, flush=True)
            return reply

    def _ping(self, data):
        del data
        return {
            "status": "ok",
            "backend": self.detector.name,
            "device": self.device,
            "embedder": self.embedder.name if self.embedder else "",
            "describers": self.describer.names,
        }

    def _segment(self, data):
        image = _image(data.get("image"))
        embed = bool(data.get("embed", False)) and self.embedder is not None
        # Stripped and deduplicated, so equal vocabularies hit the same cache entry.
        phrases = list(
            dict.fromkeys(
                str(phrase).strip() for phrase in data.get("phrases") or [] if str(phrase).strip()
            )
        )
        started = time.perf_counter()
        instances = self.detector.segment(
            image,
            phrases,
            float(data.get("box_threshold", self.box_threshold)),
            float(data.get("text_threshold", 0.25)),
        )
        reply = {"model": self.detector.name, "instances": instances}
        if embed and instances:
            embedded = time.perf_counter()
            embeddings = self.embedder.embed_images(
                [masked_crop(image, instance["roi"], instance["mask"]) for instance in instances]
            )
            for instance, embedding in zip(instances, embeddings, strict=True):
                instance["embedding"] = embedding
            reply["embed_ms"] = (time.perf_counter() - embedded) * 1000.0
        reply["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return reply

    def _embed_text(self, data):
        texts = [str(text) for text in data.get("texts") or []]
        if not texts or not all(text.strip() for text in texts):
            raise ValueError("texts must be a non-empty list of non-empty strings")
        return {"model": self._embedder().name, "embeddings": self._embedder().embed_text(texts)}

    def _embed_image(self, data):
        images = [_image(image, "each image") for image in data.get("images") or []]
        if not images:
            raise ValueError("images must be a non-empty list")
        return {"model": self._embedder().name, "embeddings": self._embedder().embed_images(images)}

    def _embedder(self):
        if self.embedder is None:
            raise ValueError("this server was started with --embedder none")
        return self.embedder

    def _describe(self, data):
        task = data.get("task")
        if task not in ("object", "room"):
            raise ValueError(f"task must be 'object' or 'room', got {task!r}")
        images = [_image(image, "each image") for image in data.get("images") or []]
        if task == "object" and not images:
            raise ValueError("an object needs at least one image")
        if len(images) > MAX_DESCRIBE_IMAGES:
            raise ValueError(f"at most {MAX_DESCRIBE_IMAGES} images per request")
        context = data.get("context") or {}
        if not isinstance(context, dict):
            raise ValueError("context must be a map")
        started = time.perf_counter()
        answer = self.describer.describe(task, images, context)
        answer["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return answer


def _pick(registry, name, kind):
    if name not in registry:
        sys.exit(f"unknown {kind} {name!r}; one of: {', '.join(registry)}")
    return registry[name]


def build_server(config):
    import torch

    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    names = [name.strip() for name in str(config["describer"]).split(",") if name.strip()]
    describer = DescriberChain(_pick(DESCRIBERS, name, "describer")(config) for name in names)
    detector = _pick(DETECTORS, config["detector"], "detector")(config, device)
    embedder = _pick(EMBEDDERS, config["embedder"], "embedder")(config, device)
    return SemanticServer(detector, embedder, describer, device, float(config["box_threshold"]))


def _decode(obj):
    """msgpack_numpy's hook, minus object arrays: those are unpickled, which runs a peer's code."""
    import msgpack_numpy as mnp

    if isinstance(obj, dict) and obj.get(b"nd") and obj.get(b"kind") == b"O":
        raise ValueError("object arrays are refused")
    return mnp.decode(obj)


def _phrases(request):
    """How many phrases a request asks about, which is what its turn on the GPU costs."""
    data = request.get("data") if isinstance(request, dict) else None
    phrases = data.get("phrases") if isinstance(data, dict) else None
    return len(phrases) if isinstance(phrases, list) else 0


def _answer(socket, server, endpoints):
    """Answers `endpoints` on one ROUTER socket until its context is terminated.

    Of the requests waiting, the one with the fewest phrases goes first, so a pick's phrase does
    not wait behind two mapping detectors' thirty-odd each. Every request gets an answer: a REQ
    client left without one is stuck.
    """
    import msgpack
    import msgpack_numpy as mnp
    import zmq

    def received(frames):
        try:
            request = msgpack.unpackb(frames[-1], object_hook=_decode, raw=False)
        except Exception as error:  # noqa: BLE001 - see above
            return frames[:-1], -1, {"error": f"undecodable request: {error}"}
        return frames[:-1], _phrases(request), request

    # ponytail: shortest first can hold a long request back under a stream of short ones; one
    # arm's detector at 1 Hz leaves the mapping detectors half the GPU. Weigh in waiting if not.
    waiting = []
    try:
        while True:
            if not waiting:
                waiting.append(received(socket.recv_multipart()))
            while socket.poll(0):
                waiting.append(received(socket.recv_multipart()))
            envelope, cost, request = waiting.pop(
                min(range(len(waiting)), key=lambda i: waiting[i][1])
            )
            reply = request if cost < 0 else server.handle(request, endpoints)
            socket.send_multipart([*envelope, msgpack.packb(reply, default=mnp.encode)])
    except zmq.ContextTerminated:
        pass
    finally:
        socket.close(linger=0)


def serve(server, config):
    try:
        import zmq
    except ImportError as error:
        sys.exit(f"{error}. {VENV_HINT}")

    context = zmq.Context()
    models, describe = context.socket(zmq.ROUTER), context.socket(zmq.ROUTER)
    models.bind(f"tcp://{config['host']}:{config['port']}")
    describe.bind(f"tcp://{config['host']}:{config['port'] + 1}")
    # The describe thread touches no GPU model: Gemini is remote, the local VLM another process.
    threading.Thread(
        target=_answer, args=(describe, server, {"ping", "describe"}), daemon=True
    ).start()
    embedder = server.embedder.name if server.embedder else "no embedder"
    print(
        f"serving {server.detector.name}, {embedder} on {server.device} at "
        f"tcp://{config['host']}:{config['port']}, describers "
        f"{','.join(server.describer.names) or 'none'} on port {config['port'] + 1}",
        flush=True,
    )
    try:
        _answer(models, server, {"ping", "segment", "embed_text", "embed_image"})
    except KeyboardInterrupt:
        pass
    finally:
        context.term()
