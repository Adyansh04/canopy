#!/usr/bin/env python3
"""Serves detection, embeddings and descriptions for the semantic map, on the host GPU.

The robot's image carries no torch or CUDA, so its nodes reach the models over ZMQ.
canopy_perception's detector and describer are the clients. Unlike a grounding segmenter that runs
a pass per phrase, `phrases` here is the detector's vocabulary, scored against every region in one
forward pass, so a list of a hundred names costs about what one does.

Each backend is chosen by a flag or by --config, and the replies name no model-specific fields:

  --detector   yoloe     YOLOE-26 prompted with the request's phrases (default)
               yoloe-pf  YOLOE-26 prompt-free: its built-in vocabulary; phrases are ignored
  --embedder   siglip2   SigLIP 2 image and text embeddings (default), or none
  --describer  gemini    Gemini REST API, free tier: each task's models in turn, each under its
                         own caps a minute and a day (DEFAULTS["gemini"])
               openai    any OpenAI-compatible server; start-vlm.sh beside this file runs one
               none      echoes the detector's label at confidence 0, so describe never fails
               A list such as gemini,openai (the default) falls through in order when one is
               rate limited or unreachable.

Models:
  YOLOE-26l-seg, -seg-pf   https://docs.ultralytics.com/models/yoloe (arXiv 2503.07465)
  MobileCLIP2-B text       https://github.com/apple/ml-mobileclip (YOLOE's text encoder)
  SigLIP 2 B/16-256        https://huggingface.co/google/siglip2-base-patch16-256
  Gemini 3.5 Flash-Lite    https://ai.google.dev/gemini-api/docs/models, as are 3.1 Flash-Lite,
                           3.8 Flash and 3.6 Flash
  Gemma 4 26B A4B          https://ai.google.dev/gemma, through the same API
  Qwen3.5-4B Q4_K_M        https://huggingface.co/Qwen/Qwen3.5-4B, GGUF from
                           https://huggingface.co/unsloth/Qwen3.5-4B-GGUF

    ./servers/semantic_server.py
    ./servers/semantic_server.py --describer openai          # offline, after start-vlm.sh start
    ./servers/semantic_server.py --config semantic.yaml --set yoloe.imgsz=800
    ./servers/semantic_server.py --self-test frame.png --describe-test

Run servers/setup.sh first. --config takes a YAML file shaped like DEFAULTS in
semantic/config.py.

Wire protocol, msgpack with msgpack_numpy for the arrays. Any failure is {"error": "..."}:

    {"endpoint": "ping"} -> {"status": "ok", "backend", "device", "embedder", "describers"}
    {"endpoint": "segment", "data": {"image": uint8 (H, W, 3) RGB, "phrases": ["chair", ...],
                                     "box_threshold": 0.3, "embed": false}}
      -> {"model", "elapsed_ms", "instances": [{"label", "score", "roi": [x, y, w, h],
          "mask": uint8 (h, w) 0 or 255, "embedding": float32 (D,), only with embed}]}
    {"endpoint": "embed_text", "data": {"texts": ["a mug", ...]}}
      -> {"model", "embeddings": float32 (N, D), L2-normalised}
    {"endpoint": "embed_image", "data": {"images": [uint8 (H, W, 3), ...]}} -> the same
    {"endpoint": "describe", "data": {"task": "object", "images": [crops],
                                      "context": {"labels": {"chair": 5, "stool": 1}}}}
    {"endpoint": "describe", "data": {"task": "room", "images": [views],
                                      "context": {"objects": {"bed": 1}, "room_type": "bedroom"}}}
      -> {"model", "backend", "name", "caption", "label_ok", "room_type", "confidence",
          "elapsed_ms"}

describe is answered on port + 1 (5562), everything else on port (5561), ping on both: a describe
call waits seconds on the network and must not hold up a segment request behind it.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from semantic.config import VENV_HINT, VOCABULARY, load_config
from semantic.describers import padded_crop
from semantic.server import DESCRIBERS, DETECTORS, EMBEDDERS, build_server, serve


def _load_phrases(value):
    """Comma-separated phrases, or the phrases of a detector parameter file."""
    if value.endswith((".yaml", ".yml")):
        import yaml

        return list(
            # Whatever node the file is keyed for: /** or a name.
            next(iter(yaml.safe_load(Path(value).read_text()).values()))["ros__parameters"][
                "phrases"
            ]
        )
    return [phrase.strip() for phrase in value.split(",") if phrase.strip()]


def self_test(server, args):
    """Runs segment (and describe) over image files and prints what came back, without a socket."""
    import torch
    from PIL import Image

    phrases = _load_phrases(args.phrases)
    print(
        f"{server.detector.name} with {len(phrases)} phrases, "
        f"embedder {server.embedder.name if server.embedder else 'none'}"
    )
    for path in args.self_test:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        reply = server.handle(
            {
                "endpoint": "segment",
                "data": {"image": image, "phrases": phrases, "embed": server.embedder is not None},
            }
        )
        if "error" in reply:
            print(f"{path}  {reply['error']}")
            continue
        instances = sorted(reply["instances"], key=lambda instance: -instance["score"])
        print(
            f"{path}  {image.shape[1]}x{image.shape[0]}  {len(instances)} instances, "
            f"{reply['elapsed_ms']:.0f} ms (embed {reply.get('embed_ms', 0.0):.0f} ms)"
        )
        for instance in instances:
            print(
                f"    {instance['label']:18s} score {instance['score']:.2f}  "
                f"roi {instance['roi']}  {int((instance['mask'] > 0).sum())} px"
            )
        if args.describe_test and instances:
            best = instances[0]
            answer = server.handle(
                {
                    "endpoint": "describe",
                    "data": {
                        "task": "object",
                        "images": [padded_crop(image, best["roi"])],
                        "context": {"labels": {best["label"]: 1}},
                    },
                }
            )
            if "error" in answer:
                print(f"    describe: {answer['error']}")
            else:
                print(
                    f"    describe {best['label']!r} via {answer['backend']} ({answer['model']}, "
                    f"{answer['elapsed_ms']:.0f} ms): {answer['name']!r}, label_ok "
                    f"{answer['label_ok']}, confidence {answer['confidence']:.2f}, "
                    f"{answer['caption']!r}"
                )
    if torch.cuda.is_available():
        print(
            f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB allocated, "
            f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB reserved by torch"
        )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="YAML file shaped like DEFAULTS")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override one config value, e.g. gemini.object=[gemma-4-26b-a4b-it]",
    )
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--device", help="auto, cuda or cpu")
    parser.add_argument("--detector", help=f"one of {', '.join(DETECTORS)}")
    parser.add_argument("--embedder", help=f"one of {', '.join(EMBEDDERS)}")
    parser.add_argument("--describer", help=f"comma separated, from {', '.join(DESCRIBERS)}")
    parser.add_argument(
        "--self-test",
        nargs="+",
        metavar="IMAGE",
        help="run segment over these images and exit, instead of serving",
    )
    parser.add_argument(
        "--describe-test",
        action="store_true",
        help="with --self-test, also describe each image's best instance "
        "(counts against the Gemini caps)",
    )
    parser.add_argument(
        "--phrases",
        default=str(VOCABULARY),
        help="for --self-test: comma separated, or a detector parameter file",
    )
    args = parser.parse_args()

    config = load_config(args)
    if args.self_test and not args.describe_test:
        config["describer"] = ""
    try:
        server = build_server(config)
    except (ImportError, OSError) as error:
        sys.exit(f"{error}\n\n{VENV_HINT}")
    except (KeyError, ValueError) as error:
        sys.exit(f"bad config: {type(error).__name__} {error}")
    if args.self_test:
        return self_test(server, args)
    serve(server, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
