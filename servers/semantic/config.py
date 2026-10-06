"""The server's configuration: DEFAULTS, then a --config file, the flags and --set overrides."""

import copy
import os
from pathlib import Path

VENV_HINT = "servers/setup.sh has not been run, or the wrong interpreter is being used"
REPO = Path(__file__).resolve().parents[2]
VOCABULARY = REPO / "canopy_perception/config/detector.yaml"
# Where servers/setup.sh put the weights; its CANOPY_HOME.
WEIGHTS = f"{os.environ.get('CANOPY_HOME', '~/.local/share/canopy')}/weights"


DEFAULTS = {
    "host": "127.0.0.1",
    "port": 5561,
    "device": "auto",
    "detector": "sam3.1",
    "embedder": "siglip2",
    "describer": "gemini,openai",
    # When a request carries none; the detector always sends its own.
    "box_threshold": 0.25,
    # SAM 3.1's checkpoint carries its image detector too, which is what segment asks. `batch`
    # phrases go through its encoder together, and `mask_batch` of those that matched through its
    # mask head: fewer of each for a smaller GPU, at some speed.
    "sam3.1": {"checkpoint": f"{WEIGHTS}/sam3.1_multiplex.pt", "batch": 32, "mask_batch": 8},
    "yoloe": {"weights": f"{WEIGHTS}/yoloe-26l-seg.pt", "imgsz": 640, "half": True, "max_det": 100},
    "yoloe-pf": {
        "weights": f"{WEIGHTS}/yoloe-26l-seg-pf.pt",
        "imgsz": 640,
        "half": True,
        "max_det": 100,
    },
    # The revision servers/setup.sh downloads: another would move every saved embedding.
    "siglip2": {
        "model": "google/siglip2-base-patch16-256",
        "revision": "3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab",
    },
    "gemini": {
        "key_file": "~/.config/canopy/gemini.env",
        "usage_file": "~/.config/canopy/gemini_usage.json",
        # One budget for a whole route: canopy_perception's detector waits 20 s for this server, its
        # describer 30, and a describe call holds up every segment behind it.
        "timeout_s": 10.0,
        # Less of the budget than a model needs to answer, and it is not asked: the request counts.
        "min_attempt_s": 2.0,
        # Google limits each model separately, per Cloud project (AI Studio, Rate limits). The caps
        # sit a little under those, since other apps may share the project.
        "models": {
            "gemini-3.5-flash-lite": {"per_minute": 14, "per_day": 480},
            "gemma-4-26b-a4b-it": {"per_minute": 28, "per_day": 14000},
            "gemini-3.1-flash-lite": {"per_minute": 14, "per_day": 480},
            "gemini-3.8-flash": {"per_minute": 4, "per_day": 18, "thinking": "low"},
            "gemini-3.6-flash": {"per_minute": 4, "per_day": 18, "thinking": "low"},
        },
        # Asked in order. Objects are many, so they go where the requests are; rooms are few and
        # steer every search in them, so they go to the larger models first.
        "object": ["gemini-3.5-flash-lite", "gemma-4-26b-a4b-it", "gemini-3.1-flash-lite"],
        "room": [
            "gemini-3.8-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash-lite",
            "gemma-4-26b-a4b-it",
        ],
    },
    # After Gemini's 10 s, still inside the describer's 30 s wait.
    "openai": {"base_url": "http://127.0.0.1:8080/v1", "model": "qwen3.5-4b", "timeout_s": 15.0},
}


def _merge(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(args):
    """DEFAULTS, then --config, then the backend flags, then each --set section.key=value."""
    import yaml

    config = copy.deepcopy(DEFAULTS)
    if args.config:
        _merge(config, yaml.safe_load(Path(args.config).read_text()) or {})
    for key in ("host", "port", "device", "detector", "embedder", "describer"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    for assignment in args.set:
        key, _, value = assignment.partition("=")
        *sections, leaf = key.split(".")
        target = config
        for section in sections:
            target = target.setdefault(section, {})
        target[leaf] = yaml.safe_load(value)
    return config
