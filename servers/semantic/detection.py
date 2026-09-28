"""Detectors: `name` and segment(image, phrases, box_threshold, text_threshold) -> instances.

A grounding segmenter (Grounding DINO with SAM 2, or SAM 3) fits as it is, given a DETECTORS entry
in server.py.
"""

import contextlib
import os
from pathlib import Path

import numpy as np

from .config import VENV_HINT


def _roi_and_crop(mask):
    """Full-frame boolean mask to (roi, cropped uint8 mask), or None when it is empty."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    crop = np.ascontiguousarray(mask[y0:y1, x0:x1].astype(np.uint8) * 255)
    return [x0, y0, x1 - x0, y1 - y0], crop


def _box_iou(a, b):
    """Intersection over union of two [x, y, w, h] boxes."""
    width = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    height = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    overlap = width * height
    return overlap / (a[2] * a[3] + b[2] * b[3] - overlap)


def _suppress_duplicates(instances, iou_threshold=0.7):
    """Drops an instance whose box repeats a better-scored one's, whatever either is called.

    YOLOE-26's end-to-end head skips NMS, and neighbouring anchors can each report one object;
    every copy would become a track. 0.7 is ultralytics' own NMS threshold.
    """
    kept = []
    for instance in sorted(instances, key=lambda candidate: -candidate["score"]):
        if all(_box_iou(instance["roi"], other["roi"]) <= iou_threshold for other in kept):
            kept.append(instance)
    return kept


class YoloeDetector:
    """YOLOE-26 instance masks; text-prompted by the phrases, or prompt-free."""

    def __init__(
        self, model, device, weights_dir, prompt_free=False, imgsz=640, half=True, max_det=100
    ):
        self.name = "yoloe-pf" if prompt_free else "yoloe"
        self._model = model
        self._device = device
        self._weights_dir = Path(weights_dir)
        self._prompt_free = prompt_free
        self._imgsz = imgsz
        self._half = half
        self._max_det = max_det
        self._vocabulary = None

    def segment(self, image, phrases, box_threshold, text_threshold):
        del text_threshold  # one calibrated score per region
        if not self._prompt_free:
            self._use_vocabulary(tuple(phrases))
        result = self._model.predict(
            # Ultralytics reads a numpy image as OpenCV BGR; the wire carries RGB.
            np.ascontiguousarray(image[..., ::-1]),
            conf=box_threshold,
            imgsz=self._imgsz,
            quantize=16 if self._half else 32,
            device=self._device,
            max_det=self._max_det,
            # Full-resolution masks, cut to each box.
            retina_masks=True,
            # One label per region: otherwise the end-to-end head emits a region once per class
            # that scores above the threshold, and each copy would become a track.
            agnostic_nms=True,
            verbose=False,
        )[0]
        if result.masks is None:
            return []
        instances = []
        masks = result.masks.data.bool().cpu().numpy()
        for mask, score, cls in zip(
            masks, result.boxes.conf.tolist(), result.boxes.cls.tolist(), strict=True
        ):
            cropped = _roi_and_crop(mask)
            if cropped is None:
                continue
            roi, data = cropped
            instances.append(
                {"label": result.names[int(cls)], "score": float(score), "roi": roi, "mask": data}
            )
        return _suppress_duplicates(instances)

    def _use_vocabulary(self, vocabulary):
        if not vocabulary:
            raise ValueError("at least one phrase is required; --detector yoloe-pf takes none")
        if vocabulary == self._vocabulary:
            return
        # set_classes re-encodes every phrase and rebuilds the predictor, so only a new list pays.
        # The text encoder is looked up in the working directory.
        with contextlib.chdir(self._weights_dir):
            self._model.set_classes(list(vocabulary))
        self._vocabulary = vocabulary


def build_yoloe(section, device, prompt_free):
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    os.environ.setdefault("YOLO_OFFLINE", "True")  # no analytics, no surprise downloads
    from ultralytics import YOLOE

    weights = Path(section["weights"]).expanduser()
    if not weights.is_file():
        raise FileNotFoundError(f"{weights} is missing; {VENV_HINT}")
    model = YOLOE(str(weights))
    model.to(device)
    return YoloeDetector(
        model,
        device,
        weights.parent,
        prompt_free,
        int(section["imgsz"]),
        bool(section["half"]),
        int(section["max_det"]),
    )
