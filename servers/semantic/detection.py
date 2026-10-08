"""Detectors: `name` and segment(image, phrases, box_threshold, text_threshold) -> instances."""

import contextlib
import functools
import os
from pathlib import Path

import numpy as np
from PIL import Image

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


class Sam3Detector:
    """SAM 3.1 instance masks for each phrase, from `find` (see build_sam3).

    A score is SAM 3's own: a query's match times the phrase's presence in the image.
    """

    def __init__(self, find):
        self.name = "sam3.1"
        self._find = find

    def segment(self, image, phrases, box_threshold, text_threshold):
        del text_threshold  # one calibrated score
        instances = []
        for phrase, masks, scores in self._find(image, phrases, box_threshold):
            for mask, score in zip(masks, scores, strict=True):
                cropped = _roi_and_crop(mask)
                if cropped is not None:
                    roi, data = cropped
                    instances.append(
                        {"label": phrase, "score": float(score), "roi": roi, "mask": data}
                    )
        # Each phrase is asked on its own, so a chair can also be an armchair: one track each.
        return _suppress_duplicates(instances)


def build_sam3(section, device):
    """SAM 3.1's image detector, its phrases in batches through the fusion encoder and decoder.

    Each phrase conditions all 5184 image tokens on its own, so the cost grows with the word list:
    0.45 s for the image, then about 40 ms a phrase on a laptop RTX 4080. A batch keeps the GPU busy
    where a phrase at a time waits on kernel launches, and only the phrases a query matched get
    masks, since most name nothing in a frame: twice as fast as Meta's processor a phrase at a time,
    with the same answers.
    """
    checkpoint = Path(section["checkpoint"]).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{checkpoint} is missing; {VENV_HINT}")
    import torch
    from sam3.model.data_misc import FindStage, interpolate
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        device=device, checkpoint_path=str(checkpoint), load_from_HF=False
    )
    processor = Sam3Processor(model, device=device)
    batch, mask_batch = int(section["batch"]), int(section["mask_batch"])

    @functools.lru_cache(maxsize=64)  # a word list comes back with every frame
    def encode(words):
        return model.backbone.forward_text(list(words), device=device)

    @torch.inference_mode()
    def find(image, phrases, threshold):
        found = []
        height, width = image.shape[:2]
        # bf16 autocast, as Meta's examples run it; bf16 weights lose the detections.
        with torch.autocast("cuda", torch.bfloat16, enabled=str(device).startswith("cuda")):
            # A PIL image: the processor takes a numpy image's last two axes for its size.
            image_out = processor.set_image(Image.fromarray(image))["backbone_out"]
            for first in range(0, len(phrases), batch):
                words = tuple(phrases[first : first + batch])
                n = len(words)
                backbone_out = dict(image_out, **encode(words))
                find_input = FindStage(
                    img_ids=torch.zeros(n, dtype=torch.long, device=device),
                    text_ids=torch.arange(n, device=device),
                    input_boxes=None,
                    input_boxes_mask=None,
                    input_boxes_label=None,
                    input_points=None,
                    input_points_mask=None,
                )
                prompt, prompt_mask, backbone_out = model._encode_prompt(
                    backbone_out, find_input, model._get_dummy_prompt(num_prompts=n)
                )
                backbone_out, encoder_out, _ = model._run_encoder(
                    backbone_out, find_input, prompt, prompt_mask
                )
                memory = encoder_out["encoder_hidden_states"]
                out = {
                    "encoder_hidden_states": memory,
                    "prev_encoder_out": {"encoder_out": encoder_out, "backbone_out": backbone_out},
                }
                out, hs = model._run_decoder(
                    memory=memory,
                    pos_embed=encoder_out["pos_embed"],
                    src_mask=encoder_out["padding_mask"],
                    out=out,
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                    encoder_out=encoder_out,
                )
                scores = out["pred_logits"].sigmoid() * out["presence_logit_dec"].sigmoid()[:, None]
                keep = scores[..., 0] > threshold
                matched = keep.any(dim=1).nonzero()[:, 0]
                if matched.numel() == 0:
                    continue  # split would still hand over one empty batch
                # A few phrases at a time: a mask head's pass takes ~0.2 GB a phrase.
                for rows in matched.split(mask_batch):
                    heads = {
                        "queries": out["queries"][rows],
                        "presence_feats": out["presence_feats"][:, rows],
                        "presence_logit_dec": out["presence_logit_dec"][rows],
                        "pred_logits": out["pred_logits"][rows],
                        "pred_boxes": out["pred_boxes"][rows],
                        "pred_boxes_xyxy": out["pred_boxes_xyxy"][rows],
                        "encoder_hidden_states": memory[:, rows],
                    }
                    model._run_segmentation_heads(
                        out=heads,
                        backbone_out=backbone_out,
                        img_ids=find_input.img_ids[rows],
                        vis_feat_sizes=encoder_out["vis_feat_sizes"],
                        encoder_hidden_states=memory[:, rows],
                        prompt=prompt[:, rows],
                        prompt_mask=prompt_mask[rows],
                        hs=hs[:, rows],
                    )
                    for slot, row in enumerate(rows.tolist()):
                        logits = heads["pred_masks"][slot][keep[row]][:, None]
                        masks = interpolate(
                            logits, (height, width), mode="bilinear", align_corners=False
                        )
                        masks = masks > 0  # Meta's sigmoid over 0.5
                        found.append(
                            (
                                words[row],
                                masks[:, 0].cpu().numpy(),
                                scores[row, keep[row], 0].tolist(),
                            )
                        )
        return found

    return Sam3Detector(find)


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
