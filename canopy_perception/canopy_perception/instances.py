"""What the model server does to a segmentation before it answers, for one from a service.

The server drops a box that repeats a better one whatever either is called, and embeds each
instance alone on a grey square. Instances that came from a segmentation service (sam3_inference)
get the same here, so the world model cannot tell the two apart.
"""

import numpy as np

# The background an instance is embedded on, as servers/semantic/embedding.py's.
GREY = 128


def box_iou(a, b):
    """Intersection over union of two [x, y, w, h] boxes."""
    width = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    height = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    overlap = width * height
    return overlap / (a[2] * a[3] + b[2] * b[3] - overlap)


def suppress_duplicates(instances, iou_threshold=0.7):
    """Drops an instance whose box repeats a better-scored one's, whatever either is called.

    Each phrase is segmented on its own, so a chair can also be an armchair: one track each
    otherwise. 0.7 is the server's own threshold.
    """
    kept = []
    for instance in sorted(instances, key=lambda candidate: -candidate["score"]):
        if all(box_iou(instance["roi"], other["roi"]) <= iou_threshold for other in kept):
            kept.append(instance)
    return kept


def masked_crop(image, roi, mask, pad=0.1):
    """The instance alone on a grey square with a margin: what gets embedded for it."""
    x, y, w, h = (int(value) for value in roi)
    side = int(round(max(w, h) * (1.0 + 2.0 * pad)))
    canvas = np.full((side, side, 3), GREY, np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    np.copyto(
        canvas[top : top + h, left : left + w],
        image[y : y + h, x : x + w],
        where=np.asarray(mask)[..., None] > 0,
    )
    return canvas


def instances_from_segments(segments):
    """A SegmentObjects answer's segments as the server's instances: label, score, roi, mask."""
    instances = []
    for segment in segments:
        rows = np.frombuffer(segment.mask.data, dtype=np.uint8).reshape(
            segment.mask.height, segment.mask.step
        )
        roi = segment.roi
        instances.append(
            {
                "label": segment.label,
                "score": float(segment.score),
                "roi": [roi.x_offset, roi.y_offset, roi.width, roi.height],
                "mask": rows[:, : segment.mask.width],
            }
        )
    return instances
