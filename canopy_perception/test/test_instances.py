"""The server's post-processing, redone for instances from a segmentation service."""

from types import SimpleNamespace

import numpy as np

from canopy_perception.instances import (
    GREY,
    box_iou,
    instances_from_segments,
    masked_crop,
    suppress_duplicates,
)


def test_identical_boxes_overlap_fully_and_apart_not_at_all():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou([0, 0, 10, 10], [20, 20, 5, 5]) == 0.0


def test_a_repeated_box_keeps_only_the_better_score_whatever_its_label():
    instances = [
        {"label": "block", "score": 0.6, "roi": [101, 61, 40, 30]},
        {"label": "red block", "score": 0.9, "roi": [100, 60, 40, 30]},
        {"label": "block", "score": 0.5, "roi": [10, 10, 20, 20]},
    ]

    kept = suppress_duplicates(instances)

    assert [(i["label"], i["roi"][0]) for i in kept] == [("red block", 100), ("block", 10)]


def test_the_crop_shows_the_instance_alone_on_grey():
    image = np.full((50, 60, 3), 7, np.uint8)
    mask = np.zeros((10, 20), np.uint8)
    mask[:, :10] = 255

    crop = masked_crop(image, [5, 5, 20, 10], mask)

    assert crop.shape == (24, 24, 3)
    # Inside the mask the image, beside it the grey background.
    assert (crop[7, 2] == 7).all()
    assert (crop[7, 20] == GREY).all()


def test_a_segment_mask_padded_to_its_step_comes_back_unpadded():
    rows = np.zeros((3, 8), np.uint8)
    rows[:, :5] = 255
    segment = SimpleNamespace(
        label="cup",
        score=0.8,
        roi=SimpleNamespace(x_offset=4, y_offset=2, width=5, height=3),
        mask=SimpleNamespace(height=3, width=5, step=8, data=rows.tobytes()),
    )

    (instance,) = instances_from_segments([segment])

    assert instance["roi"] == [4, 2, 5, 3]
    assert instance["mask"].shape == (3, 5)
    assert (instance["mask"] == 255).all()
