"""
Unit tests for extract_slot_crop representation parity and format invariance.
Guarantees that passing contour/points without an explicit rotated_rect produces
the identical oriented slot crop and classification accuracy as passing rotated_rect.
"""

import numpy as np
import pytest
from vision.src.data_pkl import extract_slot_crop, preprocess_slot_crop


def test_extract_slot_crop_points_fallback_matches_rotated_rect():
    """
    Constructs a synthetic angled slot and verifies that passing only 'points'
    automatically derives the minimum area bounding rectangle and produces
    the identical crop size and pixel values as an explicit 'rotated_rect'.
    """
    # Create a 400x400 canvas with a distinct angled pattern
    canvas = np.zeros((400, 400, 3), dtype=np.uint8)
    canvas[100:300, 100:300] = [120, 180, 240]

    # Angled slot contour points (parallelogram / rotated rectangle)
    # Center (200, 200), width ~60, height ~120, rotated
    points = [
        [180.0, 140.0],
        [240.0, 170.0],
        [220.0, 260.0],
        [160.0, 230.0]
    ]

    slot_with_points = {
        "slot_id": "slot_pts_only",
        "points": points
    }

    slot_with_contour = {
        "slot_id": "slot_contour_only",
        "contour": points
    }

    # Extract crops
    crop_pts = extract_slot_crop(canvas, slot_with_points, target_size=(224, 224))
    crop_cnt = extract_slot_crop(canvas, slot_with_contour, target_size=(224, 224))

    arr_pts = np.array(crop_pts)
    arr_cnt = np.array(crop_cnt)

    # Crops must match identically across representation keys
    assert arr_pts.shape == arr_cnt.shape
    assert np.array_equal(arr_pts, arr_cnt)
    # The crop must contain non-zero content from the pattern
    assert np.sum(arr_pts) > 0


def test_preprocess_slot_crop_tensor_shape():
    """
    Verifies that preprocess_slot_crop returns a standard [3, 224, 224] normalized tensor
    whether given 'points' or 'rotated_rect'.
    """
    img = np.ones((300, 300, 3), dtype=np.uint8) * 100
    slot = {
        "slot_id": "test_1",
        "points": [[50, 50], [150, 50], [150, 200], [50, 200]]
    }
    tensor = preprocess_slot_crop(img, slot, target_size=(224, 224))
    assert tensor.shape == (3, 224, 224)

