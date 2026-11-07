"""
Unit tests for data loaders, transformations, and leakage-safe splits.
"""

from pathlib import Path
import pytest
import torch

from src.data_pkl import PKLotDataset, get_default_eval_transforms, get_default_train_transforms
from src.splits import (
    assert_no_data_leakage,
    make_time_based_split,
    make_unseen_lot_split,
)

@pytest.fixture
def sample_pklot_pairs():
    base_dir = Path(__file__).resolve().parent.parent
    pklot_dir = base_dir / "data" / "PKLot"
    img_files = sorted(list(pklot_dir.rglob("*.jpg")))
    pairs = []
    for im in img_files:
        xml_p = im.with_suffix(".xml")
        if xml_p.exists():
            pairs.append((im, xml_p))
    return pairs

def test_pklot_crop_shape_and_label(sample_pklot_pairs):
    assert len(sample_pklot_pairs) > 0, "No PKLot samples found for testing"

    # Take first 2 images
    subset_pairs = sample_pklot_pairs[:2]
    dataset = PKLotDataset(subset_pairs, transform=get_default_eval_transforms(224))

    assert len(dataset) > 0, "PKLotDataset is empty"
    tensor, label = dataset[0]

    # Verify tensor shape: (C, H, W) = (3, 224, 224)
    assert tensor.shape == (3, 224, 224), f"Expected shape (3, 224, 224), got {tensor.shape}"
    # Verify label mapping: must be 0 (free) or 1 (occupied)
    assert label in (0, 1), f"Label must be 0 or 1, got {label}"

def test_unseen_lot_split_no_leakage(sample_pklot_pairs):
    assert len(sample_pklot_pairs) >= 3

    splits = make_unseen_lot_split(
        sample_pklot_pairs,
        test_lot="UFPR05",
        val_lot="UFPR04",
        train_lot="PUCPR"
    )

    # Assert no image overlaps across train, val, test
    assert_no_data_leakage(splits)

    # Test lot isolation
    test_lots = set(img.parent.parent.parent.name for img, _ in splits["test"])
    assert test_lots == {"UFPR05"}, f"Test split must only contain UFPR05, got {test_lots}"

def test_time_based_split_no_leakage(sample_pklot_pairs):
    assert len(sample_pklot_pairs) >= 3

    splits = make_time_based_split(sample_pklot_pairs, train_ratio=0.5, val_ratio=0.25)

    # Assert no image overlaps
    assert_no_data_leakage(splits)

    # Assert no date overlaps within any lot
    for split_a_name in ["train", "val", "test"]:
        for split_b_name in ["train", "val", "test"]:
            if split_a_name >= split_b_name:
                continue
            days_a = set((img.parent.parent.parent.name, img.parent.name) for img, _ in splits[split_a_name])
            days_b = set((img.parent.parent.parent.name, img.parent.name) for img, _ in splits[split_b_name])
            overlap = days_a.intersection(days_b)
            assert len(overlap) == 0, f"Date overlap detected between {split_a_name} and {split_b_name}: {overlap}"
