"""
Unit tests for data splitting protocols including leave-one-lot-out (LOO) and leakage verification.
"""

from pathlib import Path
import pytest

from vision.src.splits import (
    assert_no_data_leakage,
    make_all_lots_split,
    make_leave_one_lot_out_splits,
    make_time_based_split,
    PKLOT_LOTS
)


def create_dummy_sample_paths():
    """
    Creates dummy paths mimicking PKLot directory layout:
    PKLot/<lot>/<weather>/<date>/<filename>.jpg
    """
    samples = []
    lots = ["PUCPR", "UFPR04", "UFPR05"]
    weathers = ["Sunny", "Rainy"]

    for lot in lots:
        for w in weathers:
            for day_idx in range(1, 6):
                date_str = f"2026-10-0{day_idx}"
                for img_idx in range(1, 4):
                    p = Path(f"data/PKLot/{lot}/{w}/{date_str}/{lot}_{date_str}_{img_idx:03d}.jpg")
                    xml_p = p.with_suffix(".xml")
                    samples.append((p, xml_p))
    return samples


def test_leave_one_lot_out_3_folds_no_leakage():
    samples = create_dummy_sample_paths()

    for fold in range(3):
        splits = make_leave_one_lot_out_splits(samples, fold=fold)

        # Verification of no leakage
        assert_no_data_leakage(splits)

        test_lot = PKLOT_LOTS[fold]
        # Verify test split only contains test_lot
        for img_p, _ in splits["test"]:
            assert img_p.parent.parent.parent.name == test_lot

        # Verify train and val splits do NOT contain test_lot
        for img_p, _ in splits["train"]:
            assert img_p.parent.parent.parent.name != test_lot
        for img_p, _ in splits["val"]:
            assert img_p.parent.parent.parent.name != test_lot

        # Verify train and val are non-empty
        assert len(splits["train"]) > 0
        assert len(splits["val"]) > 0
        assert len(splits["test"]) > 0


def test_time_based_split_no_leakage():
    samples = create_dummy_sample_paths()
    splits = make_time_based_split(samples, train_ratio=0.6, val_ratio=0.2)

    assert_no_data_leakage(splits)
    assert len(splits["train"]) > 0
    assert len(splits["val"]) > 0
    assert len(splits["test"]) > 0


def test_make_all_lots_split_no_leakage():
    samples = create_dummy_sample_paths()
    splits = make_all_lots_split(samples, val_day_ratio=0.2)

    assert_no_data_leakage(splits)
    assert len(splits["train"]) > 0
    assert len(splits["val"]) > 0
    assert len(splits["test"]) > 0
    assert len(splits["val"]) == len(splits["test"])

    # Verify all 3 lots are represented in train and val
    train_lots = set(img_p.parent.parent.parent.name for img_p, _ in splits["train"])
    val_lots = set(img_p.parent.parent.parent.name for img_p, _ in splits["val"])
    assert train_lots == {"PUCPR", "UFPR04", "UFPR05"}
    assert val_lots == {"PUCPR", "UFPR04", "UFPR05"}


