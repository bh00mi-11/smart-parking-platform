"""
Data splitting protocols for parking occupancy datasets.
Implements two leakage-safe splitting protocols:
1. 'unseen-lot': Train/val on two PKLot lots, test on the remaining third lot.
2. 'time-based': Split strictly by DAY (never random per slot), ensuring crops
   from the same full-frame image never leak across splits.

Also provides:
- assert_no_data_leakage()
- print_split_summary()
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import xml.etree.ElementTree as ET

def get_pklot_image_metadata(img_path: Path, xml_path: Path) -> Dict:
    """
    Extracts metadata from PKLot file paths and XML headers.
    """
    parts = img_path.parts
    # Assumes path like .../PKLot/<lot>/<weather>/<date>/<filename>.jpg
    filename = img_path.name
    date_str = img_path.parent.name
    weather = img_path.parent.parent.name
    lot = img_path.parent.parent.parent.name

    # Parse occupied and free counts
    tree = ET.parse(xml_path)
    root = tree.getroot()
    occupied = 0
    free = 0
    for space in root.findall("space"):
        if int(space.get("occupied", 0)) == 1:
            occupied += 1
        else:
            free += 1

    return {
        "img_path": img_path,
        "xml_path": xml_path,
        "lot": lot,
        "weather": weather,
        "day": date_str,
        "occupied": occupied,
        "free": free,
        "total_slots": occupied + free
    }

def make_unseen_lot_split(
    samples: List[Tuple[Path, Path]],
    test_lot: str = "UFPR05",
    val_lot: str = "UFPR04",
    train_lot: str = "PUCPR"
) -> Dict[str, List[Tuple[Path, Path]]]:
    """
    Creates an unseen-lot split:
    Train on train_lot, validate on val_lot, test on test_lot.
    If only 2 lots are provided, train/val are split by day on the training lots.
    """
    splits: Dict[str, List[Tuple[Path, Path]]] = {"train": [], "val": [], "test": []}

    for img_p, xml_p in samples:
        lot = img_p.parent.parent.parent.name
        if lot == test_lot:
            splits["test"].append((img_p, xml_p))
        elif lot == val_lot:
            splits["val"].append((img_p, xml_p))
        else:
            splits["train"].append((img_p, xml_p))

    # If any split is empty, gracefully distribute training lots by day
    if len(splits["train"]) == 0 or len(splits["val"]) == 0:
        non_test = [(img, xml) for img, xml in samples if img.parent.parent.parent.name != test_lot]
        days = sorted(list(set(img.parent.name for img, xml in non_test)))
        val_days = set(days[-max(1, len(days) // 3):])
        splits["train"] = [(img, xml) for img, xml in non_test if img.parent.name not in val_days]
        splits["val"] = [(img, xml) for img, xml in non_test if img.parent.name in val_days]

    return splits

def make_time_based_split(
    samples: List[Tuple[Path, Path]],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Dict[str, List[Tuple[Path, Path]]]:
    """
    Splits samples strictly by DAY within each parking lot.
    Ensures that every crop from a given camera frame stays together,
    preventing intra-image and temporal data leakage.
    """
    # Group by (lot, day)
    lot_days = defaultdict(set)
    day_to_samples = defaultdict(list)

    for img_p, xml_p in samples:
        lot = img_p.parent.parent.parent.name
        day = img_p.parent.name
        lot_days[lot].add(day)
        day_to_samples[(lot, day)].append((img_p, xml_p))

    splits: Dict[str, List[Tuple[Path, Path]]] = {"train": [], "val": [], "test": []}

    for lot, days_set in lot_days.items():
        sorted_days = sorted(list(days_set))
        n_days = len(sorted_days)

        if n_days == 1:
            # Fallback if only 1 day: assign to train
            train_days = sorted_days
            val_days = []
            test_days = []
        elif n_days == 2:
            train_days = [sorted_days[0]]
            val_days = []
            test_days = [sorted_days[1]]
        elif n_days == 3:
            train_days = [sorted_days[0]]
            val_days = [sorted_days[1]]
            test_days = [sorted_days[2]]
        else:
            n_train = max(1, int(n_days * train_ratio))
            n_val = max(1, int(n_days * val_ratio))
            train_days = sorted_days[:n_train]
            val_days = sorted_days[n_train:n_train + n_val]
            test_days = sorted_days[n_train + n_val:]

        for d in train_days:
            splits["train"].extend(day_to_samples[(lot, d)])
        for d in val_days:
            splits["val"].extend(day_to_samples[(lot, d)])
        for d in test_days:
            splits["test"].extend(day_to_samples[(lot, d)])

    return splits

def assert_no_data_leakage(splits: Dict[str, List[Tuple[Path, Path]]]):
    """
    Asserts that no full-frame image appears in more than one split partition.
    Raises AssertionError if leakage is detected.
    """
    image_sets: Dict[str, Set[str]] = {}
    for split_name, sample_list in splits.items():
        image_sets[split_name] = set(str(img.name) for img, _ in sample_list)

    split_names = list(splits.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            overlap = image_sets[s1].intersection(image_sets[s2])
            assert len(overlap) == 0, (
                f"Data leakage detected between '{s1}' and '{s2}'! "
                f"Overlapping images: {overlap}"
            )
    print("[PASSED] Verification passed: 0 image leakage between split partitions.")

def get_split_stats(splits: Dict[str, List[Tuple[Path, Path]]]) -> Dict:
    """
    Computes split sizes, total slot counts, and label balance (occupied vs free).
    """
    stats = {}
    for sname, pairs in splits.items():
        total_slots = 0
        occupied_slots = 0
        free_slots = 0
        days = set()
        lots = set()

        for img_p, xml_p in pairs:
            days.add(img_p.parent.name)
            lots.add(img_p.parent.parent.parent.name)
            tree = ET.parse(xml_p)
            root = tree.getroot()
            for space in root.findall("space"):
                total_slots += 1
                if int(space.get("occupied", 0)) == 1:
                    occupied_slots += 1
                else:
                    free_slots += 1

        stats[sname] = {
            "num_images": len(pairs),
            "num_slots": total_slots,
            "occupied": occupied_slots,
            "free": free_slots,
            "occupancy_rate": (occupied_slots / total_slots) if total_slots > 0 else 0.0,
            "days": sorted(list(days)),
            "lots": sorted(list(lots))
        }
    return stats

def print_split_summary(split_name_desc: str, splits: Dict[str, List[Tuple[Path, Path]]]):
    assert_no_data_leakage(splits)
    stats = get_split_stats(splits)

    print(f"\n=== Split Summary: {split_name_desc} ===")
    print(f"{'Split':<8} | {'Images':<8} | {'Slots':<8} | {'Occupied':<8} | {'Free':<8} | {'Occ %':<8} | {'Lots'}")
    print("-" * 75)
    for sname, s in stats.items():
        pct = s["occupancy_rate"] * 100
        lots_str = ", ".join(s["lots"])
        print(f"{sname:<8} | {s['num_images']:<8} | {s['num_slots']:<8} | {s['occupied']:<8} | {s['free']:<8} | {pct:<7.1f}% | {lots_str}")
