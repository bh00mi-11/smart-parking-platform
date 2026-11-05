"""
Inspection script for PKLot and CNRPark datasets.
Computes real dataset statistics:
- Number of images and slot annotations per lot, per weather condition, and per day.
- Label balance (occupied vs free).
- Image resolutions.
- Extracts and saves a grid of 12 sample slot crops (6 occupied, 6 free) to outputs/sample_crops_grid.png.
"""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def parse_pklot_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lot_id = root.get("id", "unknown")
    slots = []
    for space in root.findall("space"):
        s_id = space.get("id")
        occ = int(space.get("occupied", 0))

        # rotatedRect
        rot = space.find("rotatedRect")
        rot_data = None
        if rot is not None:
            c = rot.find("center")
            s = rot.find("size")
            a = rot.find("angle")
            if c is not None and s is not None and a is not None:
                rot_data = {
                    "cx": float(c.get("x")),
                    "cy": float(c.get("y")),
                    "w": float(s.get("w")),
                    "h": float(s.get("h")),
                    "angle": float(a.get("d"))
                }

        # contour points
        cnt = space.find("contour")
        points = []
        if cnt is not None:
            for pt in cnt.findall("point"):
                points.append((float(pt.get("x")), float(pt.get("y"))))

        slots.append({
            "id": s_id,
            "occupied": occ,
            "rotated_rect": rot_data,
            "contour": points
        })
    return lot_id, slots

def crop_slot(image: np.ndarray, slot_data: dict, crop_size=(224, 224)) -> np.ndarray:
    """
    Extracts slot crop handling rotation accurately via cv2.getRotationMatrix2D
    and warpAffine, or falls back to contour bounding box.
    """
    rot = slot_data.get("rotated_rect")
    if rot is not None and rot["w"] > 0 and rot["h"] > 0:
        center = (rot["cx"], rot["cy"])
        size = (int(rot["w"]), int(rot["h"]))
        angle = rot["angle"]

        # Calculate bounding box of rotated rect to avoid cutting corners
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        h_img, w_img = image.shape[:2]
        rotated_img = cv2.warpAffine(image, M, (w_img, h_img), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Crop upright box
        x = int(center[0] - size[0] / 2)
        y = int(center[1] - size[1] / 2)
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(size[0], w_img - x))
        h = max(1, min(size[1], h_img - y))
        crop = rotated_img[y:y+h, x:x+w]
    elif slot_data.get("contour"):
        pts = np.array(slot_data["contour"], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        crop = image[y:y+h, x:x+w]
    else:
        crop = np.zeros((crop_size[1], crop_size[0], 3), dtype=np.uint8)

    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        return cv2.resize(image[:64, :64], crop_size)
    return cv2.resize(crop, crop_size, interpolation=cv2.INTER_LINEAR)

def inspect_datasets():
    base_dir = Path(__file__).resolve().parent.parent
    pklot_dir = base_dir / "data" / "PKLot"
    outputs_dir = base_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PKLot REAL DATA INSPECTION REPORT")
    print("=" * 60)

    # Discover pairs
    img_files = sorted(list(pklot_dir.rglob("*.jpg")))
    print(f"Total full-frame PKLot images found: {len(img_files)}")

    stats_by_lot = defaultdict(lambda: {"images": 0, "slots": 0, "occupied": 0, "free": 0})
    stats_by_weather = defaultdict(lambda: {"images": 0, "slots": 0, "occupied": 0, "free": 0})
    stats_by_day = defaultdict(lambda: {"images": 0, "slots": 0, "occupied": 0, "free": 0})
    resolutions = set()

    occupied_crops = []
    free_crops = []

    for img_p in img_files:
        xml_p = img_p.with_suffix(".xml")
        if not xml_p.exists():
            continue

        parts = img_p.relative_to(pklot_dir).parts
        lot = parts[0]
        weather = parts[1]
        day = parts[2]

        img_bgr = cv2.imread(str(img_p))
        if img_bgr is not None:
            resolutions.add((img_bgr.shape[1], img_bgr.shape[0]))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = None

        lot_id, slots = parse_pklot_xml(xml_p)

        stats_by_lot[lot]["images"] += 1
        stats_by_lot[lot]["slots"] += len(slots)

        stats_by_weather[weather]["images"] += 1
        stats_by_weather[weather]["slots"] += len(slots)

        stats_by_day[day]["images"] += 1
        stats_by_day[day]["slots"] += len(slots)

        for slot in slots:
            occ = slot["occupied"]
            if occ == 1:
                stats_by_lot[lot]["occupied"] += 1
                stats_by_weather[weather]["occupied"] += 1
                stats_by_day[day]["occupied"] += 1
                if img_rgb is not None and len(occupied_crops) < 15:
                    c = crop_slot(img_rgb, slot)
                    occupied_crops.append(c)
            else:
                stats_by_lot[lot]["free"] += 1
                stats_by_weather[weather]["free"] += 1
                stats_by_day[day]["free"] += 1
                if img_rgb is not None and len(free_crops) < 15:
                    c = crop_slot(img_rgb, slot)
                    free_crops.append(c)

    print("\n--- Statistics by Parking Lot ---")
    print(f"{'Lot':<12} | {'Images':<8} | {'Total Slots':<12} | {'Occupied':<10} | {'Free':<10} | {'Occupancy %':<12}")
    print("-" * 72)
    for lot, s in sorted(stats_by_lot.items()):
        occ_pct = (s["occupied"] / s["slots"] * 100) if s["slots"] > 0 else 0
        print(f"{lot:<12} | {s['images']:<8} | {s['slots']:<12} | {s['occupied']:<10} | {s['free']:<10} | {occ_pct:.1f}%")

    print("\n--- Statistics by Weather Condition ---")
    print(f"{'Weather':<12} | {'Images':<8} | {'Total Slots':<12} | {'Occupied':<10} | {'Free':<10} | {'Occupancy %':<12}")
    print("-" * 72)
    for w, s in sorted(stats_by_weather.items()):
        occ_pct = (s["occupied"] / s["slots"] * 100) if s["slots"] > 0 else 0
        print(f"{w:<12} | {s['images']:<8} | {s['slots']:<12} | {s['occupied']:<10} | {s['free']:<10} | {occ_pct:.1f}%")

    print("\n--- Statistics by Day ---")
    print(f"{'Day':<12} | {'Images':<8} | {'Total Slots':<12} | {'Occupied':<10} | {'Free':<10} | {'Occupancy %':<12}")
    print("-" * 72)
    for d, s in sorted(stats_by_day.items()):
        occ_pct = (s["occupied"] / s["slots"] * 100) if s["slots"] > 0 else 0
        print(f"{d:<12} | {s['images']:<8} | {s['slots']:<12} | {s['occupied']:<10} | {s['free']:<10} | {occ_pct:.1f}%")

    print(f"\nImage resolutions found (W x H): {resolutions}")

    # Save 12 sample slot crops grid (6 occupied, 6 free)
    grid_img_path = outputs_dir / "sample_crops_grid.png"
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    fig.suptitle("PKLot Sample Slot Crops: 6 Occupied (top) vs 6 Free (bottom)", fontsize=14, fontweight="bold")

    selected_occ = occupied_crops[:6]
    selected_free = free_crops[:6]

    for idx in range(6):
        ax = axes[idx // 4, idx % 4]
        ax.imshow(selected_occ[idx])
        ax.set_title(f"Occupied #{idx+1}", color="crimson", fontweight="bold")
        ax.axis("off")

    for idx in range(6):
        flat_idx = 6 + idx
        ax = axes[flat_idx // 4, flat_idx % 4]
        ax.imshow(selected_free[idx])
        ax.set_title(f"Free #{idx+1}", color="forestgreen", fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(grid_img_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved 12-sample grid to: {grid_img_path}")

    # Inspect CNRPark
    cnr_dir = base_dir / "data" / "CNRPark"
    if cnr_dir.exists():
        busy_files = list(cnr_dir.rglob("busy/*.jpg"))
        free_files = list(cnr_dir.rglob("free/*.jpg"))
        print("\n" + "=" * 60)
        print("CNRPark REAL DATA INSPECTION REPORT")
        print("=" * 60)
        print(f"Total patch files: {len(busy_files) + len(free_files)}")
        print(f"Busy (Occupied) patches: {len(busy_files)}")
        print(f"Free patches:            {len(free_files)}")
        if busy_files:
            sample_im = Image.open(busy_files[0])
            print(f"CNRPark patch resolution: {sample_im.size}")

if __name__ == "__main__":
    inspect_datasets()
