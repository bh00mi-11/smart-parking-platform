"""
Demonstration script showing end-to-end processing:
Raw camera frame -> Slot classification / YOLO counting -> Probabilistic Fusion -> OccupancyUpdate JSON.
Saves annotated images to vision/outputs/.
"""

import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import cv2

# Ensure vision root in path
_example_dir = Path(__file__).resolve().parent
_vision_dir = _example_dir.parent
if str(_vision_dir) not in sys.path:
    sys.path.insert(0, str(_vision_dir))

from src.camera_adapter import load_classifier_model, process_frame
from src.classify_slots import classify_slots
from src.yolo_count import count_vehicles


def parse_pklot_xml_slots(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    slots = []
    for space in root.findall("space"):
        s_id = space.get("id", "0")
        contour = space.find("contour")
        points = []
        if contour is not None:
            for pt in contour.findall("point"):
                points.append([float(pt.get("x")), float(pt.get("y"))])
        slots.append({"slot_id": s_id, "points": points})
    return slots


def run_demo():
    output_dir = _vision_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = _vision_dir / "data" / "PKLot"
    sample_images = sorted(list(data_dir.rglob("*.jpg")))

    if not sample_images:
        print("[Demo Error] No sample images found in vision/data/PKLot.")
        return

    print("=" * 65)
    print("DEMO: CAMERA FRAME TO OCCUPANCY UPDATE")
    print("=" * 65)

    # 1. Managed Lot Demo
    managed_img_p = sample_images[0]
    managed_xml_p = managed_img_p.with_suffix(".xml")

    if managed_xml_p.exists():
        slots = parse_pklot_xml_slots(managed_xml_p)
        zone_config = {
            "zone_type": "managed",
            "capacity": len(slots),
            "slots": slots
        }

        print(f"\n[1] Processing Managed Lot Frame: {managed_img_p.name} ({len(slots)} slots)")

        # Attempt to load trained classifier or fall back to mock for demo without failing
        try:
            model = load_classifier_model(device="cpu")
        except FileNotFoundError:
            print("  (Trained weights not found; using initialized MobileNetV3 for demonstration)")
            from src.train_classifier import build_model
            model = build_model(num_classes=2)
            model.eval()

        update_managed = process_frame(
            facility_id="FAC-PKLOT",
            zone_id="ZONE-MANAGED",
            image=str(managed_img_p),
            zone_config=zone_config,
            classifier_model=model,
            device="cpu"
        )

        print("\n--- Managed Lot OccupancyUpdate JSON ---")
        print(json.dumps(update_managed.model_dump(mode="json"), indent=2))

        # Annotate image
        raw_img = cv2.imread(str(managed_img_p))
        if raw_img is not None:
            cls_res = classify_slots(raw_img, slots, model=model, device="cpu")
            for r, slot in zip(cls_res, slots):
                pts = slot.get("points")
                if pts and len(pts) >= 3:
                    pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                    color = (0, 0, 255) if r["occupied"] == 1 else (0, 255, 0)
                    cv2.polylines(raw_img, [pts_np], isClosed=True, color=color, thickness=2)
            annotated_out = output_dir / f"annotated_managed_{managed_img_p.name}"
            cv2.imwrite(str(annotated_out), raw_img)
            print(f"  -> Saved annotated managed frame to: {annotated_out.name}")

    # 2. Open Zone Demo (using YOLO vehicle counting)
    print(f"\n[2] Processing Open Zone Frame: {managed_img_p.name}")
    open_zone_config = {
        "zone_type": "open",
        "capacity": 50
    }
    update_open = process_frame(
        facility_id="FAC-STREET",
        zone_id="ZONE-OPEN",
        image=str(managed_img_p),
        zone_config=open_zone_config,
        device="cpu"
    )
    print("\n--- Open Zone OccupancyUpdate JSON ---")
    print(json.dumps(update_open.model_dump(mode="json"), indent=2))
    print("=" * 65)


if __name__ == "__main__":
    import numpy as np
    run_demo()

