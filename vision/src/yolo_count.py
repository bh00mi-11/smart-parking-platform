"""
Vehicle detection and counting module using pretrained YOLO11n.
Designed for open zones / unmanaged parking areas where slot boundaries are not marked.
Detects COCO vehicle classes: car, motorcycle, bus, and truck.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union
import cv2
import numpy as np
from ultralytics import YOLO

# COCO vehicle class IDs for YOLO:
# 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Singleton / lazy-loaded model instance
_MODEL = None

def get_yolo_model(model_name: str = "yolo11n.pt") -> YOLO:
    global _MODEL
    if _MODEL is None:
        _MODEL = YOLO(model_name)
    return _MODEL

def count_vehicles(
    image_path: Union[str, Path],
    conf: float = 0.25,
    save_annotated_path: Union[str, Path, None] = None
) -> Tuple[int, List[Dict]]:
    """
    Counts vehicles in an image using YOLO11n.

    Args:
        image_path: Path to the target image file.
        conf: Confidence threshold (default: 0.25).
        save_annotated_path: Optional path to save visual detection bounding boxes.

    Returns:
        count: Total number of detected vehicles (car, motorcycle, bus, truck).
        detections: List of detection dicts with keys:
                    [box (x1, y1, x2, y2), class_id, class_name, confidence].
    """
    model = get_yolo_model()
    image_path_str = str(image_path)

    results = model(image_path_str, conf=conf, verbose=False)[0]

    detections = []
    vehicle_boxes = []

    for box in results.boxes:
        cls_id = int(box.cls.item())
        confidence = float(box.conf.item())
        xyxy = box.xyxy[0].cpu().numpy().tolist()

        if cls_id in VEHICLE_CLASS_IDS:
            cls_name = VEHICLE_CLASS_IDS[cls_id]
            detections.append({
                "box": [round(coord, 2) for coord in xyxy],
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(confidence, 4)
            })
            vehicle_boxes.append((xyxy, cls_name, confidence))

    count = len(detections)

    if save_annotated_path is not None:
        save_annotated_path = Path(save_annotated_path)
        save_annotated_path.parent.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(image_path_str)
        if img is not None:
            # Draw detections
            for (x1, y1, x2, y2), cname, score in vehicle_boxes:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{cname} {score:.2f}"
                cv2.putText(
                    img, label, (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA
                )

            # Draw summary counter header
            header_text = f"YOLO11n Vehicle Count: {count}"
            cv2.rectangle(img, (10, 10), (320, 50), (0, 0, 0), -1)
            cv2.putText(
                img, header_text, (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA
            )
            cv2.imwrite(str(save_annotated_path), img)

    return count, detections

def evaluate_on_pklot_samples(
    images_with_xmls: List[Tuple[Path, Path]],
    conf: float = 0.25,
    output_dir: Union[str, Path] = "vision/outputs"
) -> List[Dict]:
    """
    Runs YOLO vehicle counting across a set of PKLot images,
    compares against true occupied slot count parsed from XMLs,
    and returns a summary table.
    """
    import xml.etree.ElementTree as ET

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = output_dir / "annotated_yolo"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    results_table = []

    for img_p, xml_p in images_with_xmls:
        # 1. Parse true occupied count
        tree = ET.parse(xml_p)
        root = tree.getroot()
        true_occupied = sum(
            1 for space in root.findall("space")
            if int(space.get("occupied", 0)) == 1
        )
        total_slots = len(root.findall("space"))

        # 2. Run YOLO11n count
        annotated_path = annotated_dir / f"pred_{img_p.name}"
        pred_count, detections = count_vehicles(
            image_path=img_p,
            conf=conf,
            save_annotated_path=annotated_path
        )

        abs_error = abs(pred_count - true_occupied)

        lot_name = img_p.parents[2].name if len(img_p.parents) >= 3 else "Unknown"
        weather = img_p.parents[1].name if len(img_p.parents) >= 2 else "Unknown"

        record = {
            "image": img_p.name,
            "lot": lot_name,
            "weather": weather,
            "total_slots": total_slots,
            "true_occupied": true_occupied,
            "predicted_count": pred_count,
            "abs_error": abs_error,
            "annotated_path": str(annotated_path.relative_to(output_dir.parent))
        }
        results_table.append(record)

    return results_table
