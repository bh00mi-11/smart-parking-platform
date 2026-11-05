"""
Vehicle detection and counting module using pretrained YOLO11n.
Designed for open zones / unmanaged parking areas where slot boundaries are not marked,
as well as cross-evaluating open-zone vehicle detection against marked slot layouts.

Detects COCO vehicle classes: car, motorcycle, bus, and truck.
Uses native OpenCV geometric utilities (cv2.pointPolygonTest) for robust polygon inclusion.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from ultralytics import YOLO

# COCO vehicle class IDs for YOLO:
# 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

_MODEL: Optional[YOLO] = None


def get_yolo_model(model_name: str = "yolo11n.pt") -> YOLO:
    global _MODEL
    if _MODEL is None:
        _MODEL = YOLO(model_name)
    return _MODEL


def count_vehicles(
    image: Union[str, Path, np.ndarray],
    conf: float = 0.25,
    imgsz: Optional[int] = None,
    save_annotated_path: Union[str, Path, None] = None
) -> Tuple[int, List[Dict]]:
    """
    Counts vehicles in an image using YOLO11n.
    """
    model = get_yolo_model()
    kwargs = {"conf": conf, "verbose": False}
    if imgsz is not None:
        kwargs["imgsz"] = imgsz

    if isinstance(image, (str, Path)):
        img_input = str(image)
        img_arr = cv2.imread(img_input)
    else:
        img_input = image
        img_arr = image.copy()

    results = model(img_input, **kwargs)[0]

    detections = []
    vehicle_boxes = []

    for box in results.boxes:
        cls_id = int(box.cls.item())
        confidence = float(box.conf.item())
        xyxy = box.xyxy[0].cpu().numpy().tolist()

        if cls_id in VEHICLE_CLASS_IDS:
            cls_name = VEHICLE_CLASS_IDS[cls_id]
            cx = (xyxy[0] + xyxy[2]) / 2.0
            cy = (xyxy[1] + xyxy[3]) / 2.0
            det = {
                "box": [round(c, 2) for c in xyxy],
                "center": (round(cx, 2), round(cy, 2)),
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(confidence, 4)
            }
            detections.append(det)
            vehicle_boxes.append((xyxy, cls_name, confidence))

    count = len(detections)

    if save_annotated_path is not None and img_arr is not None:
        save_annotated_path = Path(save_annotated_path)
        save_annotated_path.parent.mkdir(parents=True, exist_ok=True)
        draw_img = img_arr.copy()
        for (x1, y1, x2, y2), cname, score in vehicle_boxes:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cname} {score:.2f}"
            cv2.putText(
                draw_img, label, (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA
            )

        header_text = f"YOLO11n Vehicle Count: {count}"
        cv2.rectangle(draw_img, (10, 10), (320, 50), (0, 0, 0), -1)
        cv2.putText(
            draw_img, header_text, (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA
        )
        cv2.imwrite(str(save_annotated_path), draw_img)

    return count, detections


def count_in_polygons(
    image: Union[str, Path, np.ndarray],
    polygons: List[List[Union[List[float], Tuple[float, float]]]],
    conf: float = 0.25,
    imgsz: Optional[int] = None
) -> Tuple[int, List[Dict]]:
    """
    Counts vehicles whose bounding box center falls inside any of the specified zone polygons.
    Uses cv2.pointPolygonTest for containment.
    """
    _, all_detections = count_vehicles(image, conf=conf, imgsz=imgsz)

    valid_contours = []
    for poly in polygons:
        if len(poly) >= 3:
            pts = np.array(poly, dtype=np.float32).reshape((-1, 1, 2))
            valid_contours.append(pts)

    if not valid_contours:
        return len(all_detections), all_detections

    inside_detections = []
    for det in all_detections:
        cx, cy = det["center"]
        is_inside = False
        for contour in valid_contours:
            dist = cv2.pointPolygonTest(contour, (float(cx), float(cy)), measureDist=False)
            if dist >= 0:
                is_inside = True
                break
        if is_inside:
            inside_detections.append(det)

    return len(inside_detections), inside_detections


def map_detections_to_slots(
    detections: List[Dict],
    slots: List[Dict],
    iou_threshold: float = 0.15
) -> List[Dict]:
    """
    Maps YOLO vehicle bounding box detections to slot polygons.
    A slot is labelled occupied (1) if:
    - Any detected vehicle's center falls inside the slot polygon (via cv2.pointPolygonTest), OR
    - Bounding box intersection overlap exceeds threshold.
    """
    slot_results = []

    for slot in slots:
        slot_id = slot["slot_id"]
        points = slot.get("points")
        occupied = 0
        best_conf = 0.0

        if points is not None and len(points) >= 3:
            pts_np = np.array(points, dtype=np.float32).reshape((-1, 1, 2))
            # Slot bounding rect
            sx, sy, sw, sh = cv2.boundingRect(pts_np)

            for det in detections:
                cx, cy = det["center"]
                dist = cv2.pointPolygonTest(pts_np, (float(cx), float(cy)), measureDist=False)
                if dist >= 0:
                    occupied = 1
                    best_conf = max(best_conf, det["confidence"])
                    break

                # Overlap test with slot bounding box
                bx1, by1, bx2, by2 = det["box"]
                ix1 = max(sx, bx1)
                iy1 = max(sy, by1)
                ix2 = min(sx + sw, bx2)
                iy2 = min(sy + sh, by2)

                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    slot_area = max(sw * sh, 1)
                    if (inter_area / slot_area) >= iou_threshold:
                        occupied = 1
                        best_conf = max(best_conf, det["confidence"])
                        break

        slot_results.append({
            "slot_id": slot_id,
            "occupied": occupied,
            "confidence": best_conf if occupied == 1 else 0.85
        })

    return slot_results
