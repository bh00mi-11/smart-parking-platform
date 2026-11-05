"""
Per-lot evaluation for YOLO11n vehicle counting across PKLot lots:
Evaluates PUCPR, UFPR04, and UFPR05 separately (never one overall average),
at default imgsz (640) and imgsz=1280.
Compares predicted vehicle count vs true occupied-slot count parsed from XMLs.
"""

from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

_src_dir = Path(__file__).resolve().parent
_vision_dir = _src_dir.parent
if str(_vision_dir) not in sys.path:
    sys.path.insert(0, str(_vision_dir))

from src.yolo_count import count_vehicles


def evaluate_yolo_per_lot(output_dir: Path = _vision_dir / "outputs") -> Dict:
    data_dir = _vision_dir / "data" / "PKLot"
    output_dir.mkdir(parents=True, exist_ok=True)

    lots = ["PUCPR", "UFPR04", "UFPR05"]
    lot_samples: Dict[str, List[Tuple[Path, Path]]] = defaultdict(list)

    for lot in lots:
        lot_path = data_dir / lot
        if not lot_path.exists():
            continue
        imgs = sorted(list(lot_path.rglob("*.jpg")))
        for im in imgs:
            xml_p = im.with_suffix(".xml")
            if xml_p.exists():
                lot_samples[lot].append((im, xml_p))

    # Evaluate 2 diverse samples per lot to keep runtime fast and report real numbers
    eval_results = {}

    print("=" * 70)
    print("YOLO11n PER-LOT VEHICLE COUNTING EVALUATION")
    print("PUCPR: Overhead view with distant small vehicles (known weakness for open zones)")
    print("UFPR04 & UFPR05: Angled camera perspective with larger vehicle signatures")
    print("=" * 70)

    for lot in lots:
        samples = lot_samples.get(lot, [])
        if not samples:
            print(f"[Warning] No samples found for lot {lot}")
            continue

        selected = [samples[0], samples[min(len(samples) - 1, len(samples) // 2)]]
        eval_results[lot] = {}

        for imgsz in [640, 1280]:
            lot_errors = []
            frame_records = []

            for im_p, xml_p in selected:
                tree = ET.parse(xml_p)
                root = tree.getroot()
                true_occ = sum(1 for s in root.findall("space") if int(s.get("occupied", 0)) == 1)
                total_s = len(root.findall("space"))

                pred_cnt, _ = count_vehicles(str(im_p), conf=0.25, imgsz=imgsz)
                abs_err = abs(pred_cnt - true_occ)
                lot_errors.append(abs_err)

                frame_records.append({
                    "image": im_p.name,
                    "total_slots": total_s,
                    "true_occupied": true_occ,
                    "pred_count": pred_cnt,
                    "abs_error": abs_err
                })

            mean_mae = float(sum(lot_errors) / len(lot_errors)) if lot_errors else 0.0
            eval_results[lot][f"imgsz_{imgsz}"] = {
                "mean_abs_error": round(mean_mae, 2),
                "frames": frame_records
            }

            print(f"Lot: {lot:<8} | imgsz: {imgsz:<5} | Mean Absolute Error: {mean_mae:.2f} vehicles")

    with open(output_dir / "yolo_per_lot_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    return eval_results


if __name__ == "__main__":
    evaluate_yolo_per_lot()

