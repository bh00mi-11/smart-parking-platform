"""
Run full evaluation of MobileNetV3 (LOO weights) vs YOLO11n mapped to slots
across all sampled frames in vision/data/PKLot (~154 frames per lot).
"""

import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

_curr_dir = Path(__file__).resolve().parent
_vision_dir = _curr_dir.parent
if str(_vision_dir) not in sys.path:
    sys.path.insert(0, str(_vision_dir))

from src.classify_slots import classify_slots
from src.train_classifier import build_model
from src.data_pkl import parse_pklot_xml_slots
from src.yolo_count import count_vehicles, map_detections_to_slots


def load_fold_model(weights_path: Path, device: torch.device) -> torch.nn.Module:
    model = build_model(num_classes=2)
    checkpoint = torch.load(weights_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def main():
    device = torch.device("cpu")
    data_dir = _vision_dir / "data" / "PKLot"
    outputs_dir = _vision_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    lot_to_fold = {
        "PUCPR": 0,
        "UFPR04": 1,
        "UFPR05": 2
    }

    fold_models = {}
    for lot, fold_idx in lot_to_fold.items():
        weights_path = outputs_dir / f"mobilenetv3_loo_fold{fold_idx}.pth"
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing fold weights: {weights_path}")
        fold_models[lot] = load_fold_model(weights_path, device)

    # Warmup YOLO
    sample_first = sorted(list(data_dir.rglob("*.jpg")))[0]
    count_vehicles(str(sample_first))

    results = {
        "evaluation_type": "leave_one_lot_out_classifier_vs_yolo_full_sample",
        "weights_status": "leave_one_lot_out_weights (fold0 for PUCPR, fold1 for UFPR04, fold2 for UFPR05)",
        "per_lot_comparison": {}
    }

    for lot, fold_idx in lot_to_fold.items():
        print(f"\n=======================================================")
        print(f"Evaluating {lot} (LOO Fold {fold_idx})...")
        print(f"=======================================================")
        model = fold_models[lot]
        lot_dir = data_dir / lot
        jpgs = sorted(list(lot_dir.rglob("*.jpg")))
        pairs = [(p, p.with_suffix(".xml")) for p in jpgs if p.with_suffix(".xml").exists()]
        print(f"Found {len(pairs)} frames for {lot}")

        y_true = []
        y_pred_cls = []
        y_pred_yolo = []

        total_cls_time = 0.0
        total_yolo_time = 0.0
        total_slots_count = 0

        for idx, (img_p, xml_p) in enumerate(pairs, 1):
            _, slots = parse_pklot_xml_slots(xml_p)
            n_slots = len(slots)
            total_slots_count += n_slots

            for s in slots:
                y_true.append(s["occupied"])

            # 1. Classifier
            t0 = time.time()
            cls_res = classify_slots(str(img_p), slots, model=model, device=device)
            t1 = time.time()
            total_cls_time += (t1 - t0)
            y_pred_cls.extend([r["occupied"] for r in cls_res])

            # 2. YOLO
            t2 = time.time()
            _, dets = count_vehicles(str(img_p), conf=0.25)
            yolo_res = map_detections_to_slots(dets, slots)
            t3 = time.time()
            total_yolo_time += (t3 - t2)
            y_pred_yolo.extend([r["occupied"] for r in yolo_res])

            if idx % 30 == 0 or idx == len(pairs):
                print(f"  Processed {idx}/{len(pairs)} frames ({total_slots_count} slots)...")

        def _metrics(yt, yp, total_time):
            acc = float(accuracy_score(yt, yp))
            prec = float(precision_score(yt, yp, zero_division=0))
            rec = float(recall_score(yt, yp, zero_division=0))
            f1 = float(f1_score(yt, yp, zero_division=0))
            ms_per_slot = (total_time / max(1, len(yt))) * 1000.0
            return {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "ms_per_slot": round(ms_per_slot, 3)
            }

        cls_metrics = _metrics(y_true, y_pred_cls, total_cls_time)
        yolo_metrics = _metrics(y_true, y_pred_yolo, total_yolo_time)

        print(f"Results for {lot}:")
        print(f"  MobileNetV3: Acc={cls_metrics['accuracy']:.4f}, Prec={cls_metrics['precision']:.4f}, Rec={cls_metrics['recall']:.4f}, F1={cls_metrics['f1_score']:.4f}, Speed={cls_metrics['ms_per_slot']:.2f} ms/slot")
        print(f"  YOLO11n:     Acc={yolo_metrics['accuracy']:.4f}, Prec={yolo_metrics['precision']:.4f}, Rec={yolo_metrics['recall']:.4f}, F1={yolo_metrics['f1_score']:.4f}, Speed={yolo_metrics['ms_per_slot']:.2f} ms/slot")

        results["per_lot_comparison"][lot] = {
            "num_images": len(pairs),
            "total_annotated_slots": total_slots_count,
            "weights_used": f"mobilenetv3_loo_fold{fold_idx}.pth",
            "metrics": {
                "mobilenetv3": cls_metrics,
                "yolo11n_mapped": yolo_metrics
            }
        }

    out_file = outputs_dir / "classifier_vs_yolo_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved updated results to: {out_file}")


if __name__ == "__main__":
    main()

