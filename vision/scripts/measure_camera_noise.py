"""
Script to measure real camera noise across sampled PKLot frames.
Evaluates each lot with its corresponding Leave-One-Lot-Out (LOO) model weights:
- PUCPR  -> fold0 (mobilenetv3_loo_fold0.pth)
- UFPR04 -> fold1 (mobilenetv3_loo_fold1.pth)
- UFPR05 -> fold2 (mobilenetv3_loo_fold2.pth)

For each frame:
- Computes true occupied count from XML ground truth
- Classifies each slot using classify_slots / LOO model
- Computes error = predicted_occupied - true_occupied

Saves summary statistics to vision/outputs/camera_noise.json:
- bias (mean error)
- standard deviation (std)
- mean absolute error (mae)
- frame counts
"""

import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import torch

_scripts_dir = Path(__file__).resolve().parent
_vision_dir = _scripts_dir.parent
if str(_vision_dir) not in sys.path:
    sys.path.insert(0, str(_vision_dir))

from src.classify_slots import classify_slots
from src.train_classifier import build_model
from src.data_pkl import parse_pklot_xml_slots


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

    all_jpgs = sorted(list(data_dir.rglob("*.jpg")))
    pairs = [(p, p.with_suffix(".xml")) for p in all_jpgs if p.with_suffix(".xml").exists()]

    print("=" * 65)
    print("MEASURING REAL CAMERA NOISE ON SAMPLED PKLOT FRAMES")
    print(f"Total sampled frames found: {len(pairs)}")
    print("=" * 65)

    lot_errors = {lot: [] for lot in lot_to_fold}
    lot_frame_counts = {lot: 0 for lot in lot_to_fold}
    lot_true_occupied = {lot: [] for lot in lot_to_fold}
    lot_pred_occupied = {lot: [] for lot in lot_to_fold}

    for idx, (img_path, xml_path) in enumerate(pairs, 1):
        rel = img_path.relative_to(data_dir)
        lot = rel.parts[0]
        if lot not in lot_to_fold:
            continue

        model = fold_models[lot]
        _, slots = parse_pklot_xml_slots(str(xml_path))
        if not slots:
            continue

        true_occ = sum(1 for s in slots if s.get("occupied") == 1)

        # Prepare slot_boxes for classify_slots
        slot_boxes = []
        for s in slots:
            slot_boxes.append({
                "slot_id": s.get("slot_id"),
                "rotated_rect": s.get("rotated_rect"),
                "contour": s.get("contour", [])
            })

        preds = classify_slots(str(img_path), slot_boxes, model=model, device=device)
        pred_occ = sum(1 for p in preds if p.get("occupied") == 1)

        error = pred_occ - true_occ  # positive means over-predicting occupied (bias)
        lot_errors[lot].append(error)
        lot_true_occupied[lot].append(true_occ)
        lot_pred_occupied[lot].append(pred_occ)
        lot_frame_counts[lot] += 1

        if idx % 50 == 0 or idx == len(pairs):
            print(f"[{idx}/{len(pairs)}] Processed frame: {rel} | True: {true_occ}, Pred: {pred_occ}, Diff: {error:+d}")

    summary = {
        "description": "Measured camera observation noise using Leave-One-Lot-Out (LOO) models on unseen PKLot test lots",
        "lots": {}
    }

    all_errors = []
    print("\n" + "=" * 65)
    print("CAMERA NOISE SUMMARY BY LOT")
    print("=" * 65)

    for lot in sorted(lot_to_fold.keys()):
        errs = np.array(lot_errors[lot], dtype=np.float64)
        all_errors.extend(errs.tolist())
        bias = float(np.mean(errs)) if len(errs) > 0 else 0.0
        std = float(np.std(errs)) if len(errs) > 0 else 0.0
        mae = float(np.mean(np.abs(errs))) if len(errs) > 0 else 0.0
        frames = lot_frame_counts[lot]

        summary["lots"][lot] = {
            "loo_fold": lot_to_fold[lot],
            "frame_count": frames,
            "bias_mean_error": round(bias, 4),
            "std_deviation": round(std, 4),
            "mae": round(mae, 4)
        }
        print(f"Lot {lot:<8} (Folds: {lot_to_fold[lot]}, Frames: {frames}): Bias={bias:+.3f}, Std={std:.3f}, MAE={mae:.3f}")

    all_errs_arr = np.array(all_errors, dtype=np.float64)
    overall_bias = float(np.mean(all_errs_arr))
    overall_std = float(np.std(all_errs_arr))
    overall_mae = float(np.mean(np.abs(all_errs_arr)))

    summary["overall"] = {
        "total_frames": len(all_errors),
        "bias_mean_error": round(overall_bias, 4),
        "std_deviation": round(overall_std, 4),
        "mae": round(overall_mae, 4)
    }
    print("-" * 65)
    print(f"OVERALL  (Total Frames: {len(all_errors)}): Bias={overall_bias:+.3f}, Std={overall_std:.3f}, MAE={overall_mae:.3f}")
    print("=" * 65)

    out_file = outputs_dir / "camera_noise.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Successfully saved camera noise statistics to {out_file}")


if __name__ == "__main__":
    main()
