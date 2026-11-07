"""
Reproducibility Verification Script.
Reloads downloaded fold weights (or checks output predictions) against test crops,
re-evaluates with preprocess_slot_crop and MobileNetV3-small,
and verifies 100% agreement with Colab results.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

_scripts_dir = Path(__file__).resolve().parent
_vision_dir = _scripts_dir.parent
if str(_vision_dir) not in sys.path:
    sys.path.insert(0, str(_vision_dir))

from src.data_pkl import preprocess_slot_crop, parse_pklot_xml_slots


def build_eval_model(weights_path: Path, device: torch.device) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 2)
    checkpoint = torch.load(weights_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def verify_reproducibility(
    repro_json_path: Path,
    outputs_dir: Path,
    device_name: str = "cpu"
) -> bool:
    if not repro_json_path.exists():
        print(f"[REPRO ERROR] File not found: {repro_json_path}")
        print("Run the Colab training notebook or training pipeline to generate repro_check.json first.")
        return False

    with open(repro_json_path, "r", encoding="utf-8") as f:
        repro_data = json.load(f)

    device = torch.device(device_name)
    all_matched = True
    total_checked = 0
    total_matched = 0

    print("=" * 65)
    print("Reproducibility Verification: Local vs Colab Check")
    print(f"Reading reference predictions from: {repro_json_path.name}")
    print("=" * 65)

    for fold_entry in repro_data.get("folds", []):
        fold = fold_entry["fold"]
        lot = fold_entry.get("lot", f"Fold {fold}")
        samples = fold_entry["samples"]
        ckpt_path = outputs_dir / f"mobilenetv3_loo_fold{fold}.pth"

        if not ckpt_path.exists():
            print(f"[Fold {fold} - {lot}] Skipping: weights not found at {ckpt_path.name}")
            all_matched = False
            continue

        model = build_eval_model(ckpt_path, device)
        fold_matches = 0

        for s in samples:
            img_rel = s["image_path"]
            img_path = _vision_dir / img_rel
            if not img_path.exists():
                img_path = Path(img_rel)

            if not img_path.exists():
                print(f"[Warning] Sample image missing: {img_rel}")
                continue

            # Load full frame image
            bgr_img = cv2.imread(str(img_path))
            if bgr_img is None:
                continue

            rot_rect = s["rotated_rect"]
            slot_id = s.get("slot_id")
            ref_pred = int(s["pred"])

            slot_info = {
                "id": slot_id,
                "rotated_rect": rot_rect,
                "contour": s.get("contour", [])
            }

            tensor_crop = preprocess_slot_crop(bgr_img, slot_info)
            tensor_crop = tensor_crop.unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor_crop)
                local_pred = int(torch.argmax(logits, dim=1).item())

            total_checked += 1
            if local_pred == ref_pred:
                fold_matches += 1
                total_matched += 1
            else:
                print(f"  [MISMATCH] Fold {fold} Slot {slot_id}: Colab={ref_pred}, Local={local_pred}")

        n_samples = len(samples)
        pct = (fold_matches / n_samples * 100.0) if n_samples > 0 else 0.0
        print(f"Fold {fold} ({lot:<6}): {fold_matches}/{n_samples} matched ({pct:.1f}%)")
        if fold_matches != n_samples:
            all_matched = False

    print("-" * 65)
    overall_pct = (total_matched / total_checked * 100.0) if total_checked > 0 else 0.0
    print(f"Overall Reproducibility: {total_matched}/{total_checked} matched ({overall_pct:.2f}%)")
    if all_matched and total_checked > 0:
        print("[SUCCESS] 100% Bitwise / Decision Parity with Colab results!")
    else:
        print("[NOTICE] Some samples or weights could not be fully verified locally.")

    return all_matched and total_checked > 0


def main():
    root = Path(__file__).resolve().parent.parent
    outputs_dir = root / "outputs"
    repro_json = outputs_dir / "repro_check.json"

    success = verify_reproducibility(repro_json, outputs_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
