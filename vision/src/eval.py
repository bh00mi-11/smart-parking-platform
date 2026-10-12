"""
Evaluation module for MobileNetV3-small parking slot occupancy classifier.
Computes and reports on the test split:
- Accuracy, Precision, Recall, F1 score
- Confusion matrix (saves figure to outputs/confusion_matrix.png)
- Inference speed (ms per crop on CPU)
- Saves full metrics to outputs/metrics.json
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: Union[str, torch.device] = "cpu",
    output_dir: Union[str, Path] = "vision/outputs"
) -> Dict:
    """
    Evaluates the model on test split, measures latency, generates confusion matrix plot,
    and returns metrics dictionary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    total_latency_sec = 0.0
    total_crops = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            start_t = time.perf_counter()
            outputs = model(inputs)
            if device.type == "cuda" if isinstance(device, torch.device) else str(device).startswith("cuda"):
                torch.cuda.synchronize()
            latency = time.perf_counter() - start_t

            preds = torch.argmax(outputs, dim=1)

            total_latency_sec += latency
            total_crops += inputs.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute metrics
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1]).tolist()

    ms_per_crop = (total_latency_sec / total_crops * 1000.0) if total_crops > 0 else 0.0

    metrics = {
        "test_crops_count": int(total_crops),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": {
            "labels": ["Free (0)", "Occupied (1)"],
            "matrix": cm
        },
        "inference_latency_cpu_ms_per_crop": round(float(ms_per_crop), 2),
        "throughput_crops_per_sec": round(float(total_crops / total_latency_sec), 1) if total_latency_sec > 0 else 0.0
    }

    # Save metrics JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Eval] Saved metrics to {metrics_path}")

    # Plot confusion matrix
    cm_img_path = output_dir / "confusion_matrix.png"
    plt.figure(figsize=(6, 5))
    cm_arr = np.array(cm)
    plt.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Test Split (MobileNetV3-Small)", fontsize=12, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Free (0)", "Occupied (1)"])
    plt.yticks(tick_marks, ["Free (0)", "Occupied (1)"])

    # Annotate counts inside cells
    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            plt.text(
                j, i, format(cm_arr[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm_arr[i, j] > thresh else "black",
                fontweight="bold"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(cm_img_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Saved confusion matrix figure to {cm_img_path}")

    return metrics
