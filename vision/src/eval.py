"""
Comprehensive evaluation module for MobileNetV3 slot classifier and YOLO comparison.
Includes:
- Accuracy, Precision, Recall, F1 score
- Confusion matrix and reliability diagram plots
- Expected Calibration Error (ECE)
- Temperature scaling calibration fitted on validation split
- Per-slot evaluation comparing MobileNetV3 vs YOLO11n on identical test slots
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .yolo_count import count_vehicles, map_detections_to_slots


def accuracy_score(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred)) if len(y_true) > 0 else 0.0


def precision_score(y_true, y_pred, zero_division=0) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else float(zero_division)


def recall_score(y_true, y_pred, zero_division=0) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else float(zero_division)


def f1_score(y_true, y_pred, zero_division=0) -> float:
    prec = precision_score(y_true, y_pred, zero_division=zero_division)
    rec = recall_score(y_true, y_pred, zero_division=zero_division)
    if (prec + rec) == 0:
        return float(zero_division)
    return float(2 * prec * rec / (prec + rec))


def confusion_matrix(y_true, y_pred, labels=[0, 1]) -> np.ndarray:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in labels and p in labels:
            cm[labels.index(t), labels.index(p)] += 1
    return cm


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> Tuple[float, List[Dict]]:
    """
    Computes Expected Calibration Error (ECE) for binary classification probabilities.
    Args:
        probs: Array of predicted positive class probabilities in [0, 1].
        targets: Array of binary ground truth labels in {0, 1}.
        n_bins: Number of equal-width bins in [0, 1].
    Returns:
        ece: Scalar expected calibration error.
        bin_details: List of dicts with statistics per bin.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    bin_details = []

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (probs >= low) & (probs <= high)
        else:
            in_bin = (probs >= low) & (probs < high)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(targets[in_bin]))
            bin_conf = float(np.mean(probs[in_bin]))
            weight = bin_count / n
            ece += weight * abs(bin_acc - bin_conf)
            bin_details.append({
                "bin": i,
                "range": [round(low, 2), round(high, 2)],
                "count": bin_count,
                "confidence": round(bin_conf, 4),
                "accuracy": round(bin_acc, 4)
            })
        else:
            bin_details.append({
                "bin": i,
                "range": [round(low, 2), round(high, 2)],
                "count": 0,
                "confidence": round((low + high) / 2.0, 4),
                "accuracy": 0.0
            })

    return float(ece), bin_details


def plot_reliability_diagram(
    bin_details: List[Dict],
    ece: float,
    save_path: Union[str, Path],
    title: str = "Reliability Diagram"
):
    """
    Plots and saves a calibration reliability diagram.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    confs = [b["confidence"] for b in bin_details if b["count"] > 0]
    accs = [b["accuracy"] for b in bin_details if b["count"] > 0]

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    if confs:
        plt.plot(confs, accs, "s-", color="#1f77b4", label=f"Model (ECE = {ece:.3f})")
    plt.xlabel("Mean Predicted Confidence")
    plt.ylabel("Observed Accuracy")
    plt.title(title, fontsize=11, fontweight="bold")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


class TemperatureScaler(nn.Module):
    """
    Applies post-hoc temperature scaling to calibrate neural network logits.
    Fits scalar T > 0 on validation logits using negative log likelihood.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = self.temperature.clamp(min=1e-3)
        return logits / temp

    def fit(self, val_logits: torch.Tensor, val_targets: torch.Tensor, max_iter: int = 50) -> float:
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.05, max_iter=max_iter)
        nll_criterion = nn.CrossEntropyLoss()

        def _eval():
            optimizer.zero_grad()
            scaled = self.forward(val_logits)
            loss = nll_criterion(scaled, val_targets)
            loss.backward()
            return loss

        optimizer.step(_eval)
        return float(self.temperature.item())


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: Union[str, torch.device] = "cpu",
    output_dir: Union[str, Path] = "vision/outputs",
    val_loader: Optional[DataLoader] = None,
    save_prefix: str = ""
) -> Dict:
    """
    Evaluates model on dataloader, measures latency, ECE, temperature scaling,
    and generates confusion matrix and calibration plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    model.eval()

    all_logits = []
    all_preds = []
    all_targets = []
    total_latency_sec = 0.0
    total_crops = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            start_t = time.perf_counter()
            logits = model(inputs)
            if device.type == "cuda" if isinstance(device, torch.device) else str(device).startswith("cuda"):
                torch.cuda.synchronize()
            latency = time.perf_counter() - start_t

            preds = torch.argmax(logits, dim=1)
            total_latency_sec += latency
            total_crops += inputs.size(0)

            all_logits.append(logits.cpu())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    all_logits_t = torch.cat(all_logits, dim=0)
    all_probs = torch.softmax(all_logits_t, dim=1)[:, 1].numpy()
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Core metrics
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1]).tolist()
    ms_per_crop = (total_latency_sec / total_crops * 1000.0) if total_crops > 0 else 0.0

    # Uncalibrated ECE
    ece_uncal, bin_details = compute_ece(all_probs, all_targets)

    # Optional Temperature Scaling if val_loader provided
    fitted_temp = 1.0
    ece_cal = ece_uncal
    if val_loader is not None:
        val_logits_list = []
        val_targets_list = []
        with torch.no_grad():
            for v_in, v_tgt in val_loader:
                v_in = v_in.to(device)
                val_logits_list.append(model(v_in).cpu())
                val_targets_list.extend(v_tgt.numpy().tolist())
        v_logits = torch.cat(val_logits_list, dim=0)
        v_targets = torch.tensor(val_targets_list, dtype=torch.long)

        scaler = TemperatureScaler()
        fitted_temp = scaler.fit(v_logits, v_targets)
        cal_logits = scaler(all_logits_t)
        cal_probs = torch.softmax(cal_logits, dim=1)[:, 1].detach().numpy()
        ece_cal, _ = compute_ece(cal_probs, all_targets)

    prefix_str = f"{save_prefix}_" if save_prefix else ""

    # Plot reliability diagram
    rel_plot_path = output_dir / f"{prefix_str}reliability_diagram.png"
    plot_reliability_diagram(
        bin_details, ece_uncal, rel_plot_path,
        title=f"Reliability Diagram (ECE={ece_uncal:.3f})"
    )

    # Plot confusion matrix
    cm_img_path = output_dir / f"{prefix_str}confusion_matrix.png"
    plt.figure(figsize=(6, 5))
    cm_arr = np.array(cm)
    plt.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix ({save_prefix or 'MobileNetV3'})", fontsize=11, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Free (0)", "Occupied (1)"])
    plt.yticks(tick_marks, ["Free (0)", "Occupied (1)"])
    thresh = cm_arr.max() / 2.0 if cm_arr.max() > 0 else 1.0
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

    metrics = {
        "test_crops_count": int(total_crops),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "ece_uncalibrated": round(float(ece_uncal), 4),
        "ece_calibrated": round(float(ece_cal), 4),
        "fitted_temperature": round(float(fitted_temp), 3),
        "confusion_matrix": {
            "labels": ["Free (0)", "Occupied (1)"],
            "matrix": cm
        },
        "inference_latency_cpu_ms_per_crop": round(float(ms_per_crop), 2),
        "throughput_crops_per_sec": round(float(total_crops / total_latency_sec), 1) if total_latency_sec > 0 else 0.0
    }

    metrics_path = output_dir / f"{prefix_str}metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def compare_classifier_vs_yolo(
    test_samples: List[Tuple[Path, Path]],
    classifier_model: nn.Module,
    yolo_conf: float = 0.25,
    device: Union[str, torch.device] = "cpu"
) -> Dict[str, Dict]:
    """
    Evaluates both MobileNetV3 and YOLO (mapped to slots) on the EXACT same test slots.
    Returns comparison metrics per model.
    """
    from .classify_slots import classify_slots

    y_true = []
    y_pred_classifier = []
    y_pred_yolo = []

    for img_p, xml_p in test_samples:
        tree = ET.parse(xml_p)
        root = tree.getroot()

        from .data_pkl import parse_pklot_xml_slots
        _, slots = parse_pklot_xml_slots(xml_p)
        for s in slots:
            if "points" not in s and "contour" in s:
                s["points"] = [[float(p[0]), float(p[1])] for p in s["contour"]]
            y_true.append(s["occupied"])

        # 1. Classifier predictions
        cls_res = classify_slots(str(img_p), slots, model=classifier_model, device=device)
        y_pred_classifier.extend([r["occupied"] for r in cls_res])

        # 2. YOLO mapped to slots
        _, dets = count_vehicles(str(img_p), conf=yolo_conf)
        yolo_res = map_detections_to_slots(dets, slots)
        y_pred_yolo.extend([r["occupied"] for r in yolo_res])

    def _get_metrics(y_t, y_p):
        return {
            "accuracy": round(float(accuracy_score(y_t, y_p)), 4),
            "precision": round(float(precision_score(y_t, y_p, zero_division=0)), 4),
            "recall": round(float(recall_score(y_t, y_p, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_t, y_p, zero_division=0)), 4)
        }

    return {
        "mobilenetv3": _get_metrics(y_true, y_pred_classifier),
        "yolo11n_mapped": _get_metrics(y_true, y_pred_yolo)
    }
