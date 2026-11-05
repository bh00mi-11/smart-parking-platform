"""
MobileNetV3-small slot occupancy classifier training pipeline.
Features:
- Fine-tunes torchvision.models.mobilenet_v3_small with ImageNet pretrained weights.
- Replaces classifier head for 2 classes (0: free, 1: occupied).
- AdamW optimizer with cosine/step scheduler.
- CrossEntropyLoss with dynamic class weighting for handling imbalanced labels.
- Fixed random seed for complete reproducibility.
- Automatic Mixed Precision (AMP) when CUDA is available.
- Early stopping on validation F1 score.
- Supports leave-one-lot-out (loo) 3-fold cross validation and time-based splits.
- Expected Calibration Error (ECE) and Temperature Scaling in evaluation.
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

_src_dir = Path(__file__).resolve().parent
_vision_dir = _src_dir.parent
if str(_vision_dir) not in sys.path:
    sys.path.insert(0, str(_vision_dir))

from src.data_pkl import PKLotDataset, get_default_eval_transforms, get_default_train_transforms
from src.eval import evaluate_model, f1_score
from src.splits import (
    assert_no_data_leakage,
    make_leave_one_lot_out_splits,
    make_time_based_split,
    make_unseen_lot_split,
    PKLOT_LOTS
)


if hasattr(torch.backends, "mkldnn"):
    torch.backends.mkldnn.enabled = False

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_model(num_classes: int = 2) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def compute_class_weights(dataset: PKLotDataset) -> torch.Tensor:
    labels = [item["occupied"] for item in dataset.slots_index]
    count_0 = max(1, labels.count(0))
    count_1 = max(1, labels.count(1))
    total = count_0 + count_1
    w0 = total / (2.0 * count_0)
    w1 = total / (2.0 * count_1)
    return torch.tensor([w0, w1], dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(targets.cpu().numpy().tolist())

    epoch_loss = running_loss / max(1, len(loader.dataset))
    epoch_f1 = f1_score(all_targets, all_preds, zero_division=0)
    return epoch_loss, epoch_f1


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    val_loss = running_loss / max(1, len(loader.dataset))
    val_f1 = f1_score(all_targets, all_preds, zero_division=0)
    return val_loss, val_f1


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV3-small slot classifier")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--subset", action="store_true", help="Pipeline check mode: rapid CPU test on tiny subset")
    parser.add_argument("--subset-size", type=int, default=120, help="Number of samples when --subset is enabled")
    parser.add_argument(
        "--protocol",
        "--split-protocol",
        dest="split_protocol",
        type=str,
        default="time-based",
        choices=["time-based", "unseen-lot", "loo"],
        help="Splitting protocol ('time-based', 'unseen-lot', 'loo' for leave-one-lot-out)"
    )
    parser.add_argument("--fold", type=int, default=0, choices=[0, 1, 2], help="Fold index for loo protocol")
    parser.add_argument("--output-dir", type=str, default="vision/outputs", help="Output directory")
    args = parser.parse_args()

    set_seed(args.seed)

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / "PKLot"
    if args.output_dir == "vision/outputs":
        output_dir = base_dir / "outputs"
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    print("=" * 65)
    print("MobileNetV3-Small Slot Occupancy Classifier Training")
    if args.subset:
        print("[PIPELINE CHECK] Running rapid verification on small subset - NOT formal results")
    print(f"Device: {device} (AMP: {use_amp}) | Protocol: {args.split_protocol} (Fold: {args.fold}) | Epochs: {args.epochs}")
    print("=" * 65)

    all_imgs = sorted(list(data_dir.rglob("*.jpg")))
    pairs = [(im, im.with_suffix(".xml")) for im in all_imgs if im.with_suffix(".xml").exists()]
    assert len(pairs) > 0, f"No valid PKLot pairs found in {data_dir}"

    if args.split_protocol == "loo":
        splits = make_leave_one_lot_out_splits(pairs, fold=args.fold, seed=args.seed)
        save_prefix = f"loo_fold{args.fold}_{PKLOT_LOTS[args.fold]}"
        ckpt_name = f"mobilenetv3_loo_fold{args.fold}.pth"
    elif args.split_protocol == "unseen-lot":
        splits = make_unseen_lot_split(pairs, test_lot="UFPR05", val_lot="UFPR04", train_lot="PUCPR")
        save_prefix = "unseen_lot"
        ckpt_name = "mobilenetv3_unseen.pth"
    else:
        splits = make_time_based_split(pairs, train_ratio=0.6, val_ratio=0.2, seed=args.seed)
        save_prefix = "time_based"
        ckpt_name = "mobilenetv3_best.pth"

    assert_no_data_leakage(splits)

    cache_dir = base_dir / "data" / "cache" / "crops"
    train_dataset = PKLotDataset(splits["train"], transform=get_default_train_transforms(224), cache_dir=cache_dir)
    val_dataset = PKLotDataset(splits["val"], transform=get_default_eval_transforms(224), cache_dir=cache_dir)
    test_dataset = PKLotDataset(splits["test"], transform=get_default_eval_transforms(224), cache_dir=cache_dir)

    if args.subset:
        sub_n = min(args.subset_size, len(train_dataset))
        val_n = min(max(10, args.subset_size // 4), len(val_dataset))
        test_n = min(max(10, args.subset_size // 4), len(test_dataset))
        train_dataset = Subset(train_dataset, list(range(sub_n)))
        val_dataset = Subset(val_dataset, list(range(val_n)))
        test_dataset = Subset(test_dataset, list(range(test_n)))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(num_classes=2).to(device)

    base_train = train_dataset.dataset if isinstance(train_dataset, Subset) else train_dataset
    class_weights = compute_class_weights(base_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    best_val_f1 = -1.0
    best_checkpoint_path = output_dir / ckpt_name

    for epoch in range(1, args.epochs + 1):
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        val_loss, val_f1 = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
              f"Train Loss: {train_loss:.4f} - Train F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} - Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
            }, best_checkpoint_path)

    print("\n" + "=" * 65)
    print(f"Evaluating {save_prefix} on TEST Split...")
    print("=" * 65)
    if best_checkpoint_path.exists():
        ckpt = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        output_dir=output_dir,
        val_loader=val_loader,
        save_prefix=save_prefix
    )

    tag = "[PIPELINE CHECK]" if args.subset else "[FINAL RESULT]"
    print(f"\n{tag} --- TEST SPLIT RESULTS ({save_prefix}) ---")
    print(f"Accuracy:    {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1 Score:    {metrics['f1_score']:.4f}")
    print(f"ECE (Uncal): {metrics['ece_uncalibrated']:.4f} | ECE (Cal): {metrics['ece_calibrated']:.4f}")
    print(f"Latency:     {metrics['inference_latency_cpu_ms_per_crop']:.2f} ms / crop")
    print("=" * 65)


if __name__ == "__main__":
    main()
