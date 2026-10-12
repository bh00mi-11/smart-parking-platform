# Vision Module: Computer Vision and Fusion

Responsible for:
1. **Managed Lots (marked slots)**: Slot-level occupancy classification (0 = Free, 1 = Occupied) using a fine-tuned MobileNetV3-Small classifier on slot crops.
2. **Open Zones (unmarked areas)**: Vehicle counting and localization using pretrained YOLO11n.
3. **Backend Integration**: Direct inference function `classify_slots(image, slot_boxes)` for integration with FastAPI / database services.

---

## 1. Environment Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- PyTorch 2.0+ (CPU or CUDA GPU)
- All packages are free and open-source.

### Installation
From the repository root or `vision/` folder:
```bash
cd vision
pip install -r requirements.txt
```

---

## 2. Dataset Sources & Licenses

| Dataset | Official Source / Archive | License | Description |
|---|---|---|---|
| **PKLot** | Laboratório Visão Robótica e Imagem (VRI) - UFPR / PUCPR <br> Primary Archive: `http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz` <br> Metadata & Mirrors: `https://huggingface.co/datasets/Voxel51/PKLot` | Creative Commons Attribution 4.0 International (CC BY 4.0) | 12,417 high-res (1280x720) parking images with rotated rect & polygon XML annotations across 3 lots (PUCPR, UFPR04, UFPR05) and 3 weathers (Sunny, Cloudy, Rainy). |
| **CNRPark+EXT** | Institute of Information Science and Technologies (ISTI-CNR) / AIMH Lab <br> Repo: `https://github.com/fabiocarrara/deep-parking` <br> Patches Archive: `https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNRPark-Patches-150x150.zip` | Research / Academic Use (Open) | 150x150 slot patch crops categorized into `busy/` (occupied) and `free/` across camera viewpoints and weather conditions. |
| **Official Splits** | deep-parking splits archive: `https://github.com/fabiocarrara/deep-parking/releases/download/archive/splits.zip` | Academic Open Source | Benchmark list files defining lot and day splits for both PKLot (695,897 slots) and CNRPark-EXT (144,965 slots). |

### Downloading Data
Run the automated downloader script:
```bash
python scripts/download_data.py
```
This downloads and extracts the datasets into `vision/data/` (gitignored).

To generate data inspection statistics and sample crop grids:
```bash
python scripts/inspect_data.py
```

---

## 3. Real Dataset Inspection Statistics

From our real runs on the extracted dataset files:

### PKLot Dataset Statistics
- **Image Resolution**: 1280 x 720 (uniform)
- **Total full-frame images evaluated**: 12
- **Total slot annotations evaluated**: 1,110 slots

#### By Parking Lot:
| Lot | Images | Total Slots | Occupied Slots | Free Slots | Occupancy Rate |
|---|---|---|---|---|---|
| **PUCPR** | 4 | 780 | 422 | 358 | 54.1% |
| **UFPR04** | 4 | 146 | 35 | 111 | 24.0% |
| **UFPR05** | 4 | 184 | 119 | 65 | 64.7% |

#### By Weather Condition:
| Weather | Images | Total Slots | Occupied Slots | Free Slots | Occupancy Rate |
|---|---|---|---|---|---|
| **Cloudy** | 3 | 278 | 48 | 230 | 17.3% |
| **Rainy** | 4 | 313 | 250 | 63 | 79.9% |
| **Sunny** | 5 | 519 | 278 | 241 | 53.6% |

### CNRPark-Patches Statistics
- **Total Patch Files**: 12,584
- **Occupied (`busy/`) Patches**: 8,402 (66.8%)
- **Free (`free/`) Patches**: 4,182 (33.2%)
- **Patch Resolution**: 150 x 150

A 12-sample grid (6 occupied, 6 free) is saved to [`outputs/sample_crops_grid.png`](outputs/sample_crops_grid.png).

---

## 4. Leakage-Safe Splitting Protocols

Implemented in `src/splits.py`:

1. **Unseen-Lot Split (`unseen-lot`)**:
   - `train`: PUCPR (4 images, 780 slots, 54.1% occupied)
   - `val`: UFPR04 (4 images, 146 slots, 24.0% occupied)
   - `test`: UFPR05 (4 images, 184 slots, 64.7% occupied)
   - Guarantees evaluation strictly on a physical parking environment unseen during training.

2. **Time-Based Split (`time-based`)**:
   - Splits strictly by **date / day** within each lot (train 60%, val 20%, test 20%).
   - Crops from the same camera frame or the same calendar day never cross split boundaries.

### Leakage Verification
All splits are validated with `assert_no_data_leakage(splits)`, ensuring 0 overlapping full-frame images between partitions. Verified with automated pytest unit tests:
```bash
pytest tests/test_loaders_and_splits.py -v
```

---

## 5. Task 2: YOLO11n Vehicle Counting Baseline

Implemented in `src/yolo_count.py`.
Detects COCO vehicle classes: `car`, `motorcycle`, `bus`, `truck` with confidence threshold `conf=0.25`.

### Real Results on PKLot Images (Predicted vs True Count):
| Image | Lot | Weather | True Occupied Slots | Predicted Vehicles | Absolute Error |
|---|---|---|---|---|---|
| `2012-10-16_06_21_43.jpg` | PUCPR | Cloudy | 3 | 0 | 3 |
| `2012-10-25_19_33_54.jpg` | PUCPR | Rainy | 188 | 0 | 188 |
| `2012-09-11_15_16_58.jpg` | PUCPR | Sunny | 115 | 0 | 115 |
| `2012-09-11_15_27_08.jpg` | PUCPR | Sunny | 116 | 0 | 116 |
| `2012-12-22_15_55_12.jpg` | UFPR04 | Cloudy | 0 | 0 | 0 |
| `2012-12-07_16_42_25.jpg` | UFPR04 | Rainy | 35 | 32 | 3 |
| `2012-12-28_09_05_04.jpg` | UFPR04 | Rainy | 0 | 1 | 1 |
| `2012-12-26_08_25_03.jpg` | UFPR04 | Sunny | 0 | 1 | 1 |
| `2013-04-12_15_35_10.jpg` | UFPR05 | Cloudy | 45 | 34 | 11 |
| `2013-04-14_15_30_10.jpg` | UFPR05 | Rainy | 27 | 29 | 2 |
| `2013-02-22_06_05_00.jpg` | UFPR05 | Sunny | 2 | 2 | 0 |
| `2013-04-16_07_45_02.jpg` | UFPR05 | Sunny | 45 | 42 | 3 |

- **UFPR04 & UFPR05 MAE**: **2.62 vehicles** (exceptional counting accuracy on angled surveillance viewpoints).
- **PUCPR MAE**: High error due to severe long-distance elevated perspective where cars occupy only ~15x10 pixels in standard 640-stride COCO YOLO.
- **Overall Baseline MAE across all 12 images**: **36.92 vehicles**.
- Annotated output images saved to [`outputs/annotated_yolo/`](outputs/annotated_yolo/).

---

## 6. Task 4: MobileNetV3-Small Slot Occupancy Classifier

Fine-tuned `torchvision.models.mobilenet_v3_small` with ImageNet pretrained weights.
- Optimizer: AdamW (`lr=3e-4`, `weight_decay=1e-2`)
- Loss: CrossEntropyLoss with dynamic inverse-frequency class weights (`w_free=1.14`, `w_occ=0.89`)
- Random seed: 42 (fully reproducible)
- Early stopping monitored on validation F1 score.

### Real Evaluation Metrics (Test Split):
From `outputs/metrics.json`:
| Metric | Value |
|---|---|
| **Test Crops Evaluated** | 30 crops |
| **Accuracy** | **90.00%** |
| **Precision** | **0.9643** |
| **Recall** | **0.9310** |
| **F1 Score** | **0.9474** |
| **CPU Latency per Crop** | **3.84 ms** |
| **CPU Throughput** | **260.2 crops / sec** |

### Confusion Matrix:
- True Free: 0 correctly identified, 1 misclassified as occupied.
- True Occupied: 27 correctly identified, 2 misclassified as free.
- Confusion matrix image saved to [`outputs/confusion_matrix.png`](outputs/confusion_matrix.png).

---

## 7. How to Run Each Task

### 1. Download Data
```bash
python scripts/download_data.py
```

### 2. Inspect Data
```bash
python scripts/inspect_data.py
```

### 3. Run YOLO11n Vehicle Counting
```bash
python src/yolo_count.py
```

### 4. Run PyTest Unit Tests
```bash
pytest tests/test_loaders_and_splits.py -v
```

### 5. Run Fast Local Training Verification (CPU Subset)
```bash
python src/train_classifier.py --subset --subset-size 60 --epochs 3 --batch-size 16
```

### 6. Full Training on Google Colab (Free GPU)
Use the self-contained notebook: [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb).
Mounts Google Drive, clones the repository on branch `vision-baseline`, downloads datasets, runs AMP mixed precision training, and exports checkpoint and metrics to Drive.

---

## 8. Backend API: `classify_slots`

The vision module provides a clean inference API ready for Member 3's backend:
```python
from src.train_classifier import build_model
from src.classify_slots import classify_slots
import torch

model = build_model(num_classes=2)
checkpoint = torch.load("outputs/mobilenetv3_best.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])

slot_boxes = [
    {"slot_id": "slot_01", "bbox": [100, 200, 80, 150]},
    {"slot_id": "slot_02", "rotated_rect": {"cx": 450.0, "cy": 300.0, "w": 75.0, "h": 140.0, "angle": -35.0}}
]

predictions = classify_slots("path/to/parking_image.jpg", slot_boxes, model, device="cpu")
# Output:
# [
#   {"slot_id": "slot_01", "occupied": 1, "confidence": 0.9852},
#   {"slot_id": "slot_02", "occupied": 0, "confidence": 0.9610}
# ]
```
