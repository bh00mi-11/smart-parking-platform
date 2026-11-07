# Vision Module: Computer Vision & Probabilistic Fusion

**Team Role**: Member 1 (Computer Vision & Fusion)  
**Module Directory**: `vision/`  
**Shared Contract**: [`docs/api_contract.md`](../docs/api_contract.md)

> [!IMPORTANT]
> **Methodological Scope**: All datasets used (PKLot, CNRPark) are public academic benchmarks, and fusion simulation streams are generated using reproducible stochastic generators. **These results validate the engineering methodology, uncertainty calibration, and software pipeline, not real-world parking facility accuracy.**

---

## 1. Architectural Overview

1. **Managed Lots (marked slots)**:
   - Per-slot crop classification ($0 = \text{Free}, 1 = \text{Occupied}$) using fine-tuned **MobileNetV3-Small**.
   - Supports arbitrary polygons, oriented bounding boxes, and rotated rectangles via OpenCV affine transforms.
2. **Open Zones (unmarked surface parking)**:
   - Vehicle localization and counting using **YOLO11n** (`car`, `motorcycle`, `bus`, `truck`).
   - Supports boundary containment via `count_in_polygons(image, polygons)` and mapping to arbitrary slot shapes via `map_detections_to_slots`.
3. **Probabilistic Fusion with Freshness-Based Confidence**:
   - Only three valid occupancy sources: **`camera`**, **`manual`**, and **`simulator`**. (No IoT sensors or external APIs).
   - Combines camera counts, manual reports, barrier check-in/out event deltas, and historical priors using **inverse-variance weighting**.
   - As observations age, uncertainty diffuses ($\sigma^2(t) = \sigma_0^2 + \beta \Delta t$) and confidence decays exponentially:
     $$\text{confidence} = R_{\text{source}} \times \exp\left(-\frac{\Delta t}{\tau}\right)$$
   - Outputs an 80% coverage range ($[\text{free\_low}, \text{free\_high}]$).
   - Only manual reports or exact gate counts may collapse a managed-lot range (`free_low == free_high`). Cameras do not collapse the range to prevent overclaiming; their range reflects per-slot binomial uncertainty ($\sum p(1-p) + \sigma_{sys}^2$).

---

## 2. Directory Structure

```
vision/
├── data/                       # PKLot dataset root (gitignored)
├── examples/
│   └── demo_camera_to_occupancy.py # End-to-end camera frame to OccupancyUpdate JSON
├── notebooks/
│   └── colab_train.ipynb       # 3-Fold LOO GPU training notebook for Google Colab
├── outputs/                    # Evaluation artifacts, JSON metrics & plots (gitignored checkpoints)
├── scripts/
│   ├── download_data.py        # Automated PKLot / CNRPark downloader
│   └── inspect_data.py         # Dataset inspection and sample grid generator
├── src/
│   ├── __init__.py
│   ├── camera_adapter.py       # Camera pipeline -> OccupancyUpdate JSON
│   ├── classify_slots.py       # Direct slot classification API
│   ├── data_pkl.py             # PKLot rotated crop dataset loader
│   ├── eval.py                 # ECE, Reliability diagram, Temp scaling, YOLO comparison
│   ├── eval_fusion.py          # Fusion calibration evaluation on held-out seeds
│   ├── eval_yolo_lots.py       # Per-lot YOLO counting benchmark (PUCPR, UFPR04, UFPR05)
│   ├── fusion.py               # Probabilistic sensor fusion engine
│   ├── schemas.py              # Pydantic models (OccupancyUpdate, CheckInEvent)
│   ├── sim_stub.py             # Replaceable diurnal synthetic simulation generator
│   ├── splits.py               # Time-based and 3-fold leave-one-lot-out (LOO) splits
│   ├── train_classifier.py     # MobileNetV3 training with LOO and AMP
│   └── yolo_count.py           # YOLO11n polygon counting & slot mapping
└── tests/
    ├── conftest.py
    ├── test_camera_adapter.py  # Mocked classifier camera adapter tests
    ├── test_fusion.py          # Freshness decay, interval widening, event shifts tests
    ├── test_schemas.py         # Bounds and collapse validation tests
    └── test_splits.py          # LOO 3-fold zero-leakage tests
```

---

## 3. Environment Setup & Weights Placement

### Installation
```powershell
pip install -r vision/requirements.txt
```

### Where to Put Classifier Weights
The camera adapter and evaluation look for the trained weights checkpoint in order:
1. Custom path passed directly to function.
2. `MOBILENETV3_WEIGHTS_PATH` environment variable.
3. `vision/outputs/mobilenetv3_loo_fold*.pth` or `vision/outputs/mobilenetv3_best.pth`.

*Note: Model weights (`*.pth`, `*.pt`) and dataset archives over 50 MB are strictly gitignored.*

---

## 4. How Teammates Call the Vision Module

### 4.1 Process Camera Frame (`camera_adapter.py`)
```python
from vision.src.camera_adapter import process_frame

zone_config = {
    "zone_type": "managed",  # or "open"
    "capacity": 50,
    "slots": [
        {"slot_id": "A1", "points": [[100, 150], [180, 150], [180, 300], [100, 300]]},
        {"slot_id": "A2", "points": [[190, 150], [270, 150], [270, 300], [190, 300]]}
    ]
}

# Processes frame, executes classifier / YOLO, and fuses into OccupancyUpdate
update = process_frame(
    facility_id="FAC-DOWNTOWN",
    zone_id="ZONE-L1",
    image="path/to/frame.jpg",
    zone_config=zone_config,
    camera_reliability=0.90  # configurable, default 0.90
)
print(update.model_dump_json(indent=2))
```

### 4.2 Fuse Asynchronous Multi-Source Updates (`fusion.py`)
```python
from datetime import datetime, timezone
from vision.src.fusion import estimate_occupancy

zone = {"facility_id": "FAC-01", "zone_id": "ZONE-A", "capacity": 60, "zone_type": "open"}
observations = {
    "camera": {"timestamp": "2026-11-04T21:00:00Z", "count": 35},
    "manual": {"timestamp": "2026-11-04T20:30:00Z", "count": 32},
    "prior": {"mean": 30.0, "std": 6.0}
}

update = estimate_occupancy(zone, observations, now=datetime.now(timezone.utc))
print(f"80% Availability: {update.free_low} to {update.free_high} free slots (Confidence: {update.confidence})")
```

---

## 5. REAL EXPERIMENTAL RESULTS (From Executed Local Runs)

All figures below were measured directly from local test executions and evaluations.

### 5.1 Automated Test Suite Execution
- **Command**: `pytest vision/tests`
- **Result**: **19 passed in 7.92s (100% success)**
- **Test Coverage**:
  - `test_schemas.py`: Bound invariants ($0 \le \text{free\_low} \le \text{free\_high} \le \text{capacity}$) and exact range collapse restricted to manual/gate sources.
  - `test_fusion.py`: Freshness decay, temporal uncertainty widening, multi-source weighting, check-in delta shifts, capacity clamping, and prior fallbacks.
  - `test_camera_adapter.py`: Mocked PyTorch MobileNetV3 slot classification, non-zero interval derivation, configurable camera reliability ($R < 1.0$), and missing slot validation.
  - `test_splits.py`: 3-fold leave-one-lot-out (LOO) zero-leakage verification across PKLot splits.

---

### 5.2 MobileNetV3 3-Fold Leave-One-Lot-Out (LOO) Cross-Validation

The MobileNetV3-Small classifier was evaluated across the 3 PKLot facilities using a rigorous Leave-One-Lot-Out (LOO) protocol: in each fold, two facilities are used for training/validation and the remaining held-out facility is used exclusively for testing.

> [!WARNING]
> **Data Scale Notice**: The Colab run evaluated test crops corresponding to the 12 local PKLot frames (4 frames per lot: 780 crops for PUCPR, 146 crops for UFPR04, and 184 crops for UFPR05; total 1,110 test crops). **This run used 12 frames, so it is not yet a large-data result.** Full multi-thousand frame training on the entire 12,000+ PKLot archive remains future work on high-capacity GPU clusters.

#### Per-Fold Leave-One-Lot-Out Metrics (from `loo_fold*_metrics.json`):
| Fold / Held-Out Lot | Test Crops ($N$) | Accuracy | Precision | Recall | F1 Score | Calibrated ECE | Checkpoint File |
|---|---|---|---|---|---|---|---|
| **Fold 0 (PUCPR)** | 780 | 88.21% | 97.97% | 79.86% | 0.8799 | 0.2657 | `mobilenetv3_loo_fold0.pth` |
| **Fold 1 (UFPR04)** | 146 | 73.29% | 47.30% | 100.00% | 0.6422 | 0.2365 | `mobilenetv3_loo_fold1.pth` |
| **Fold 2 (UFPR05)** | 184 | 99.46% | 100.00% | 99.16% | 0.9958 | 0.0871 | `mobilenetv3_loo_fold2.pth` |
| **Macro Average** | **1,110** | **86.99%** | **81.76%** | **93.01%** | **0.8393** | **0.1964** | *All 3 LOO models* |

---

### 5.3 Fusion Calibration & Baseline Comparisons (`eval_fusion.py`)

#### Experimental Protocol:
- **Tuning Seeds**: `[101, 102]` (used strictly for tuning hyperparameters).
- **Held-Out Evaluation Seeds**: `[201, 202, 203]` across managed (capacity 60) and open (capacity 80) zones.
- **Evaluation Points**: 1,728 evaluation steps across synthetic diurnal trajectories.
- **Nominal Target Coverage**: 80.0% coverage interval ($z = 1.17$).

#### MAE and Baseline Comparison (Held-Out Seeds [201, 202, 203]):
| Metric / Method | Measured Value | Target / Comparison |
|---|---|---|
| **Fusion Occupancy MAE** | **2.151 slots** | Multi-source fused estimate |
| **Baseline 1 (Last Observation) MAE** | **1.605 slots** | Raw last sensor observation count |
| **Baseline 2 (Prior Only) MAE** | 1.758 slots | Historical diurnal profile only |
| **Overall 80% Range Coverage** | **79.75%** | Target: $80.0\%$ (strictly calibrated) |

> [!NOTE]
> **Honest Assessment on MAE**: Fusion MAE (2.151 slots) does **not** beat the Last-Observation baseline MAE (1.605 slots). This occurs because synthetic observations undergo occasional drop/noise bursts while raw last-observation counts act as an unregularized snapshot, and inverse-variance blending with the broader historical prior slightly pulls extreme values toward the diurnal mean. **The core engineering value of the fusion module is its calibrated uncertainty representation** (79.75% empirical coverage vs 80.0% nominal), rather than pure point-prediction MAE minimization.

#### Empirical Coverage & Interval Width by Observation Age Bucket (Held-Out Seeds [201, 202, 203]):
| Observation Age Bucket | Sample Count ($N$) | Empirical Coverage Rate | Mean Interval Width |
|---|---|---|---|
| **`< 5 min`** | 461 | **76.8%** | 4.4 slots |
| **`5 - 15 min`** | 434 | **80.7%** | 5.5 slots |
| **`15 - 30 min`** | 430 | **81.9%** | 6.3 slots |
| **`> 30 min`** | 403 | **79.9%** | 6.9 slots |

*All age buckets are tightly calibrated around the nominal 80% target (ranging 76.8% to 81.9%), completely resolving the previous under-5-min undercoverage (previously 55.1%) and stale overcoverage (previously 94–99%).*

#### Sensitivity on Held-Out Noise Settings (Seeds [201, 202, 203]):
| Noise Scenario | Camera Noise ($\sigma$) | Event Drop Rate | Fusion MAE | Last-Obs MAE | Overall 80% Coverage |
|---|---|---|---|---|---|
| **Standard Noise** | 1.2 slots | 10% | 2.151 slots | 1.600 slots | **79.75%** |
| **High Noise** | 2.0 slots | 20% | 2.337 slots | 1.973 slots | **76.50%** |
| **Low Noise** | 0.5 slots | 3% | 1.986 slots | 1.229 slots | **83.62%** |

---

### 5.4 YOLO11n Per-Lot Vehicle Counting Evaluation (`eval_yolo_lots.py`)

Evaluated on all available local PKLot frames across diverse weather conditions (Sunny, Cloudy, Rainy).

> [!WARNING]
> **Sample Size Notice**: Due to local repository footprint constraints, 4 full-resolution frames per lot are currently available locally (totaling 1,110 annotated slot spaces). **Small sample sizes make per-lot figures unstable.** Colab/cloud evaluations with the full dataset are recommended for definitive benchmark numbers.

| Lot Identifier | Camera Perspective | Images ($N$) | Total Annotated Slots | imgsz=640 MAE | imgsz=1280 MAE | Analysis & Observations |
|---|---|---|---|---|---|---|
| **PUCPR** | Far Overhead View | 4 | 780 | **105.50 vehicles** | **79.00 vehicles** | Extreme camera distance and small vehicle bounding boxes (~15–25 px). High resolution (1280) reduces MAE by 25.1%, confirming resolution sensitivity for distant open lots. |
| **UFPR04** | Angled Surveillance | 4 | 146 | **1.25 vehicles** | **6.25 vehicles** | High counting accuracy at imgsz=640 across overcast and rain; higher resolution introduces slight false positives on background patterns. |
| **UFPR05** | Angled Surveillance | 4 | 184 | **4.00 vehicles** | **8.75 vehicles** | Reliable vehicle detection across sunny and cloudy daylight conditions. |

---

### 5.5 Per-Test-Lot Model Comparison: MobileNetV3 (LOO Weights) vs YOLO Mapped to Slots

Evaluated on the exact same test slots per lot using `compare_classifier_vs_yolo()` with corresponding leave-one-lot-out weights (`mobilenetv3_loo_fold0.pth` for PUCPR, `mobilenetv3_loo_fold1.pth` for UFPR04, and `mobilenetv3_loo_fold2.pth` for UFPR05):

| Test Lot | Evaluated Images | Annotated Slots | Weights Used | Model / Approach | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|---|---|---|---|
| **PUCPR** | 4 | 780 | `fold0.pth` | **MobileNetV3 (LOO)** | **84.74%** | 88.16% | 82.94% | **0.8547** |
| | | | | YOLO11n (Mapped) | 45.90% | 0.00% | 0.00% | 0.0000 |
| **UFPR04** | 4 | 146 | `fold1.pth` | MobileNetV3 (LOO) | 82.88% | 58.62% | 97.14% | 0.7312 |
| | | | | **YOLO11n (Mapped)** | **91.10%** | 95.83% | 65.71% | **0.7797** |
| **UFPR05** | 4 | 184 | `fold2.pth` | **MobileNetV3 (LOO)** | 87.50% | 85.82% | 96.64% | **0.9091** |
| | | | | YOLO11n (Mapped) | 80.98% | 93.75% | 75.63% | 0.8372 |

---

### 5.6 End-to-End Camera Demo (`demo_camera_to_occupancy.py`)

*Label: Pipeline Check (executed on PKLot frame `2012-10-16_06_21_43.jpg`).*

#### Managed Lot Output (Camera Range Overclaiming Fixed):
```json
{
  "facility_id": "FAC-PKLOT",
  "zone_id": "ZONE-MANAGED",
  "timestamp": "2026-11-06T14:52:40.910205Z",
  "zone_type": "managed",
  "capacity": 195,
  "free_low": 91,
  "free_high": 108,
  "source": "camera",
  "confidence": 0.9,
  "age_seconds": 0.0,
  "interval_level": 0.8
}
```
*Notice: `free_low` (91) and `free_high` (108) do not collapse to a single point for `source="camera"`, correctly reflecting per-slot binomial uncertainty $\sum p(1-p) + \sigma_{sys}^2$. Confidence is clamped to `0.90` (camera source reliability), preventing overclaiming.*

#### Open Zone Output:
```json
{
  "facility_id": "FAC-STREET",
  "zone_id": "ZONE-OPEN",
  "timestamp": "2026-11-06T14:52:47.334698Z",
  "zone_type": "open",
  "capacity": 50,
  "free_low": 48,
  "free_high": 50,
  "source": "camera",
  "confidence": 0.88,
  "age_seconds": 0.0,
  "interval_level": 0.8
}
```
Annotated visual verification saved to: `vision/outputs/annotated_managed_2012-10-16_06_21_43.jpg`.
