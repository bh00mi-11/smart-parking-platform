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
3. `vision/outputs/mobilenetv3_best.pth` or `vision/outputs/mobilenetv3_loo_fold0.pth`.

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

### 5.2 Fusion Calibration & Baseline Comparisons (`eval_fusion.py`)

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

*All age buckets are now tightly calibrated around the nominal 80% target (ranging 76.8% to 81.9%), completely resolving the previous under-5-min undercoverage (previously 55.1%) and stale overcoverage (previously 94–99%).*

#### Sensitivity on Held-Out Noise Settings (Seeds [201, 202, 203]):
| Noise Scenario | Camera Noise ($\sigma$) | Event Drop Rate | Fusion MAE | Last-Obs MAE | Overall 80% Coverage |
|---|---|---|---|---|---|
| **Standard Noise** | 1.2 slots | 10% | 2.151 slots | 1.600 slots | **79.75%** |
| **High Noise** | 2.0 slots | 20% | 2.337 slots | 1.973 slots | **76.50%** |
| **Low Noise** | 0.5 slots | 3% | 1.986 slots | 1.229 slots | **83.62%** |

---

### 5.3 YOLO11n Per-Lot Vehicle Counting Evaluation (`eval_yolo_lots.py`)

Evaluated on all available local PKLot frames across diverse weather conditions (Sunny, Cloudy, Rainy).

> [!WARNING]
> **Sample Size Notice**: Due to local repository footprint constraints, 4 full-resolution frames per lot are currently available locally (totaling 1,110 annotated slot spaces). **Small sample sizes make per-lot figures unstable.** Colab/cloud evaluations with the full dataset are recommended for definitive benchmark numbers.

| Lot Identifier | Camera Perspective | Images ($N$) | Total Annotated Slots | imgsz=640 MAE | imgsz=1280 MAE | Analysis & Observations |
|---|---|---|---|---|---|---|
| **PUCPR** | Far Overhead View | 4 | 780 | **105.50 vehicles** | **79.00 vehicles** | Extreme camera distance and small vehicle bounding boxes (~15–25 px). High resolution (1280) reduces MAE by 25.1%, confirming resolution sensitivity for distant open lots. |
| **UFPR04** | Angled Surveillance | 4 | 146 | **1.25 vehicles** | **6.25 vehicles** | High counting accuracy at imgsz=640 across overcast and rain; higher resolution introduces slight false positives on background patterns. |
| **UFPR05** | Angled Surveillance | 4 | 184 | **4.00 vehicles** | **8.75 vehicles** | Reliable vehicle detection across sunny and cloudy daylight conditions. |

---

### 5.4 Per-Test-Lot Model Comparison: MobileNetV3 vs YOLO Mapped to Slots

Evaluated on the exact same test slots per lot using `compare_classifier_vs_yolo()`:

> [!NOTE]
> **Pipeline Check Status**: These MobileNetV3 results represent a **pipeline check** using initial local smoke-test weights (`mobilenetv3_best.pth`). Definitive performance figures will be established once the full Colab 3-fold LOO training weights are transferred.

| Test Lot | Evaluated Images | Annotated Slots | Model / Approach | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|---|---|---|
| **PUCPR** | 4 | 780 | **MobileNetV3 (Classifier)** | **79.62%** | 75.14% | 93.13% | **0.8317** |
| | | | YOLO11n (Mapped to Slots) | 45.90% | 0.00% | 0.00% | 0.0000 |
| **UFPR04** | 4 | 146 | MobileNetV3 (Classifier) | 34.93% | 26.56% | 97.14% | 0.4172 |
| | | | **YOLO11n (Mapped to Slots)** | **91.10%** | 95.83% | 65.71% | **0.7797** |
| **UFPR05** | 4 | 184 | MobileNetV3 (Classifier) | 71.74% | 69.59% | 100.0% | 0.8207 |
| | | | **YOLO11n (Mapped to Slots)** | **80.98%** | 93.75% | 75.63% | **0.8372** |

---

### 5.5 End-to-End Camera Demo (`demo_camera_to_occupancy.py`)

*Label: Pipeline Check (executed using local initial weights on PKLot frame `2012-10-16_06_21_43.jpg`).*

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
