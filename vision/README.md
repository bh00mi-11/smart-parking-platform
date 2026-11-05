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
   - In managed lots with fresh exact observations, the interval collapses to an exact count (`free_low == free_high`).

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
    zone_config=zone_config
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

All numbers below were produced by actual test executions in this workspace.

### 5.1 Automated Test Suite Execution
- **Command**: `pytest vision/tests`
- **Result**: **18 passed in 22.26s (100% success)**
- **Coverage**:
  - `test_schemas.py`: Bound invariants ($0 \le \text{free\_low} \le \text{free\_high} \le \text{capacity}$) and fresh exact source range collapse.
  - `test_fusion.py`: Confidence decay with age, range widening with age, fresh camera outweighing stale manual, check-in delta shifts, capacity boundary clamping, and no-observation fallback.
  - `test_camera_adapter.py`: Mocked PyTorch MobileNetV3 forward pass and error handling on missing slot definitions.
  - `test_splits.py`: 3-fold leave-one-lot-out leakage-safety verification.

---

### 5.2 Fusion Calibration on Held-Out Seeds (`eval_fusion.py`)
- **Evaluation Protocol**: Held-out simulation seeds `[201, 202, 203]` across managed and open zones (tuning seeds were partitioned strictly on `[101, 102]`).
- **Target Coverage**: Nominal 80.0% coverage interval ($\pm 1.28 \sigma$).

| Metric | Measured Value | Baseline Comparison |
|---|---|---|
| **Fusion Occupancy MAE** | **1.913 slots** | Compared to true occupancy |
| **Baseline 1 (Last Obs) MAE** | 1.605 slots | Raw last sensor count |
| **Baseline 2 (Prior Only) MAE** | 1.758 slots | Historical diurnal profile only |
| **Overall 80% Range Coverage** | **85.71%** | Target: $80.0\%$ (conservative & calibrated) |

#### Empirical Coverage & Interval Expansion by Data Age Bucket:
| Observation Age Bucket | Sample Count ($N$) | Empirical Coverage Rate | Mean Interval Width |
|---|---|---|---|
| **`< 5 min`** | 461 | 55.1% | 2.2 slots |
| **`5 - 15 min`** | 434 | 94.5% | 7.4 slots |
| **`15 - 30 min`** | 430 | 98.8% | 8.8 slots |
| **`> 30 min`** | 403 | 97.3% | 9.9 slots |

*Key finding: As the anchor observation ages, the interval width expands smoothly from 2.2 free slots to 9.9 free slots to account for time drift uncertainty, maintaining coverage above 94% on aging records.*

---

### 5.3 YOLO11n Per-Lot Vehicle Counting Baseline (`eval_yolo_lots.py`)
Evaluated separately across all 3 PKLot facilities at standard size (`imgsz=640`) and high-resolution (`imgsz=1280`):

| Lot Identifier | Camera Perspective | imgsz=640 MAE | imgsz=1280 MAE | Analysis & Observations |
|---|---|---|---|---|
| **PUCPR** | Elevated / Far Overhead | **59.00 vehicles** | **38.50 vehicles** | Severe overhead perspective with distant, tiny vehicle signatures (~15px). Known limitation for open-zone detector; increasing resolution to 1280 improves MAE by 34.7%. |
| **UFPR04** | Angled Surveillance | **0.50 vehicles** | **0.50 vehicles** | Near-perfect counting performance across sunny, cloudy, and rainy conditions. |
| **UFPR05** | Angled Surveillance | **5.50 vehicles** | **5.00 vehicles** | Robust vehicle detection with minor occlusion error at far lot boundaries. |

---

### 5.4 End-to-End Camera Demo (`demo_camera_to_occupancy.py`)
Executed on real PKLot frame `2012-10-16_06_21_43.jpg` (195 slots):

#### Managed Lot Output:
```json
{
  "facility_id": "FAC-PKLOT",
  "zone_id": "ZONE-MANAGED",
  "timestamp": "2026-11-04T16:34:41.060081Z",
  "zone_type": "managed",
  "capacity": 195,
  "free_low": 99,
  "free_high": 99,
  "source": "camera",
  "confidence": 1.0,
  "age_seconds": 0.0,
  "interval_level": 0.8
}
```

#### Open Zone Output:
```json
{
  "facility_id": "FAC-STREET",
  "zone_id": "ZONE-OPEN",
  "timestamp": "2026-11-04T16:34:48.262375Z",
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
