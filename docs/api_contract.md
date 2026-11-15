# Shared API Contract: Occupancy Updates & Events

> **Team Notice**: This contract defines the canonical schema (`OccupancyUpdate` and `CheckInEvent`) for Member 1 (Vision & Fusion), Member 2 (Forecasting), Member 3 (Backend & Simulator), and Member 4 (Mobile App). All downstream consumers and producers must conform to this schema.

---

## 1. Overview
The platform models parking availability probabilistically rather than assuming static, deterministic ground truth. Availability is emitted as an estimate range with an associated confidence level:
- As updates age or rely on noisier sources, confidence decays exponentially and the interval width ($[\text{free\_low}, \text{free\_high}]$) expands.
- For fresh exact observations in managed lots (camera slot classification or fresh manual count), the interval collapses to an exact value (`free_low == free_high`).

All three valid occupancy sources (**`manual`**, **`simulator`**, **`camera`**) emit the identical `OccupancyUpdate` payload structure.

---

## 2. Models & Fields Specification

### 2.1 `OccupancyUpdate`

| Field | Type | Required | Units / Allowed Values | Description |
|---|---|---|---|---|
| `facility_id` | `string` | Yes | Non-empty alphanumeric string | Unique facility/lot identifier (e.g. `"FAC-01"`). |
| `zone_id` | `string` | Yes | Non-empty alphanumeric string | Specific zone identifier within facility (e.g. `"ZONE-A"`). |
| `timestamp` | `string` | Yes | ISO 8601 UTC string (`YYYY-MM-DDTHH:MM:SSZ`) | Timestamp at which the occupancy estimate is evaluated. |
| `zone_type` | `string` | Yes | `"managed"` \| `"open"` | `"managed"` for marked slot facilities; `"open"` for unmetered / open-zone parking. |
| `capacity` | `integer` | Yes | $\ge 1$ | Total parking spaces allocated in this zone. |
| `free_low` | `integer` | Yes | $0 \le \text{free\_low} \le \text{free\_high}$ | Conservative lower bound of the 80% available slot estimate. |
| `free_high` | `integer` | Yes | $\text{free\_low} \le \text{free\_high} \le \text{capacity}$ | Upper bound of the 80% available slot estimate. |
| `source` | `string` | Yes | `"manual"` \| `"simulator"` \| `"camera"` | Primary observation anchor driving this estimate. |
| `confidence` | `float` | Yes | $[0.0, 1.0]$ | Freshness & source reliability score ($R \cdot e^{-\Delta t / \tau}$). Distinct from the 80% interval width. |
| `age_seconds` | `float` | Yes | $\ge 0.0$ seconds | Elapsed time since the primary observation anchor was recorded. |
| `interval_level` | `float` | No | Default: `0.8` ($80\%$) | Nominal coverage probability of $[\text{free\_low}, \text{free\_high}]$. |

#### Meaning of `confidence` vs Interval Level
- **Confidence ($[0.0, 1.0]$)**: Quantifies data trustworthiness. It is determined by the intrinsic sensor/source reliability multiplied by exponential time decay:
  $$\text{confidence} = R_{\text{source}} \times \exp\left(-\frac{\text{age\_seconds}}{\tau}\right)$$
  A confidence score of 0.95 means an accurate, high-fidelity observation was captured moments ago.
- **Interval Level ($0.80$)**: Corresponds to an 80% symmetric confidence interval ($\mu \pm 1.28 \sigma$) derived from inverse-variance sensor fusion and time drift.

---

### 2.2 `CheckInEvent`

| Field | Type | Required | Allowed Values | Description |
|---|---|---|---|---|
| `zone_id` | `string` | Yes | Non-empty string | Zone where event occurred. |
| `timestamp` | `string` | Yes | ISO 8601 UTC string | Time when the vehicle checked in or out. |
| `type` | `string` | Yes | `"in"` \| `"out"` | Nature of barrier/app transaction. |
| `user_id` | `string` | Yes | String | Anonymized user/driver identifier. |

---

## 3. JSON Payloads

### Example 1: Managed Lot (Fresh Exact Camera Observation)
In this case, the camera classified all marked spaces 4 seconds ago. Confidence is high and the range collapses to an exact number of free slots (`free_low == free_high`).

```json
{
  "facility_id": "FAC-CENTRAL",
  "zone_id": "ZONE-LEVEL-1",
  "timestamp": "2026-11-03T21:30:00Z",
  "zone_type": "managed",
  "capacity": 50,
  "free_low": 12,
  "free_high": 12,
  "source": "camera",
  "confidence": 0.99,
  "age_seconds": 4.0,
  "interval_level": 0.8
}
```

### Example 2: Open Zone (Aging Observation with 80% Range)
In an open zone detected via YOLO or manual tally 15 minutes ago, time drift has widened the variance. Free capacity is estimated between 8 and 18 spots with degraded confidence.

```json
{
  "facility_id": "FAC-NORTH",
  "zone_id": "ZONE-OPEN-SURFACE",
  "timestamp": "2026-11-03T21:30:00Z",
  "zone_type": "open",
  "capacity": 80,
  "free_low": 8,
  "free_high": 18,
  "source": "manual",
  "confidence": 0.54,
  "age_seconds": 900.0,
  "interval_level": 0.8
}
```

---

## 4. Vision & Fusion REST Endpoints (`/vision`)

The vision module exposes a mountable FastAPI router (`vision.src.api.router`) and standalone microservice (`vision.src.main_vision:app`).

### Running the Microservice
```powershell
uvicorn vision.src.main_vision:app --host 0.0.0.0 --port 8000
```
Or mount directly into Member 3's main backend application:
```python
from vision.src.api import router as vision_router
app.include_router(vision_router)
```

### 4.1 `GET /vision/health`
Returns loaded model weights path and readiness status.
- **Response `200 OK`**:
```json
{
  "status": "ok",
  "model_ready": true,
  "weights_path": "vision\\outputs\\mobilenetv3_final.pth"
}
```

### 4.2 `POST /vision/process-frame`
Multipart image upload accepting raw camera bytes and a JSON `zone_config` field. Returns an `OccupancyUpdate`.
- **Request (multipart/form-data)**:
  - `image`: raw image file (`.jpg` / `.png`).
  - `zone_config`: JSON string specifying zone parameters and slot polygons:
```json
{
  "facility_id": "FAC-PUCPR",
  "zone_id": "ZONE-PUCPR",
  "zone_type": "managed",
  "capacity": 100,
  "slots": [
    {
      "slot_id": "1",
      "points": [[297.0, 481.0], [339.0, 477.0], [361.0, 528.0], [315.0, 533.0]]
    }
  ]
}
```
- **Response `200 OK`**:
```json
{
  "facility_id": "FAC-PUCPR",
  "zone_id": "ZONE-PUCPR",
  "timestamp": "2026-11-15T07:13:41.454008Z",
  "zone_type": "managed",
  "capacity": 100,
  "free_low": 99,
  "free_high": 100,
  "source": "camera",
  "confidence": 0.9,
  "age_seconds": 0.0,
  "interval_level": 0.8
}
```

### 4.3 `POST /vision/estimate`
JSON endpoint for multi-sensor probabilistic sensor fusion combining camera counts, manual counts, check-in streams, and diurnal priors with freshness decay.
- **Request (`application/json`)**:
```json
{
  "zone": {
    "facility_id": "FAC-TEST",
    "zone_id": "ZONE-M1",
    "capacity": 60,
    "zone_type": "managed"
  },
  "observations": {
    "camera": {
      "timestamp": "2026-11-15T07:00:00Z",
      "count": 25
    }
  },
  "now": "2026-11-15T07:05:00Z"
}
```
- **Response `200 OK`**:
```json
{
  "facility_id": "FAC-TEST",
  "zone_id": "ZONE-M1",
  "timestamp": "2026-11-15T07:05:00Z",
  "zone_type": "managed",
  "capacity": 60,
  "free_low": 27,
  "free_high": 43,
  "source": "camera",
  "confidence": 0.5337,
  "age_seconds": 300.0,
  "interval_level": 0.8
}
```

---

## 5. Integration Note for Member 3 (Backend & Simulator) and Member 4 (App)
The IoT event simulator and operator manual entry forms must serialize into this exact `OccupancyUpdate` format when publishing to the broker / Redis stream or returning from REST endpoints.

