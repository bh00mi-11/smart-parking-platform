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

## 4. Integration Note for Member 3 (Backend & Simulator) and Member 4 (App)
The IoT event simulator and operator manual entry forms must serialize into this exact `OccupancyUpdate` format when publishing to the broker / Redis stream or returning from REST endpoints.

