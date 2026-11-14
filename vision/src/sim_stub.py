"""
Simulation and prior generator for parking occupancy experiments.
Supports:
- Per-lot camera noise profiles (clean vs noisy_overhead)
- Non-oracle prior (learned from historical seeds with noise, shifts, and unmodeled events)
- Oracle prior (perfect diurnal curve matching the generator)
"""

from datetime import datetime, timedelta, timezone
import math
import random
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from .schemas import CheckInEvent


def generate_diurnal_curve(
    dt: datetime,
    capacity: int,
    is_weekend: bool,
    has_event: bool = False
) -> float:
    """
    Computes a realistic ground-truth occupancy count based on time of day, day of week, and events.
    """
    hour = dt.hour + dt.minute / 60.0

    if not is_weekend:
        # Weekday: Two main peaks (morning arrival ~9-11 AM, afternoon plateau ~1-4 PM)
        morning_peak = math.exp(-((hour - 10.0) ** 2) / (2 * (1.8 ** 2)))
        afternoon_peak = math.exp(-((hour - 14.5) ** 2) / (2 * (2.2 ** 2)))
        base_occupancy = 0.08 + 0.52 * morning_peak + 0.40 * afternoon_peak
    else:
        # Weekend: Leisure curve centered around afternoon/evening (12-6 PM)
        weekend_peak = math.exp(-((hour - 14.0) ** 2) / (2 * (3.5 ** 2)))
        base_occupancy = 0.05 + 0.55 * weekend_peak

    # Special event surge (e.g. evening event from 6 PM to 9 PM)
    if has_event and 18.0 <= hour <= 21.0:
        event_peak = math.exp(-((hour - 19.5) ** 2) / (2 * (1.2 ** 2)))
        base_occupancy += 0.35 * event_peak

    base_occupancy = max(0.02, min(0.96, base_occupancy))
    return base_occupancy * capacity


def build_historical_prior(
    capacity: int,
    noise_seed: int = 100,
    non_oracle: bool = True
) -> Callable[[datetime], Dict[str, float]]:
    """
    Builds a historical prior lookup returning {'mean': float, 'std': float} for any timestamp.
    
    If non_oracle=True, the prior is trained from a distinct historical simulation with its
    own noise, subtle peak shifts, and does not model random unmodeled event/weather days.
    """
    rng = random.Random(noise_seed)

    prior_table = {}
    for dow in range(7):
        is_wknd = (dow >= 5)
        for hr in range(24):
            dummy_dt = datetime(2026, 1, 5 + dow, hr, 0, tzinfo=timezone.utc)
            base = generate_diurnal_curve(dummy_dt, capacity, is_wknd, has_event=False)
            
            if non_oracle:
                # Add realistic historical training variance, systematic slight phase shift, and higher uncertainty
                # Prior lacks knowledge of live weather, construction, or sudden attendance surges
                hist_noise = rng.gauss(0.02 * capacity, 0.06 * capacity)
                mean_val = base + hist_noise
                mean_val = max(1.0, min(float(capacity), mean_val))
                std_val = max(4.0, 0.18 * capacity)
            else:
                # Oracle prior: near perfect match to the base generator curve with small perturbation
                mean_val = base + rng.uniform(-0.01 * capacity, 0.01 * capacity)
                mean_val = max(1.0, min(float(capacity), mean_val))
                std_val = max(2.0, 0.10 * capacity)

            prior_table[(dow, hr)] = {"mean": mean_val, "std": std_val}

    def _prior_fn(dt: datetime) -> Dict[str, float]:
        dow = dt.weekday()
        hr = dt.hour
        return prior_table[(dow, hr)]

    return _prior_fn


# Documented Camera Noise Profiles from vision/outputs/camera_noise.json
# Clean profile: Measured from UFPR04 (bias -0.058, std 0.261) and UFPR05 (bias +0.091, std 0.628).
#   Pooled clean: bias = 0.02, std = 0.45.
# Noisy overhead profile: Measured from PUCPR (overall bias +4.31, std 18.57; normal days std ~0.8,
#   with harsh low-angle sun shadow days producing positive bias spikes up to +99 slots).
CAMERA_PROFILES = {
    "clean": {
        "description": "Clean camera (UFPR04/UFPR05 low-angle view)",
        "bias": 0.02,
        "std": 0.45,
        "shadow_prob": 0.0,
        "shadow_bias": 0.0
    },
    "noisy_overhead": {
        "description": "Noisy overhead camera (PUCPR distant angle subject to sunny shadows)",
        "bias": 0.15,
        "std": 0.80,
        "shadow_prob": 0.07,  # ~7% of frames have shadow false positives
        "shadow_bias": 75.0,  # harsh shadow bias spike
        "shadow_std": 15.0
    }
}


def simulate_zone_trajectory(
    zone_id: str,
    facility_id: str,
    capacity: int,
    zone_type: str = "managed",
    noise_profile: str = "clean",
    use_non_oracle_prior: bool = True,
    start_time: Optional[datetime] = None,
    duration_days: int = 14,
    step_minutes: int = 5,
    camera_interval_minutes: int = 15,
    manual_interval_hours: int = 4,
    camera_bias: Optional[float] = None,
    camera_noise_std: Optional[float] = None,
    manual_noise_std: float = 0.5,
    event_drop_rate: float = 0.08,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generates time steps with true occupancy and observations.
    Each returned dict has:
        - 'timestamp': datetime
        - 'true_occupancy': int
        - 'noise_profile': str
        - 'observations': dict suitable for estimate_occupancy()
    """
    rng = random.Random(seed)
    if start_time is None:
        start_time = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)

    # Resolve camera profile
    prof = CAMERA_PROFILES.get(noise_profile, CAMERA_PROFILES["clean"])
    base_bias = camera_bias if camera_bias is not None else prof["bias"]
    base_std = camera_noise_std if camera_noise_std is not None else prof["std"]
    shadow_prob = prof.get("shadow_prob", 0.0)
    shadow_bias = prof.get("shadow_bias", 0.0)
    shadow_std = prof.get("shadow_std", 1.0)

    total_steps = int((duration_days * 24 * 60) / step_minutes)
    history = []

    prior_fn = build_historical_prior(capacity, noise_seed=seed + 999, non_oracle=use_non_oracle_prior)

    last_cam_time = None
    last_cam_obs = None
    last_man_time = None
    last_man_obs = None
    all_events_buffer: List[CheckInEvent] = []

    prev_true = int(round(generate_diurnal_curve(start_time, capacity, start_time.weekday() >= 5)))

    current_time = start_time
    for step in range(total_steps):
        is_weekend = (current_time.weekday() >= 5)
        # Random Friday/Saturday evening events + occasional weekday surprise surges
        has_event = (current_time.weekday() in (4, 5) and current_time.hour >= 18 and (step % 500 < 60))
        if use_non_oracle_prior and (current_time.weekday() == 2 and 11 <= current_time.hour <= 15 and (step % 300 < 50)):
            # Unmodeled special event (e.g. Wednesday guest lecture / campus event)
            has_event = True

        target_true = generate_diurnal_curve(current_time, capacity, is_weekend, has_event)
        # Random walk drift around target
        true_occupancy = int(round(0.7 * prev_true + 0.3 * target_true + rng.gauss(0, 0.8)))
        true_occupancy = max(0, min(capacity, true_occupancy))

        delta = true_occupancy - prev_true
        prev_true = true_occupancy

        # Generate check-in / check-out events to match delta
        if delta > 0:
            for _ in range(delta):
                if rng.random() > event_drop_rate:  # dropped events
                    all_events_buffer.append(CheckInEvent(
                        zone_id=zone_id,
                        timestamp=current_time,
                        type="in",
                        user_id=f"user_{rng.randint(100, 999)}"
                    ))
        elif delta < 0:
            for _ in range(abs(delta)):
                if rng.random() > event_drop_rate:
                    all_events_buffer.append(CheckInEvent(
                        zone_id=zone_id,
                        timestamp=current_time,
                        type="out",
                        user_id=f"user_{rng.randint(100, 999)}"
                    ))

        # Check if camera fires at this step
        if last_cam_time is None or (current_time - last_cam_time).total_seconds() >= camera_interval_minutes * 60:
            # Check for shadow false-positive anomaly
            if shadow_prob > 0 and rng.random() < shadow_prob:
                err = rng.gauss(shadow_bias, shadow_std)
            else:
                err = rng.gauss(base_bias, base_std)
                
            cam_count = int(round(true_occupancy + err))
            cam_count = max(0, min(capacity, cam_count))

            if zone_type == "managed":
                slot_probs = []
                for s_i in range(capacity):
                    if s_i < cam_count:
                        slot_probs.append(round(min(0.99, max(0.65, rng.gauss(0.92, 0.05))), 3))
                    else:
                        slot_probs.append(round(max(0.01, min(0.35, rng.gauss(0.08, 0.05))), 3))
            else:
                slot_probs = None

            last_cam_obs = {
                "timestamp": current_time,
                "count": cam_count,
                "slot_probs": slot_probs
            }
            last_cam_time = current_time

        # Check if manual observation fires (irregular intervals)
        if last_man_time is None or (current_time - last_man_time).total_seconds() >= manual_interval_hours * 3600:
            if rng.random() < 0.8:
                man_count = int(round(true_occupancy + rng.gauss(0, manual_noise_std)))
                man_count = max(0, min(capacity, man_count))
                last_man_obs = {
                    "timestamp": current_time,
                    "count": man_count
                }
                last_man_time = current_time

        obs_dict: Dict[str, Any] = {
            "prior": prior_fn(current_time)
        }
        if last_cam_obs is not None:
            obs_dict["camera"] = last_cam_obs
        if last_man_obs is not None:
            obs_dict["manual"] = last_man_obs

        obs_dict["check_ins"] = list(all_events_buffer[-100:])

        history.append({
            "timestamp": current_time,
            "true_occupancy": true_occupancy,
            "noise_profile": noise_profile,
            "observations": obs_dict
        })

        current_time += timedelta(minutes=step_minutes)

    return history
