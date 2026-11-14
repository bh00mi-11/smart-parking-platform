"""
Synthetic simulation stub generating realistic parking occupancy time-series and observations.
Acts as a replaceable placeholder for Member 3's backend simulator.

Generates ground-truth occupancy curves incorporating:
- Diurnal commute / rush-hour peaks (morning arrival, evening departure)
- Weekend vs weekday profile modulation
- Special event attendance surges
- Noisy observation channels:
    * Camera frames at periodic intervals with classifier/detection noise
    * Manual operator entries at irregular timestamps with human counting error
    * Check-in / check-out event streams with dropped-event rates
    * Historical prior profile (mean and variance by hour of week)
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
    noise_seed: int = 100
) -> Callable[[datetime], Dict[str, float]]:
    """
    Builds a historical prior lookup returning {'mean': float, 'std': float} for any timestamp.
    """
    rng = random.Random(noise_seed)

    # Precompute 24 x 7 mean profiles
    prior_table = {}
    for dow in range(7):
        is_wknd = (dow >= 5)
        for hr in range(24):
            dummy_dt = datetime(2026, 1, 5 + dow, hr, 0, tzinfo=timezone.utc)
            base = generate_diurnal_curve(dummy_dt, capacity, is_wknd, has_event=False)
            # Add slight historical noise
            mean_val = base + rng.uniform(-0.03 * capacity, 0.03 * capacity)
            mean_val = max(1.0, min(float(capacity), mean_val))
            std_val = max(2.0, 0.12 * capacity)
            prior_table[(dow, hr)] = {"mean": mean_val, "std": std_val}

    def _prior_fn(dt: datetime) -> Dict[str, float]:
        dow = dt.weekday()
        hr = dt.hour
        return prior_table[(dow, hr)]

    return _prior_fn


def simulate_zone_trajectory(
    zone_id: str,
    facility_id: str,
    capacity: int,
    zone_type: str = "managed",
    start_time: Optional[datetime] = None,
    duration_days: int = 14,
    step_minutes: int = 5,
    camera_interval_minutes: int = 15,
    manual_interval_hours: int = 4,
    camera_bias: float = 1.45,
    camera_noise_std: float = 10.92,
    manual_noise_std: float = 0.5,
    event_drop_rate: float = 0.08,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generates time steps with true occupancy and observations.
    Each returned dict has:
        - 'timestamp': datetime
        - 'true_occupancy': int
        - 'observations': dict suitable for estimate_occupancy()
    """
    rng = random.Random(seed)
    if start_time is None:
        start_time = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)

    total_steps = int((duration_days * 24 * 60) / step_minutes)
    history = []

    prior_fn = build_historical_prior(capacity, noise_seed=seed + 999)

    last_cam_time = None
    last_cam_obs = None
    last_man_time = None
    last_man_obs = None
    all_events_buffer: List[CheckInEvent] = []

    prev_true = int(round(generate_diurnal_curve(start_time, capacity, start_time.weekday() >= 5)))

    current_time = start_time
    for step in range(total_steps):
        is_weekend = (current_time.weekday() >= 5)
        # Random Friday/Saturday evening events
        has_event = (current_time.weekday() in (4, 5) and current_time.hour >= 18 and (step % 500 < 60))

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
            cam_count = int(round(true_occupancy + camera_bias + rng.gauss(0, camera_noise_std)))
            cam_count = max(0, min(capacity, cam_count))

            if zone_type == "managed":
                # Generate slot probabilities summing near cam_count
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
            if rng.random() < 0.8:  # 80% chance of manual entry when due
                man_count = int(round(true_occupancy + rng.gauss(0, manual_noise_std)))
                man_count = max(0, min(capacity, man_count))
                last_man_obs = {
                    "timestamp": current_time,
                    "count": man_count
                }
                last_man_time = current_time

        # Active events relative to anchor
        obs_dict: Dict[str, Any] = {
            "prior": prior_fn(current_time)
        }
        if last_cam_obs is not None:
            obs_dict["camera"] = last_cam_obs
        if last_man_obs is not None:
            obs_dict["manual"] = last_man_obs

        # Include recent events
        obs_dict["check_ins"] = list(all_events_buffer[-100:])

        history.append({
            "timestamp": current_time,
            "true_occupancy": true_occupancy,
            "observations": obs_dict
        })

        current_time += timedelta(minutes=step_minutes)

    return history

