"""
Probabilistic occupancy fusion module with freshness-based confidence decay.
Fuses heterogeneous observations (camera, manual counts, check-in/out events, historical prior)
using inverse-variance weighting and temporal uncertainty diffusion.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any, Callable, Dict, List, Optional, Union

from .schemas import CheckInEvent, OccupancyUpdate


@dataclass
class FusionConfig:
    """
    Fusion hyperparameters.
    NOTE: These are starting values, tuned on tuning seeds only, not claims of optimal real-world performance.
    """
    # Source intrinsic reliabilities R in [0, 1]
    rel_camera_managed: float = 0.90
    rel_camera_open: float = 0.88
    rel_manual: float = 0.95
    rel_prior: float = 0.20

    # Half-life / decay constant tau (in seconds) for freshness
    # e.g., 600s = 10 min decay constant
    tau_seconds: float = 600.0

    # Observation base standard deviations (variance = sigma^2)
    # Calibrated against empirical PKLot camera observation noise
    sigma_sys_camera_managed: float = 0.3     # Systematic slot classifier error
    sigma_camera_open: float = 4.0            # YOLO open-zone detection count error
    sigma_manual: float = 0.5                 # Manual operator entry error
    sigma_event_rate: float = 0.1             # Extra variance per check-in/out event (drop / miscount rate)

    # Uncertainty temporal drift rate in variance units (slots^2) per minute
    variance_drift_per_min: float = 3.0

    # Managed lot fresh exact threshold (seconds)
    managed_exact_age_threshold_sec: float = 5.0

    # Normal distribution z-score for 80% symmetric interval (calibrated on tuning seeds [101, 102])
    z_80: float = 1.38


def _to_utc_timestamp(dt: Union[datetime, str]) -> datetime:
    if isinstance(dt, str):
        # Support ISO strings like 2026-11-03T21:30:00Z
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def estimate_occupancy(
    zone: Dict[str, Any],
    observations: Dict[str, Any],
    now: Union[datetime, str],
    config: Optional[FusionConfig] = None
) -> OccupancyUpdate:
    """
    Estimate parking occupancy by fusing camera observations, manual updates,
    check-in/out events, and a historical prior.

    Args:
        zone: Dict containing:
            - 'facility_id' (str)
            - 'zone_id' (str)
            - 'capacity' (int)
            - 'zone_type' ("managed" or "open")
        observations: Dict containing any of:
            - 'camera': Dict with:
                - 'timestamp': datetime or ISO str
                - 'count': Optional[int or float]
                - 'slot_probs': Optional[List[float]] (occupied probability per slot)
            - 'manual': Dict with:
                - 'timestamp': datetime or ISO str
                - 'count': int or float
            - 'check_ins': Optional[List[Union[CheckInEvent, Dict]]]
            - 'prior': Optional Dict with {'mean': float, 'std': float} or
                       Callable[[datetime], Dict[str, float]]
        now: Current timestamp (datetime or ISO str)
        config: Optional FusionConfig instance

    Returns:
        OccupancyUpdate object adhering to shared schema.
    """
    if config is None:
        config = FusionConfig()

    now_dt = _to_utc_timestamp(now)
    capacity = int(zone["capacity"])
    zone_type = zone.get("zone_type", "managed")
    facility_id = zone.get("facility_id", "FAC-DEFAULT")
    zone_id = zone.get("zone_id", "ZONE-DEFAULT")

    # Step 1: Collect individual estimates as (mean, variance)
    fused_estimates = []  # List of (mean, var, weight)
    freshest_anchor_source = None
    freshest_anchor_age = float("inf")
    freshest_anchor_rel = config.rel_prior

    # 1A. Camera Observation
    cam_obs = observations.get("camera")
    if cam_obs and "timestamp" in cam_obs:
        cam_dt = _to_utc_timestamp(cam_obs["timestamp"])
        cam_age = max(0.0, (now_dt - cam_dt).total_seconds())
        slot_probs = cam_obs.get("slot_probs")

        if zone_type == "managed" and slot_probs is not None and len(slot_probs) > 0:
            # Managed slot-level probabilities: expected occupancy = sum(p)
            cam_mean = float(sum(slot_probs))
            # Bernoulli variance sum + systematic error
            bernoulli_var = sum(p * (1.0 - p) for p in slot_probs)
            cam_var = bernoulli_var + (config.sigma_sys_camera_managed ** 2)
            cam_rel = config.rel_camera_managed
        else:
            # Open zone or direct count
            cam_count = float(cam_obs.get("count", 0.0))
            cam_mean = cam_count
            cam_var = config.sigma_camera_open ** 2
            cam_rel = config.rel_camera_open

        # Propagate forward in time: variance drift
        drift_var = (cam_age / 60.0) * config.variance_drift_per_min
        cam_var_propagated = cam_var + drift_var

        fused_estimates.append(("camera", cam_mean, cam_var_propagated, cam_age))

        if cam_age < freshest_anchor_age:
            freshest_anchor_age = cam_age
            freshest_anchor_source = "camera"
            freshest_anchor_rel = cam_rel

    # 1B. Manual Observation
    man_obs = observations.get("manual")
    if man_obs and "timestamp" in man_obs and "count" in man_obs:
        man_dt = _to_utc_timestamp(man_obs["timestamp"])
        man_age = max(0.0, (now_dt - man_dt).total_seconds())
        man_mean = float(man_obs["count"])
        man_var = config.sigma_manual ** 2
        drift_var = (man_age / 60.0) * config.variance_drift_per_min
        man_var_propagated = man_var + drift_var

        fused_estimates.append(("manual", man_mean, man_var_propagated, man_age))

        if man_age < freshest_anchor_age:
            freshest_anchor_age = man_age
            freshest_anchor_source = "manual"
            freshest_anchor_rel = config.rel_manual

    # 1C. Check-in/out delta integration relative to the freshest anchor
    check_ins = observations.get("check_ins", [])
    if check_ins and freshest_anchor_source is not None:
        # Find anchor time
        if freshest_anchor_source == "camera":
            anchor_dt = _to_utc_timestamp(cam_obs["timestamp"])
        else:
            anchor_dt = _to_utc_timestamp(man_obs["timestamp"])

        net_delta = 0
        total_events = 0
        for ev in check_ins:
            ev_dt = _to_utc_timestamp(ev.timestamp if isinstance(ev, CheckInEvent) else ev["timestamp"])
            if ev_dt > anchor_dt:
                ev_type = ev.type if isinstance(ev, CheckInEvent) else ev["type"]
                if ev_type == "in":
                    net_delta += 1
                elif ev_type == "out":
                    net_delta -= 1
                total_events += 1

        if total_events > 0:
            # Adjust the newest anchor mean with the net check-ins
            # Event variance grows with number of events and event rate variance
            # We replace/update the anchor in fused_estimates
            updated_estimates = []
            for name, m, v, age in fused_estimates:
                if name == freshest_anchor_source:
                    new_mean = m + net_delta
                    new_var = v + total_events * (config.sigma_event_rate ** 2)
                    updated_estimates.append((name, new_mean, new_var, age))
                else:
                    updated_estimates.append((name, m, v, age))
            fused_estimates = updated_estimates

    # 1D. Historical Prior
    prior_info = observations.get("prior")
    prior_mean: Optional[float] = None
    prior_std: Optional[float] = None

    if callable(prior_info):
        res = prior_info(now_dt)
        prior_mean = float(res["mean"])
        prior_std = float(res["std"])
    elif isinstance(prior_info, dict):
        prior_mean = float(prior_info["mean"])
        prior_std = float(prior_info["std"])

    if prior_mean is not None and prior_std is not None and prior_std > 0:
        prior_var = prior_std ** 2
        fused_estimates.append(("prior", prior_mean, prior_var, float("inf")))

    # Step 2: Combine using Inverse-Variance Weighting
    if not fused_estimates:
        # Absolute fallback if neither observation nor prior was supplied
        prior_mean = capacity * 0.5
        prior_var = (capacity * 0.3) ** 2
        fused_estimates.append(("prior", prior_mean, prior_var, float("inf")))

    sum_weights = 0.0
    weighted_mean_sum = 0.0

    for name, mean_val, var_val, _ in fused_estimates:
        safe_var = max(var_val, 1e-4)
        weight = 1.0 / safe_var
        sum_weights += weight
        weighted_mean_sum += weight * mean_val

    fused_mean = weighted_mean_sum / sum_weights
    fused_var = 1.0 / sum_weights
    fused_sd = math.sqrt(fused_var)

    # Step 3: Compute 80% interval and free slot bounds
    # Mean occupied count
    occ_mean = max(0.0, min(float(capacity), fused_mean))

    # Free count is capacity - occupied
    free_nominal = capacity - occ_mean

    # Check for managed lot exact collapse:
    # A fresh exact source in a managed lot should collapse the range to a single value.
    is_fresh_exact_managed = (
        zone_type == "managed"
        and freshest_anchor_source in ("manual", "gate")
        and freshest_anchor_age <= config.managed_exact_age_threshold_sec
    )

    if is_fresh_exact_managed:
        exact_free = int(round(free_nominal))
        exact_free = max(0, min(capacity, exact_free))
        free_low = exact_free
        free_high = exact_free
        # Fresh exact source sets age to 0.0 and confidence to full 1.0
        final_confidence = 1.0
        final_age = 0.0
        final_source = freshest_anchor_source
    else:
        # Standard 80% interval: [mean - 1.28*sd, mean + 1.28*sd] for occupied
        # Free slots: [capacity - (mean + 1.28*sd), capacity - (mean - 1.28*sd)]
        free_est_low = capacity - (occ_mean + config.z_80 * fused_sd)
        free_est_high = capacity - (occ_mean - config.z_80 * fused_sd)

        free_low = int(math.floor(free_est_low))
        free_high = int(math.ceil(free_est_high))

        # Clamp to [0, capacity] and ensure free_low <= free_high
        free_low = max(0, min(capacity, free_low))
        free_high = max(free_low, min(capacity, free_high))

        if freshest_anchor_source is not None:
            final_age = freshest_anchor_age
            final_source = freshest_anchor_source
            # Exponential freshness decay
            decay = math.exp(-final_age / max(config.tau_seconds, 1.0))
            final_confidence = round(float(freshest_anchor_rel * decay), 4)
            final_confidence = max(0.0, min(1.0, final_confidence))
        else:
            final_age = 3600.0
            final_source = "simulator"  # default fallback source
            final_confidence = round(config.rel_prior, 4)

    return OccupancyUpdate(
        facility_id=facility_id,
        zone_id=zone_id,
        timestamp=now_dt,
        zone_type=zone_type,
        capacity=capacity,
        free_low=free_low,
        free_high=free_high,
        source=final_source,
        confidence=final_confidence,
        age_seconds=round(final_age, 2),
        interval_level=0.8
    )

