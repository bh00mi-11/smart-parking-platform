"""
Unit tests for probabilistic fusion behavior with freshness-based confidence decay.
"""

from datetime import datetime, timedelta, timezone
import pytest

from vision.src.fusion import FusionConfig, estimate_occupancy
from vision.src.schemas import CheckInEvent


@pytest.fixture
def managed_zone():
    return {
        "facility_id": "FAC-TEST",
        "zone_id": "ZONE-MANAGED",
        "capacity": 50,
        "zone_type": "managed"
    }


@pytest.fixture
def open_zone():
    return {
        "facility_id": "FAC-TEST",
        "zone_id": "ZONE-OPEN",
        "capacity": 50,
        "zone_type": "open"
    }


def test_confidence_decreases_with_age(open_zone):
    now = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)
    config = FusionConfig(tau_seconds=600.0)

    # Fresh observation (age = 0)
    obs_fresh = {"manual": {"timestamp": now, "count": 20}}
    res_fresh = estimate_occupancy(open_zone, obs_fresh, now=now, config=config)

    # Aged observation (age = 10 mins = 600s)
    obs_aged = {"manual": {"timestamp": now - timedelta(seconds=600), "count": 20}}
    res_aged = estimate_occupancy(open_zone, obs_aged, now=now, config=config)

    # Very old observation (age = 30 mins)
    obs_old = {"manual": {"timestamp": now - timedelta(seconds=1800), "count": 20}}
    res_old = estimate_occupancy(open_zone, obs_old, now=now, config=config)

    assert res_fresh.confidence > res_aged.confidence > res_old.confidence
    assert res_aged.age_seconds == pytest.approx(600.0, abs=1.0)


def test_range_widens_with_age(open_zone):
    now = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)
    config = FusionConfig(variance_drift_per_min=0.5)

    obs_fresh = {"manual": {"timestamp": now, "count": 25}}
    res_fresh = estimate_occupancy(open_zone, obs_fresh, now=now, config=config)
    width_fresh = res_fresh.free_high - res_fresh.free_low

    obs_aged = {"manual": {"timestamp": now - timedelta(minutes=25), "count": 25}}
    res_aged = estimate_occupancy(open_zone, obs_aged, now=now, config=config)
    width_aged = res_aged.free_high - res_aged.free_low

    assert width_aged > width_fresh


def test_fresh_camera_reading_outweighs_stale_manual(open_zone):
    now = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)

    # Stale manual report from 40 mins ago saying 10 occupied
    manual_stale = {"timestamp": now - timedelta(minutes=40), "count": 10}
    # Fresh camera reading right now saying 40 occupied
    cam_fresh = {"timestamp": now, "count": 40}

    obs = {
        "manual": manual_stale,
        "camera": cam_fresh
    }
    res = estimate_occupancy(open_zone, obs, now=now)
    est_occupied = open_zone["capacity"] - ((res.free_low + res.free_high) / 2.0)

    # Fused estimate should be close to 40 (the fresh reading) and far from 10
    assert abs(est_occupied - 40) < abs(est_occupied - 10)
    assert res.source == "camera"


def test_check_ins_shift_estimate(open_zone):
    now = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)
    anchor_time = now - timedelta(minutes=10)

    # Anchor says 20 occupied (free = 30)
    obs_no_events = {
        "manual": {"timestamp": anchor_time, "count": 20}
    }
    res_base = estimate_occupancy(open_zone, obs_no_events, now=now)
    base_free = (res_base.free_low + res_base.free_high) / 2.0

    # 5 cars check in after anchor
    events = [
        CheckInEvent(zone_id=open_zone["zone_id"], timestamp=anchor_time + timedelta(minutes=2), type="in", user_id=f"u_{i}")
        for i in range(5)
    ]
    obs_with_events = {
        "manual": {"timestamp": anchor_time, "count": 20},
        "check_ins": events
    }
    res_shifted = estimate_occupancy(open_zone, obs_with_events, now=now)
    shifted_free = (res_shifted.free_low + res_shifted.free_high) / 2.0

    # Since 5 cars checked in, free spots should decrease
    assert shifted_free < base_free
    assert abs((base_free - shifted_free) - 5) <= 1.5


def test_outputs_stay_within_capacity(open_zone):
    now = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)
    cap = open_zone["capacity"]

    # Extreme over-occupancy reported
    obs_extreme_high = {"manual": {"timestamp": now, "count": 999}}
    res_high = estimate_occupancy(open_zone, obs_extreme_high, now=now)
    assert 0 <= res_high.free_low <= res_high.free_high <= cap

    # Negative occupancy reported
    obs_extreme_low = {"manual": {"timestamp": now, "count": -50}}
    res_low = estimate_occupancy(open_zone, obs_extreme_low, now=now)
    assert 0 <= res_low.free_low <= res_low.free_high <= cap


def test_no_observation_fallback_to_prior(open_zone):
    now = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)
    obs_prior_only = {
        "prior": {"mean": 25.0, "std": 5.0}
    }
    res = estimate_occupancy(open_zone, obs_prior_only, now=now)
    # Confidence should be low for prior only
    assert res.confidence <= 0.25
    assert 0 <= res.free_low <= res.free_high <= open_zone["capacity"]

    # Empty observation dict fallback
    res_empty = estimate_occupancy(open_zone, {}, now=now)
    assert res_empty.confidence <= 0.25
    assert 0 <= res_empty.free_low <= res_empty.free_high <= open_zone["capacity"]

