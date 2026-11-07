"""
Unit tests for OccupancyUpdate and CheckInEvent Pydantic schemas.
"""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from vision.src.schemas import CheckInEvent, OccupancyUpdate


def test_valid_occupancy_update():
    now = datetime.now(timezone.utc)
    update = OccupancyUpdate(
        facility_id="FAC-1",
        zone_id="ZONE-A",
        timestamp=now,
        zone_type="open",
        capacity=50,
        free_low=10,
        free_high=20,
        source="camera",
        confidence=0.85,
        age_seconds=30.0,
        interval_level=0.8
    )
    assert update.capacity == 50
    assert update.free_low == 10
    assert update.free_high == 20
    assert update.confidence == 0.85


def test_invalid_bounds_free_low_greater_than_high():
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        OccupancyUpdate(
            facility_id="FAC-1",
            zone_id="ZONE-A",
            timestamp=now,
            zone_type="open",
            capacity=50,
            free_low=25,
            free_high=15,  # Violates low <= high
            source="camera",
            confidence=0.8,
            age_seconds=10.0
        )


def test_invalid_bounds_free_high_exceeds_capacity():
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        OccupancyUpdate(
            facility_id="FAC-1",
            zone_id="ZONE-A",
            timestamp=now,
            zone_type="open",
            capacity=30,
            free_low=5,
            free_high=35,  # Exceeds capacity
            source="manual",
            confidence=0.9,
            age_seconds=5.0
        )


def test_managed_lot_fresh_exact_source_collapse():
    now = datetime.now(timezone.utc)
    # Fresh exact source in managed lot must have free_low == free_high
    with pytest.raises(ValidationError):
        OccupancyUpdate(
            facility_id="FAC-1",
            zone_id="ZONE-M",
            timestamp=now,
            zone_type="managed",
            capacity=40,
            free_low=10,
            free_high=12,  # Not collapsed despite fresh exact
            source="camera",
            confidence=1.0,
            age_seconds=0.0
        )

    # Valid collapsed
    valid_managed = OccupancyUpdate(
        facility_id="FAC-1",
        zone_id="ZONE-M",
        timestamp=now,
        zone_type="managed",
        capacity=40,
        free_low=12,
        free_high=12,
        source="camera",
        confidence=1.0,
        age_seconds=0.0
    )
    assert valid_managed.free_low == valid_managed.free_high


def test_check_in_event_schema():
    now = datetime.now(timezone.utc)
    ev = CheckInEvent(
        zone_id="ZONE-A",
        timestamp=now,
        type="in",
        user_id="user_123"
    )
    assert ev.type == "in"
    assert ev.user_id == "user_123"

    with pytest.raises(ValidationError):
        CheckInEvent(
            zone_id="ZONE-A",
            timestamp=now,
            type="invalid_type",  # only 'in' or 'out'
            user_id="user_123"
        )

