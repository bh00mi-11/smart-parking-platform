"""
Shared Pydantic schemas for parking occupancy analytics and events.
Strict validation rules:
- 0 <= free_low <= free_high <= capacity
- For managed lots with a fresh exact source (age_seconds == 0, confidence == 1.0, and exact count), free_low == free_high.
"""

from datetime import datetime, timezone
from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


class OccupancyUpdate(BaseModel):
    facility_id: str = Field(..., description="Unique facility identifier")
    zone_id: str = Field(..., description="Unique zone identifier within facility")
    timestamp: datetime = Field(..., description="Timestamp of the update (ISO 8601 UTC)")
    zone_type: Literal["managed", "open"] = Field(..., description="'managed' (marked slots) or 'open' (open lot)")
    capacity: int = Field(..., ge=1, description="Total slot capacity of the zone")
    free_low: int = Field(..., ge=0, description="Lower bound of 80% interval for available free slots")
    free_high: int = Field(..., ge=0, description="Upper bound of 80% interval for available free slots")
    source: Literal["manual", "simulator", "camera"] = Field(..., description="Originating source of observation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Freshness and reliability score in [0, 1]")
    age_seconds: float = Field(..., ge=0.0, description="Age in seconds of the primary anchor observation")
    interval_level: float = Field(0.8, ge=0.0, le=1.0, description="Interval coverage level, default 0.8 (80%)")

    @model_validator(mode="after")
    def validate_bounds(self) -> "OccupancyUpdate":
        if not (0 <= self.free_low <= self.free_high <= self.capacity):
            raise ValueError(
                f"Bounds violation: 0 <= free_low ({self.free_low}) <= free_high ({self.free_high}) <= capacity ({self.capacity}) required"
            )
        # Managed lot exact collapse rule:
        # If managed zone has 0 age, full confidence, free_low must equal free_high
        if self.zone_type == "managed" and self.age_seconds == 0.0 and self.confidence >= 0.999:
            if self.free_low != self.free_high:
                raise ValueError(
                    f"Managed zones with fresh exact source must have free_low == free_high. Got free_low={self.free_low}, free_high={self.free_high}"
                )
        return self


class CheckInEvent(BaseModel):
    zone_id: str = Field(..., description="Zone identifier")
    timestamp: datetime = Field(..., description="ISO 8601 UTC timestamp of check-in/out event")
    type: Literal["in", "out"] = Field(..., description="Event type: 'in' or 'out'")
    user_id: str = Field(..., description="Anonymized user identifier")

