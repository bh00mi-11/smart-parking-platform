"""
Unit tests for camera adapter with a mocked MobileNetV3 classifier.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock
import numpy as np
import pytest
import torch
import torch.nn as nn

from vision.src.camera_adapter import process_frame


class MockClassifier(nn.Module):
    """
    Mock classifier outputting high confidence logits for occupied (class 1).
    """
    def __init__(self, occupied_logits: bool = True):
        super().__init__()
        self.occupied_logits = occupied_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        if self.occupied_logits:
            # High logit for occupied (class 1)
            return torch.tensor([[-2.0, 5.0]] * batch_size)
        else:
            # High logit for free (class 0)
            return torch.tensor([[5.0, -2.0]] * batch_size)


def test_process_frame_managed_mock_classifier():
    # 100x100 blank dummy frame
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    slots = [
        {"slot_id": "S1", "points": [[10, 10], [40, 10], [40, 40], [10, 40]]},
        {"slot_id": "S2", "points": [[50, 10], [80, 10], [80, 40], [50, 40]]},
        {"slot_id": "S3", "points": [[10, 50], [40, 50], [40, 80], [10, 80]]},
    ]
    zone_config = {
        "zone_type": "managed",
        "capacity": 3,
        "slots": slots
    }

    mock_model = MockClassifier(occupied_logits=True)

    now = datetime(2026, 11, 3, 15, 0, tzinfo=timezone.utc)
    update = process_frame(
        facility_id="FAC-TEST",
        zone_id="ZONE-M",
        image=dummy_frame,
        zone_config=zone_config,
        now=now,
        classifier_model=mock_model,
        device="cpu"
    )

    assert update.facility_id == "FAC-TEST"
    assert update.zone_id == "ZONE-M"
    assert update.capacity == 3
    assert update.source == "camera"
    # All 3 slots detected as occupied -> free_low == free_high == 0
    assert update.free_low == 0
    assert update.free_high == 1
    assert update.confidence == 0.90


def test_process_frame_missing_slots_managed_error():
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    zone_config = {
        "zone_type": "managed",
        "capacity": 5,
        "slots": []  # Missing slots
    }

    with pytest.raises(ValueError):
        process_frame(
            facility_id="FAC-TEST",
            zone_id="ZONE-M",
            image=dummy_frame,
            zone_config=zone_config
        )



def test_camera_adapter_configurable_reliability():
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    slots = [
        {"slot_id": "S1", "points": [[10, 10], [40, 10], [40, 40], [10, 40]]},
        {"slot_id": "S2", "points": [[50, 10], [80, 10], [80, 40], [50, 40]]},
    ]
    zone_config = {"zone_type": "managed", "capacity": 2, "slots": slots}
    mock_model = MockClassifier(occupied_logits=True)
    update = process_frame(
        facility_id="FAC-TEST",
        zone_id="ZONE-M",
        image=dummy_frame,
        zone_config=zone_config,
        classifier_model=mock_model,
        camera_reliability=0.85
    )
    assert update.confidence == 0.85
    assert update.free_low <= update.free_high
    assert (update.free_high - update.free_low) >= 1
