"""
Tests for Vision FastAPI service endpoints (/vision/health, /vision/process-frame, /vision/estimate).
Uses Starlette TestClient with mocked and real models.
"""

import io
import json
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from vision.src.main_vision import app
import vision.src.api as api_mod

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_model_state():
    orig_model = api_mod._MODEL
    orig_path = api_mod._WEIGHTS_PATH
    yield
    api_mod._MODEL = orig_model
    api_mod._WEIGHTS_PATH = orig_path


def create_dummy_jpeg():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return io.BytesIO(buf.tobytes())


def test_health_success():
    mock_model = MagicMock(spec=torch.nn.Module)
    api_mod._MODEL = mock_model
    api_mod._WEIGHTS_PATH = "mocked/weights.pth"

    response = client.get("/vision/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_ready"] is True
    assert "mocked/weights.pth" in data["weights_path"]


def test_health_missing_weights():
    api_mod._MODEL = None
    with patch("vision.src.api.get_classifier_weights_path", side_effect=FileNotFoundError("No weights")):
        response = client.get("/vision/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_ready"] is False


def test_process_frame_success():
    mock_model = MagicMock(spec=torch.nn.Module)
    api_mod._MODEL = mock_model

    zone_config = {
        "facility_id": "FAC-TEST",
        "zone_id": "ZONE-01",
        "capacity": 2,
        "zone_type": "managed",
        "slots": [
            {"slot_id": "1", "points": [[10, 10], [50, 10], [50, 50], [10, 50]]},
            {"slot_id": "2", "points": [[60, 10], [90, 10], [90, 50], [60, 50]]}
        ]
    }

    with patch("vision.src.api.process_frame") as mock_process:
        from vision.src.schemas import OccupancyUpdate
        from datetime import datetime, timezone
        mock_process.return_value = OccupancyUpdate(
            facility_id="FAC-TEST",
            zone_id="ZONE-01",
            timestamp=datetime.now(timezone.utc),
            zone_type="managed",
            capacity=2,
            free_low=1,
            free_high=2,
            source="camera",
            confidence=0.9,
            age_seconds=0.0,
            interval_level=0.8
        )

        img_file = create_dummy_jpeg()
        response = client.post(
            "/vision/process-frame",
            files={"image": ("test.jpg", img_file, "image/jpeg")},
            data={"zone_config": json.dumps(zone_config)}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["facility_id"] == "FAC-TEST"
        assert data["zone_id"] == "ZONE-01"
        assert data["free_low"] == 1
        assert data["free_high"] == 2
        assert data["source"] == "camera"


def test_process_frame_bad_input():
    # Empty image
    zone_config = {"facility_id": "F1", "zone_id": "Z1", "capacity": 10, "zone_type": "open"}
    response = client.post(
        "/vision/process-frame",
        files={"image": ("test.jpg", b"", "image/jpeg")},
        data={"zone_config": json.dumps(zone_config)}
    )
    assert response.status_code == 400

    # Malformed JSON in zone_config
    img_file = create_dummy_jpeg()
    response = client.post(
        "/vision/process-frame",
        files={"image": ("test.jpg", img_file, "image/jpeg")},
        data={"zone_config": "not-valid-json{"}
    )
    assert response.status_code == 422


def test_estimate_success():
    req_payload = {
        "zone": {
            "facility_id": "FAC-TEST",
            "zone_id": "ZONE-01",
            "capacity": 50,
            "zone_type": "managed"
        },
        "observations": {
            "manual": {
                "timestamp": "2026-11-14T20:00:00Z",
                "count": 20
            }
        },
        "now": "2026-11-14T20:05:00Z"
    }

    response = client.post("/vision/estimate", json=req_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["facility_id"] == "FAC-TEST"
    assert data["capacity"] == 50
    assert data["free_low"] <= data["free_high"]
    assert data["source"] == "manual"

