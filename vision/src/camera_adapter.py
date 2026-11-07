"""
Camera adapter module connecting real camera frames to OccupancyUpdate via fusion.
Routes to:
- Managed zone: classify_slots() using MobileNetV3 slot classifier.
- Open zone: count_in_polygons() using YOLO11n vehicle detector.
"""

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
import torch
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from .classify_slots import classify_slots
from .fusion import FusionConfig, estimate_occupancy
from .schemas import OccupancyUpdate
from .yolo_count import count_in_polygons, count_vehicles

# Singleton cache for classifier model
_CLASSIFIER_MODEL: Optional[torch.nn.Module] = None


def get_classifier_weights_path(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolves the classifier checkpoint path from custom argument, environment variable,
    or standard location. Raises a clear error with placement instructions if missing.
    """
    if custom_path:
        p = Path(custom_path)
        if p.is_file():
            return p

    env_path = os.environ.get("MOBILENETV3_WEIGHTS_PATH")
    if env_path and Path(env_path).is_file():
        return Path(env_path)

    # Standard locations
    candidates = [
        Path("vision/outputs/mobilenetv3_best.pth"),
        Path("outputs/mobilenetv3_best.pth"),
        Path(__file__).resolve().parent.parent / "outputs" / "mobilenetv3_best.pth",
    ]
    for c in candidates:
        if c.is_file():
            return c

    raise FileNotFoundError(
        "MobileNetV3 slot classifier weights not found! "
        "Please train the model using 'python vision/src/train_classifier.py' or place the trained "
        "checkpoint at 'vision/outputs/mobilenetv3_best.pth' or set the MOBILENETV3_WEIGHTS_PATH "
        "environment variable pointing to the .pth file."
    )


def load_classifier_model(
    weights_path: Optional[Union[str, Path]] = None,
    device: Union[str, torch.device] = "cpu"
) -> torch.nn.Module:
    """
    Loads MobileNetV3-small slot classifier with 2-class head and trained weights.
    """
    global _CLASSIFIER_MODEL
    if _CLASSIFIER_MODEL is not None:
        return _CLASSIFIER_MODEL

    path = get_classifier_weights_path(weights_path)
    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, 2)

    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    _CLASSIFIER_MODEL = model
    return model


def process_frame(
    facility_id: str,
    zone_id: str,
    image: Union[np.ndarray, str, Path],
    zone_config: Dict[str, Any],
    now: Optional[Union[datetime, str]] = None,
    fusion_cfg: Optional[FusionConfig] = None,
    classifier_model: Optional[torch.nn.Module] = None,
    device: Union[str, torch.device] = "cpu",
    camera_reliability: Optional[float] = None
) -> OccupancyUpdate:
    """
    Processes a camera frame for a given zone and produces an OccupancyUpdate (source='camera').

    Args:
        facility_id: Facility identifier.
        zone_id: Zone identifier.
        image: Path or numpy array of the image frame.
        zone_config: Dict defining zone metadata:
            - 'zone_type': 'managed' or 'open'
            - 'capacity': int
            - 'slots': List[Dict] (for managed: [{'slot_id': str, 'points': [[x,y], ...]}])
            - 'zone_polygon': List[Tuple[int, int]] (optional for open zone)
        now: Optional evaluation timestamp (defaults to current UTC time).
        fusion_cfg: Optional FusionConfig.
        classifier_model: Optional pre-loaded classifier model (allows mocking in tests).
        device: 'cpu' or 'cuda'.

    Returns:
        OccupancyUpdate schema instance.
    """
    now_dt = datetime.now(timezone.utc) if now is None else now
    zone_type = zone_config.get("zone_type", "managed")
    capacity = int(zone_config["capacity"])

    if zone_type == "managed":
        slots = zone_config.get("slots", [])
        if not slots:
            raise ValueError("Managed zone requires a non-empty 'slots' list in zone_config.")

        # Load classifier model if not provided
        if classifier_model is None:
            classifier_model = load_classifier_model(device=device)

        slot_results = classify_slots(image, slots, model=classifier_model, device=device)

        # Extract slot probabilities for fusion
        # If classifier outputs occupied=1 and confidence=c, then p(occupied)=c if occ==1 else 1-c
        slot_probs = []
        for res in slot_results:
            conf = float(res.get("confidence", 0.9))
            occ = int(res.get("occupied", 0))
            p_occ = conf if occ == 1 else (1.0 - conf)
            slot_probs.append(p_occ)

        camera_obs = {
            "timestamp": now_dt,
            "count": sum(1 for r in slot_results if r.get("occupied", 0) == 1),
            "slot_probs": slot_probs
        }

    else:
        # Open zone: vehicle detection
        zone_polygon = zone_config.get("zone_polygon")
        if zone_polygon is not None and len(zone_polygon) >= 3:
            count, _ = count_in_polygons(image, [zone_polygon])
        else:
            count, _ = count_vehicles(image)

        camera_obs = {
            "timestamp": now_dt,
            "count": count,
            "slot_probs": None
        }

    zone_info = {
        "facility_id": facility_id,
        "zone_id": zone_id,
        "capacity": capacity,
        "zone_type": zone_type
    }

    if fusion_cfg is None:
        fusion_cfg = FusionConfig()
    if camera_reliability is not None:
        fusion_cfg.rel_camera_managed = min(0.999, max(0.0, float(camera_reliability)))
        fusion_cfg.rel_camera_open = min(0.999, max(0.0, float(camera_reliability)))

    update = estimate_occupancy(
        zone=zone_info,
        observations={"camera": camera_obs},
        now=now_dt,
        config=fusion_cfg
    )
    return update

