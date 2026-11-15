"""
FastAPI router for Computer Vision & Probabilistic Occupancy Fusion service.
Mountable in the main backend service with a single include_router line:
    app.include_router(vision_router)
"""

import json
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from .camera_adapter import get_classifier_weights_path, load_classifier_model, process_frame
from .fusion import estimate_occupancy
from .schemas import OccupancyUpdate

router = APIRouter(prefix="/vision", tags=["vision"])

_MODEL: Optional[torch.nn.Module] = None
_WEIGHTS_PATH: Optional[Path] = None


def get_model() -> torch.nn.Module:
    """
    Returns the cached classifier model, loading it if necessary.
    Raises HTTPException 503 if weights cannot be found.
    """
    global _MODEL, _WEIGHTS_PATH
    if _MODEL is None:
        try:
            _WEIGHTS_PATH = get_classifier_weights_path()
            _MODEL = load_classifier_model(weights_path=_WEIGHTS_PATH, device="cpu")
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Classifier model weights not found: {str(exc)}. Please check MOBILENETV3_WEIGHTS_PATH or vision/outputs/."
            )
    return _MODEL


@router.get("/health")
def health_check() -> Dict[str, Any]:
    """
    Returns loaded weights path and model readiness status.
    """
    try:
        model = get_model()
        weights_str = str(_WEIGHTS_PATH) if _WEIGHTS_PATH else "loaded"
        return {
            "status": "ok",
            "model_ready": True,
            "weights_path": weights_str
        }
    except HTTPException as exc:
        return {
            "status": "degraded",
            "model_ready": False,
            "weights_path": None,
            "detail": exc.detail
        }


@router.post("/process-frame", response_model=OccupancyUpdate)
async def process_frame_endpoint(
    image: UploadFile = File(...),
    zone_config: str = Form(..., description="JSON string specifying facility_id, zone_id, capacity, zone_type, and slots or zone_polygon")
) -> OccupancyUpdate:
    """
    Multipart endpoint accepting raw camera frame and zone configuration.
    Runs slot classification (for managed lots) or YOLO counting (for open zones),
    and fuses with uncertainty into an OccupancyUpdate payload.
    """
    try:
        config_dict = json.loads(zone_config)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid JSON in zone_config field: {str(exc)}"
        )

    facility_id = config_dict.get("facility_id")
    zone_id = config_dict.get("zone_id")
    if not facility_id or not zone_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="zone_config must specify 'facility_id' and 'zone_id'."
        )

    # Read uploaded image bytes
    contents = await image.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image file is empty."
        )

    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to decode image from uploaded bytes."
        )

    model = get_model()

    try:
        update = process_frame(
            facility_id=facility_id,
            zone_id=zone_id,
            image=img_bgr,
            zone_config=config_dict,
            classifier_model=model,
            device="cpu"
        )
        return update
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing frame: {str(exc)}"
        )


class EstimateRequest(BaseModel):
    zone: Dict[str, Any] = Field(..., description="Dict containing facility_id, zone_id, capacity, zone_type")
    observations: Dict[str, Any] = Field(..., description="Dict containing optional camera, manual, check_ins, prior")
    now: Optional[Union[datetime, str]] = Field(default=None, description="ISO timestamp of observation time")


@router.post("/estimate", response_model=OccupancyUpdate)
def estimate_occupancy_endpoint(req: EstimateRequest) -> OccupancyUpdate:
    """
    JSON endpoint for multi-sensor probabilistic occupancy estimation.
    Fuses camera observations, manual headcounts, gate check-in/out event streams,
    and historical diurnal priors with freshness decay.
    """
    now = req.now or datetime.now(timezone.utc)
    try:
        update = estimate_occupancy(
            zone=req.zone,
            observations=req.observations,
            now=now
        )
        return update
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Fusion estimation failed: {str(exc)}"
        )

