"""
Slot occupancy classification API for backend service integration.
Provides classify_slots(image, slot_boxes) returning:
[
    {"slot_id": ..., "occupied": 0 or 1, "confidence": float}
]
"""

from typing import Dict, List, Union
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from .data_pkl import preprocess_slot_crop

if hasattr(torch.backends, "mkldnn"):
    torch.backends.mkldnn.enabled = False

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_EVAL_TRANSFORMS = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def classify_slots(
    image: Union[np.ndarray, str, Image.Image],
    slot_boxes: List[Dict],
    model: torch.nn.Module,
    device: Union[str, torch.device] = "cpu"
) -> List[Dict]:
    """
    Classifies a list of slot bounding boxes or polygons within a parking image.

    Args:
        image: Full parking image (numpy BGR/RGB array, file path, or PIL Image).
        slot_boxes: List of dictionaries describing slots. Each dictionary should have:
            - 'slot_id': Identifier string or int.
            - Either 'bbox': [x, y, w, h] or 'points': [[x1, y1], [x2, y2], ...] or
              'rotated_rect': {'cx': float, 'cy': float, 'w': float, 'h': float, 'angle': float}
        model: Trained MobileNetV3-small PyTorch model.
        device: 'cpu' or 'cuda'.

    Returns:
        List of dicts:
        [
            {"slot_id": ..., "occupied": 1 or 0, "confidence": float}, ...
        ]
    """
    if isinstance(image, str):
        img_bgr = cv2.imread(image)
        if img_bgr is None:
            raise ValueError(f"Could not load image from path: {image}")
        img_np = img_bgr
    elif isinstance(image, Image.Image):
        img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        img_np = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    h_img, w_img = img_np.shape[:2]
    model = model.to(device)
    model.eval()

    results = []
    crops_tensors = []
    slot_ids = []

    for slot in slot_boxes:
        s_id = slot.get("slot_id", "unknown")
        tensor = preprocess_slot_crop(img_np, slot, target_size=(224, 224), transform=_EVAL_TRANSFORMS)
        crops_tensors.append(tensor)
        slot_ids.append(s_id)

    if not crops_tensors:
        return []

    batch_size = 32
    all_preds = []
    all_confs = []

    with torch.no_grad():
        for b_start in range(0, len(crops_tensors), batch_size):
            b_chunk = torch.stack(crops_tensors[b_start:b_start + batch_size]).to(device)
            logits = model(b_chunk)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            confs = probs.cpu().numpy()
            all_preds.extend(preds)
            all_confs.extend(confs)

    for i in range(len(slot_ids)):
        pred_label = int(all_preds[i])
        confidence_val = float(all_confs[i][pred_label])
        results.append({
            "slot_id": slot_ids[i],
            "occupied": pred_label,
            "confidence": round(confidence_val, 4)
        })

    return results
