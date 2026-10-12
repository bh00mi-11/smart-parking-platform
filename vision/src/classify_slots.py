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
        rot = slot.get("rotated_rect")
        bbox = slot.get("bbox")
        pts = slot.get("points") or slot.get("contour")

        if rot is not None and rot.get("w", 0) > 0 and rot.get("h", 0) > 0:
            center = (rot["cx"], rot["cy"])
            size = (int(rot["w"]), int(rot["h"]))
            angle = rot.get("angle", 0.0)

            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img_np, M, (w_img, h_img), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            x = int(center[0] - size[0] / 2)
            y = int(center[1] - size[1] / 2)
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = max(1, min(size[0], w_img - x))
            h = max(1, min(size[1], h_img - y))
            crop = rotated[y:y+h, x:x+w]
        elif bbox is not None:
            x, y, w, h = [int(v) for v in bbox]
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = max(1, min(w, w_img - x))
            h = max(1, min(h, h_img - y))
            crop = img_np[y:y+h, x:x+w]
        elif pts is not None and len(pts) >= 3:
            pts_np = np.array(pts, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts_np)
            crop = img_np[y:y+h, x:x+w]
        else:
            crop = img_np[:32, :32]

        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            crop = img_np[:64, :64]

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(crop_rgb)
        tensor = _EVAL_TRANSFORMS(pil_crop)

        crops_tensors.append(tensor)
        slot_ids.append(s_id)

    if not crops_tensors:
        return []

    batch_tensor = torch.stack(crops_tensors).to(device)

    with torch.no_grad():
        logits = model(batch_tensor)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        confs = probs.cpu().numpy()

    for i in range(len(slot_ids)):
        pred_label = int(preds[i])
        confidence_val = float(confs[i][pred_label])
        results.append({
            "slot_id": slot_ids[i],
            "occupied": pred_label,
            "confidence": round(confidence_val, 4)
        })

    return results
