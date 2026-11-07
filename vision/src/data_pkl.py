"""
PKLot dataset loader and slot-crop extractor.
Parses per-slot XML annotations (polygons and rotated rectangles)
and builds a PyTorch Dataset returning (tensor, label) where:
- label 1 = occupied
- label 0 = free
Supports:
- Train and eval torchvision transforms
- Proper rotated rectangle cropping via cv2.warpAffine / cv2.getRotationMatrix2D
- Optional disk caching of cropped slots for ultra-fast training iterations
"""

import os
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_default_train_transforms(image_size: int = 224) -> T.Compose:
    """
    Returns training transforms: brightness/contrast jitter,
    small random rotation, random horizontal flip, and ImageNet normalization.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_default_eval_transforms(image_size: int = 224) -> T.Compose:
    """
    Returns plain evaluation transforms: resize, ToTensor, and ImageNet normalization.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def parse_pklot_xml_slots(xml_path: Union[str, Path]) -> Tuple[str, List[Dict]]:
    """
    Parses a PKLot XML file into lot ID and slot dictionaries.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lot_id = root.get("id", "unknown")

    slots = []
    for space in root.findall("space"):
        space_id = space.get("id", "")
        occupied_str = space.get("occupied", "0")
        try:
            occupied = int(occupied_str)
        except ValueError:
            occupied = 0

        rot = space.find("rotatedRect")
        rot_data = None
        if rot is not None:
            c = rot.find("center")
            s = rot.find("size")
            a = rot.find("angle")
            if c is not None and s is not None and a is not None:
                rot_data = {
                    "cx": float(c.get("x")),
                    "cy": float(c.get("y")),
                    "w": float(s.get("w")),
                    "h": float(s.get("h")),
                    "angle": float(a.get("d"))
                }

        cnt = space.find("contour")
        points = []
        if cnt is not None:
            for pt in cnt.findall("point"):
                points.append((float(pt.get("x")), float(pt.get("y"))))

        slots.append({
            "slot_id": space_id,
            "occupied": occupied,
            "rotated_rect": rot_data,
            "contour": points
        })

    return lot_id, slots

def extract_slot_crop(
    image: np.ndarray,
    slot_info: Dict,
    target_size: Tuple[int, int] = (224, 224)
) -> Image.Image:
    """
    Extracts a slot crop from the full-frame image.
    Uses rotated rectangle parameters with affine transformation if available,
    or falls back to polygon bounding box.
    """
    rot = slot_info.get("rotated_rect")
    if rot is not None and rot["w"] > 0 and rot["h"] > 0:
        center = (rot["cx"], rot["cy"])
        size = (int(rot["w"]), int(rot["h"]))
        angle = rot["angle"]

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        h_img, w_img = image.shape[:2]
        rotated = cv2.warpAffine(
            image, M, (w_img, h_img),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        x = int(center[0] - size[0] / 2)
        y = int(center[1] - size[1] / 2)
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(size[0], w_img - x))
        h = max(1, min(size[1], h_img - y))

        crop = rotated[y:y+h, x:x+w]
    elif slot_info.get("contour"):
        pts = np.array(slot_info["contour"], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        h_img, w_img = image.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))
        crop = image[y:y+h, x:x+w]
    else:
        crop = image[:32, :32]

    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        crop = cv2.resize(image[:64, :64], target_size)

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)


def preprocess_slot_crop(
    image: np.ndarray,
    slot_info: Dict,
    target_size: Tuple[int, int] = (224, 224),
    transform: Optional[Callable] = None
) -> torch.Tensor:
    """
    Unified preprocessing function used identically across training, evaluation, and inference.
    Extracts the slot crop (using rotated-rect affine warp or polygon bounding box),
    converts to RGB, and applies standard evaluation transforms (resize, ToTensor, ImageNet normalize).
    """
    pil_crop = extract_slot_crop(image, slot_info, target_size=target_size)
    if transform is None:
        transform = get_default_eval_transforms(target_size[0])
    return transform(pil_crop)


class PKLotDataset(Dataset):
    """
    PyTorch Dataset for PKLot parking slot crops.
    """
    def __init__(
        self,
        samples: List[Tuple[Path, Path]], # List of (image_path, xml_path)
        transform: Optional[Callable] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        crop_size: int = 224
    ):
        self.samples = samples
        self.transform = transform or get_default_eval_transforms(crop_size)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.crop_size = (crop_size, crop_size)

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Index all slot metadata
        self.slots_index: List[Dict] = []
        self._build_index()

    def _build_index(self):
        for img_path, xml_path in self.samples:
            img_path = Path(img_path)
            xml_path = Path(xml_path)
            lot_id, slots = parse_pklot_xml_slots(xml_path)

            for slot in slots:
                entry = {
                    "image_path": str(img_path),
                    "xml_path": str(xml_path),
                    "slot_id": slot["slot_id"],
                    "occupied": slot["occupied"],
                    "rotated_rect": slot["rotated_rect"],
                    "contour": slot["contour"],
                }
                self.slots_index.append(entry)

    def __len__(self) -> int:
        return len(self.slots_index)

    def _get_cache_path(self, item: Dict) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        key = f"{item['image_path']}_{item['slot_id']}_{item['occupied']}"
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.jpg"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.slots_index[idx]
        label = int(item["occupied"])

        cache_path = self._get_cache_path(item)
        if cache_path is not None and cache_path.exists():
            crop_img = Image.open(cache_path).convert("RGB")
        else:
            full_img = cv2.imread(item["image_path"])
            if full_img is None:
                crop_img = Image.new("RGB", self.crop_size, (128, 128, 128))
            else:
                crop_img = extract_slot_crop(full_img, item, self.crop_size)
                if cache_path is not None:
                    crop_img.save(cache_path)

        tensor = self.transform(crop_img)
        return tensor, label
