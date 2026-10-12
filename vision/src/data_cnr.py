"""
CNRPark and CNRPark-EXT patch dataset loader.
Loads labelled patch images from:
- CNRPark patches folder (busy/ vs free/)
- Or CNR-EXT splits / annotation list files (path label)
Returns (tensor, label) where:
- label 1 = occupied (busy)
- label 0 = free
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_default_train_transforms(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_default_eval_transforms(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

class CNRParkDataset(Dataset):
    """
    PyTorch Dataset for CNRPark patches.
    Can be loaded directly from directory structure (busy/ and free/ folders)
    or from split text files (listing relative_path label).
    """
    def __init__(
        self,
        samples: Optional[List[Tuple[Path, int]]] = None,
        root_dir: Optional[Union[str, Path]] = None,
        split_file: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        crop_size: int = 224
    ):
        self.transform = transform or get_default_eval_transforms(crop_size)
        self.samples: List[Tuple[Path, int]] = []

        if samples is not None:
            self.samples = [(Path(p), l) for p, l in samples]
        elif split_file is not None and root_dir is not None:
            root_dir = Path(root_dir)
            with open(split_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        rel_path, lbl = parts[0], int(parts[1])
                        self.samples.append((root_dir / rel_path, lbl))
        elif root_dir is not None:
            root_dir = Path(root_dir)
            for busy_img in (root_dir).rglob("busy/*.jpg"):
                self.samples.append((busy_img, 1))
            for free_img in (root_dir).rglob("free/*.jpg"):
                self.samples.append((free_img, 0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        tensor = self.transform(image)
        return tensor, label
