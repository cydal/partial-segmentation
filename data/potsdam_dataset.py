import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# RGB color → class index mapping
COLOR_TO_CLASS = {
    (255, 255, 255): 0,  # Impervious surfaces
    (0,   0,   255): 1,  # Building
    (0,   255, 255): 2,  # Low vegetation
    (0,   255,   0): 3,  # Tree
    (255, 255,   0): 4,  # Car
    (255,   0,   0): 5,  # Clutter/background
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

SPLIT_RANGES = {
    "train": (0,    1680),
    "val":   (1680, 2040),
    "test":  (2040, 2400),
}


def rgb_to_label(label_rgb: np.ndarray) -> np.ndarray:
    """Convert H×W×3 uint8 RGB label image to H×W int64 class-index array."""
    H, W, _ = label_rgb.shape
    label = np.full((H, W), 5, dtype=np.int64)  # default: clutter
    for rgb, cls in COLOR_TO_CLASS.items():
        match = (
            (label_rgb[:, :, 0] == rgb[0]) &
            (label_rgb[:, :, 1] == rgb[1]) &
            (label_rgb[:, :, 2] == rgb[2])
        )
        label[match] = cls
    return label


def simulate_point_labels(label: np.ndarray, points_per_class: int) -> np.ndarray:
    """Return binary H×W float32 mask with sampled point locations set to 1."""
    H, W = label.shape
    point_mask = np.zeros((H, W), dtype=np.float32)
    rng = np.random.default_rng()  # unseeded for true randomness each call
    for cls in range(6):
        coords = np.argwhere(label == cls)
        if len(coords) == 0:
            continue
        n = min(points_per_class, len(coords))
        chosen = rng.choice(len(coords), size=n, replace=False)
        rows, cols = coords[chosen, 0], coords[chosen, 1]
        point_mask[rows, cols] = 1.0
    return point_mask


def build_augmentations():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])


class PotsdamPointDataset(Dataset):
    def __init__(self, data_root: str, split: str, points_per_class: int = 10, augment: bool = False):
        assert split in SPLIT_RANGES, f"split must be one of {list(SPLIT_RANGES)}"
        self.data_root = data_root
        self.split = split
        self.points_per_class = points_per_class
        self.augment = augment and (split == "train")

        start, end = SPLIT_RANGES[split]
        self.indices = list(range(start, end))

        self.aug = build_augmentations() if self.augment else None

        # Pad to next multiple of 16 (300 → 304) required by DeepLabV3+
        self.pad = A.PadIfNeeded(min_height=304, min_width=304, border_mode=0)
        self.normalize = A.Compose([
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_idx = self.indices[idx]
        img_path   = os.path.join(self.data_root, "Images", f"Image_{img_idx}.tif")
        label_path = os.path.join(self.data_root, "Labels", f"Label_{img_idx}.tif")

        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        label_rgb = np.array(Image.open(label_path).convert("RGB"), dtype=np.uint8)
        label = rgb_to_label(label_rgb)

        if self.aug is not None:
            augmented = self.aug(image=image, mask=label)
            image = augmented["image"]
            label = augmented["mask"]

        point_mask = simulate_point_labels(label, self.points_per_class)

        # Pad image, label, and point_mask together so shapes stay aligned
        padded = self.pad(image=image, masks=[label.astype(np.int32),
                                              (point_mask * 255).astype(np.uint8)])
        image      = padded["image"]
        label      = padded["masks"][0].astype(np.int64)
        point_mask = (padded["masks"][1] > 0).astype(np.float32)

        normalized = self.normalize(image=image)
        image_tensor = normalized["image"].float()          # [3, H, W]
        label_tensor = torch.from_numpy(label).long()       # [H, W]
        point_tensor = torch.from_numpy(point_mask).float() # [H, W]

        return {
            "image":      image_tensor,
            "label":      label_tensor,
            "point_mask": point_tensor,
        }
