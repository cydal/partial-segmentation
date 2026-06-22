import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.segmentation import slic as skimage_slic
from scipy.ndimage import distance_transform_edt

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


def compute_boundary_distance(label: np.ndarray) -> np.ndarray:
    """Return H×W float32 array: each pixel's distance to the nearest class boundary.

    Boundary pixels (where the label differs from at least one 4-neighbour) have
    distance 0; interior pixels have positive distances.
    """
    from scipy.ndimage import distance_transform_edt
    # A pixel is on a boundary if any neighbour has a different class.
    padded = np.pad(label, 1, mode="edge")
    is_boundary = (
        (padded[1:-1, 1:-1] != padded[:-2,  1:-1]) |
        (padded[1:-1, 1:-1] != padded[2:,   1:-1]) |
        (padded[1:-1, 1:-1] != padded[1:-1, :-2])  |
        (padded[1:-1, 1:-1] != padded[1:-1, 2:])
    )
    dist = distance_transform_edt(~is_boundary).astype(np.float32)
    return dist


def simulate_point_labels(label: np.ndarray, points_per_class: int,
                          sampling: str = "uniform") -> np.ndarray:
    """Return binary H×W float32 mask with sampled point locations set to 1.

    sampling options
    ----------------
    "uniform"  : uniform random within each class (original behaviour)
    "boundary" : weighted toward pixels near class boundaries (low dist → high weight)
    "interior" : weighted toward pixels far from class boundaries (high dist → high weight)
    """
    assert sampling in ("uniform", "boundary", "interior")
    H, W = label.shape
    point_mask = np.zeros((H, W), dtype=np.float32)
    rng = np.random.default_rng()

    if sampling != "uniform":
        dist = compute_boundary_distance(label)

    for cls in range(6):
        coords = np.argwhere(label == cls)
        if len(coords) == 0:
            continue
        n = min(points_per_class, len(coords))

        if sampling == "uniform":
            chosen = rng.choice(len(coords), size=n, replace=False)
        else:
            d = dist[coords[:, 0], coords[:, 1]].astype(np.float64)
            if sampling == "boundary":
                # Pixels at the boundary have d=0; give them the highest weight.
                # Use exp(-d / sigma) so weight decays smoothly with distance.
                sigma = max(d.std(), 1.0)
                w = np.exp(-d / sigma)
            else:  # interior
                # Invert: pixels far from any boundary get the highest weight.
                sigma = max(d.std(), 1.0)
                w = np.exp(d / sigma)
            w = w / w.sum()
            chosen = rng.choice(len(coords), size=n, replace=False, p=w)

        rows, cols = coords[chosen, 0], coords[chosen, 1]
        point_mask[rows, cols] = 1.0
    return point_mask


def propagate_points_to_superpixels(
    image: np.ndarray,
    point_mask: np.ndarray,
    n_segments: int = 200,
    compactness: float = 10.0,
) -> np.ndarray:
    """Expand each labeled point to all pixels in its SLIC superpixel.

    Returns a new binary float32 mask of the same shape as point_mask.
    Pixels belonging to any superpixel that contains at least one labeled
    point are set to 1; all others remain 0.
    """
    segments = skimage_slic(
        image.astype(np.float32) / 255.0,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
        channel_axis=2,
    )
    expanded = np.zeros_like(point_mask)
    for row, col in np.argwhere(point_mask > 0):
        expanded[segments == segments[row, col]] = 1.0
    return expanded


def build_augmentations():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])


class PotsdamPointDataset(Dataset):
    def __init__(self, data_root: str, split: str, points_per_class: int = 10,
                 augment: bool = False, use_slic: bool = False, slic_n_segments: int = 200,
                 sampling: str = "uniform"):
        assert split in SPLIT_RANGES, f"split must be one of {list(SPLIT_RANGES)}"
        self.data_root = data_root
        self.split = split
        self.points_per_class = points_per_class
        self.augment = augment and (split == "train")
        self.use_slic = use_slic
        self.slic_n_segments = slic_n_segments
        self.sampling = sampling

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

        point_mask = simulate_point_labels(label, self.points_per_class, self.sampling)

        if self.use_slic:
            point_mask = propagate_points_to_superpixels(
                image, point_mask, n_segments=self.slic_n_segments
            )

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
