"""Pre-flight checks for the fixed-points sampling changes (Part 1, step 7).

1. Loading the same training image twice returns the same point mask,
   up to the augmentation transform.
2. The nested property holds for uniform sampling (1 ⊂ 5 ⊂ 10 ⊂ 20 ⊂ 50).
3. Mean number of labelled pixels per image at 1, 10 and 50 points per class.
"""
import numpy as np

from data.potsdam_dataset import (
    PotsdamPointDataset, simulate_point_labels, rgb_to_label, _point_rng,
)
from PIL import Image
import os

DATA_ROOT = "/home/ubuntu/weak-supervision/patches"


def check_determinism():
    print("=== Check 1: same image → same points (up to augmentation) ===")
    # No augmentation: the returned point mask must be bit-identical.
    ds = PotsdamPointDataset(DATA_ROOT, "train", points_per_class=10,
                             augment=False, fixed_points=True, point_seed=0)
    m1 = ds[0]["point_mask"].numpy()
    m2 = ds[0]["point_mask"].numpy()
    identical = np.array_equal(m1, m2)
    print(f"  no-aug, two loads identical: {identical}  (points={int(m1.sum())})")

    # With augmentation: point count is preserved (flips/rot90 move but don't drop).
    ds_aug = PotsdamPointDataset(DATA_ROOT, "train", points_per_class=10,
                                 augment=True, fixed_points=True, point_seed=0)
    counts = [int(ds_aug[0]["point_mask"].numpy().sum()) for _ in range(5)]
    print(f"  aug, point counts over 5 loads: {counts}  (constant: {len(set(counts)) == 1})")

    # Underlying (pre-aug) points are seed-stable across dataset instances.
    ds_b = PotsdamPointDataset(DATA_ROOT, "train", points_per_class=10,
                               augment=False, fixed_points=True, point_seed=0)
    same_across_instances = np.array_equal(m1, ds_b[0]["point_mask"].numpy())
    print(f"  identical across fresh dataset instances: {same_across_instances}")
    return identical and len(set(counts)) == 1 and same_across_instances


def check_nested():
    print("\n=== Check 2: nested densities for uniform sampling ===")
    ok = True
    for img_idx in [0, 17, 123, 500, 1000]:
        label_path = os.path.join(DATA_ROOT, "Labels", f"Label_{img_idx}.tif")
        label = rgb_to_label(np.array(Image.open(label_path).convert("RGB"), dtype=np.uint8))
        masks = {}
        for n in [1, 5, 10, 20, 50]:
            # Fresh generator with the SAME seed each time → nested by construction.
            rng = _point_rng(0, img_idx)
            masks[n] = simulate_point_labels(label, n, "uniform", rng=rng)
        nested = all(
            bool(((masks[a] > 0) & ~(masks[b] > 0)).sum() == 0)  # a ⊆ b
            for a, b in [(1, 5), (5, 10), (10, 20), (20, 50)]
        )
        counts = {n: int(masks[n].sum()) for n in masks}
        print(f"  image {img_idx}: nested={nested}  counts={counts}")
        ok = ok and nested
    return ok


def check_mean_pixels():
    print("\n=== Check 3: mean labelled pixels per image (train split) ===")
    # Sample a subset for speed; sampling is deterministic so this is stable.
    n_imgs = 200
    for n in [1, 10, 50]:
        totals = []
        for img_idx in range(n_imgs):
            label_path = os.path.join(DATA_ROOT, "Labels", f"Label_{img_idx}.tif")
            label = rgb_to_label(np.array(Image.open(label_path).convert("RGB"), dtype=np.uint8))
            rng = _point_rng(0, img_idx)
            m = simulate_point_labels(label, n, "uniform", rng=rng)
            totals.append(int(m.sum()))
        print(f"  {n:>2} points/class: mean={np.mean(totals):.2f}  "
              f"min={min(totals)}  max={max(totals)}  (over {n_imgs} images)")


if __name__ == "__main__":
    c1 = check_determinism()
    c2 = check_nested()
    check_mean_pixels()
    print(f"\nDeterminism check passed: {c1}")
    print(f"Nested check passed:      {c2}")
