"""
analyze_slic_purity.py

Pre-training diagnostic: measures how well SLIC superpixels preserve label
homogeneity around simulated point annotations.

For a sample of training images and a sweep of n_segments values, this script:
  1. Simulates sparse point labels (same density as training)
  2. Runs SLIC to over-segment each image into superpixels
  3. For each labeled point, collects all pixels in the same superpixel
  4. Uses ground truth to compute:
       purity   = fraction of superpixel pixels that match the point's class
       coverage = number of pixels in the superpixel (vs 1 for the bare point)

High purity justifies propagating the point's label to its entire superpixel
during training.

Outputs
-------
  results/slic_analysis/purity_data.csv          per-point records
  results/figures/figS1_slic_purity_hist.png     purity distribution
  results/figures/figS2_slic_purity_by_class.png per-class boxplots
  results/figures/figS3_slic_nsegments_sweep.png purity & coverage vs n_segments
  results/figures/figS4_slic_qualitative.png     example purity maps
"""

import os
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from skimage.segmentation import slic, mark_boundaries

from data.potsdam_dataset import rgb_to_label, SPLIT_RANGES

# ── settings ──────────────────────────────────────────────────────────────────
DATA_ROOT = "/home/ubuntu/weak-supervision/patches"
OUT_DIR   = "results/slic_analysis"
FIG_DIR   = "results/figures"

CLASS_NAMES  = ["Impervious", "Building", "Low veg.", "Tree", "Car", "Clutter"]
CLASS_COLORS = np.array([
    [255, 255, 255],
    [  0,   0, 255],
    [  0, 255, 255],
    [  0, 255,   0],
    [255, 255,   0],
    [255,   0,   0],
], dtype=np.uint8)

N_SEGMENTS_DEFAULT = 200
N_SEGMENTS_SWEEP   = [50, 100, 200, 400, 800]
COMPACTNESS        = 10.0
POINTS_PER_CLASS   = 10
N_IMAGES           = 50
BASE_SEED          = 42


# ── helpers ───────────────────────────────────────────────────────────────────

def load_image_and_label(idx: int):
    img = np.array(
        Image.open(f"{DATA_ROOT}/Images/Image_{idx}.tif").convert("RGB"),
        dtype=np.uint8,
    )
    lbl_rgb = np.array(
        Image.open(f"{DATA_ROOT}/Labels/Label_{idx}.tif").convert("RGB"),
        dtype=np.uint8,
    )
    return img, rgb_to_label(lbl_rgb)


def sample_points_seeded(label: np.ndarray, points_per_class: int, seed: int):
    """Reproducible point sampling (unlike the dataset's unseeded version)."""
    H, W = label.shape
    point_mask = np.zeros((H, W), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for cls in range(6):
        coords = np.argwhere(label == cls)
        if len(coords) == 0:
            continue
        n = min(points_per_class, len(coords))
        chosen = rng.choice(len(coords), size=n, replace=False)
        point_mask[coords[chosen, 0], coords[chosen, 1]] = 1.0
    return point_mask


def analyze_image(img: np.ndarray, label: np.ndarray,
                  n_segments: int, point_seed: int):
    """
    Returns
    -------
    records    : list of dicts with keys class, purity, coverage, n_segments
    segments   : H×W int32 superpixel ID map
    point_mask : H×W float32 binary annotation mask
    """
    point_mask = sample_points_seeded(label, POINTS_PER_CLASS, point_seed)

    img_float = img.astype(np.float32) / 255.0
    segments  = slic(img_float, n_segments=n_segments, compactness=COMPACTNESS,
                     start_label=0, channel_axis=2)

    records = []
    for row, col in np.argwhere(point_mask > 0):
        seg_id      = segments[row, col]
        sp_mask     = segments == seg_id
        point_class = int(label[row, col])
        records.append({
            "class":      point_class,
            "purity":     float((label[sp_mask] == point_class).mean()),
            "coverage":   int(sp_mask.sum()),
            "n_segments": n_segments,
        })
    return records, segments, point_mask


# ── analysis loop ─────────────────────────────────────────────────────────────

def run_analysis(n_images: int = N_IMAGES, seed: int = BASE_SEED):
    rng = np.random.default_rng(seed)
    start, end = SPLIT_RANGES["train"]
    sample_indices = rng.choice(range(start, end), size=n_images,
                                replace=False).tolist()

    all_records = []
    print(f"Analyzing {n_images} training images across "
          f"n_segments ∈ {N_SEGMENTS_SWEEP} ...")
    for i, idx in enumerate(sample_indices):
        if i % 10 == 0:
            print(f"  {i}/{n_images}  Image_{idx}")
        img, label = load_image_and_label(idx)
        point_seed = seed + idx          # same points for all n_segments on this image
        for n_seg in N_SEGMENTS_SWEEP:
            records, _, _ = analyze_image(img, label, n_seg, point_seed)
            all_records.extend(records)

    return all_records, sample_indices


def save_csv(records, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_segments", "class", "purity", "coverage"])
        w.writeheader()
        w.writerows(records)
    print(f"Saved {path}")


def print_summary(records, n_seg: int = N_SEGMENTS_DEFAULT):
    subset    = [r for r in records if r["n_segments"] == n_seg]
    purities  = np.array([r["purity"]   for r in subset])
    coverages = np.array([r["coverage"] for r in subset])

    print(f"\n=== Purity summary  (n_segments={n_seg}) ===")
    print(f"  N points analysed : {len(purities)}")
    print(f"  Mean purity       : {purities.mean():.3f}")
    print(f"  Median purity     : {np.median(purities):.3f}")
    print(f"  p10 / p25 / p75 / p90 : "
          f"{np.percentile(purities, 10):.3f} / "
          f"{np.percentile(purities, 25):.3f} / "
          f"{np.percentile(purities, 75):.3f} / "
          f"{np.percentile(purities, 90):.3f}")
    print(f"  Fraction ≥ 0.80   : {(purities >= 0.80).mean():.1%}")
    print(f"  Fraction ≥ 0.90   : {(purities >= 0.90).mean():.1%}")
    print(f"  Mean coverage     : {coverages.mean():.1f} px  (1 px = bare point)")
    print(f"  Median coverage   : {np.median(coverages):.1f} px")
    print()
    print("  Per-class mean purity:")
    for c, name in enumerate(CLASS_NAMES):
        vals = [r["purity"] for r in subset if r["class"] == c]
        if vals:
            print(f"    {name:<15}: {np.mean(vals):.3f}  (n={len(vals)})")


# ── figures ───────────────────────────────────────────────────────────────────

def fig_purity_hist(records, n_seg: int = N_SEGMENTS_DEFAULT):
    data = np.array([r["purity"] for r in records if r["n_segments"] == n_seg])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=40, color="#2a7ae0", edgecolor="white", linewidth=0.4)

    styles = [(50, "dotted"), (75, "dashed"), (90, "solid")]
    for pct, ls in styles:
        v = np.percentile(data, pct)
        ax.axvline(v, color="#e05c2a", linestyle=ls, linewidth=1.6,
                   label=f"p{pct} = {v:.2f}")

    frac_80 = (data >= 0.8).mean()
    ax.text(0.02, 0.96, f"{frac_80:.1%} of superpixels\nhave purity ≥ 0.80",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel("Superpixel purity  (fraction of pixels matching point's class)")
    ax.set_ylabel("Count")
    ax.set_title(f"SLIC superpixel purity distribution  "
                 f"(n_segments={n_seg},  N={len(data)} points)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "figS1_slic_purity_hist.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def fig_purity_by_class(records, n_seg: int = N_SEGMENTS_DEFAULT):
    data_by_class = [
        [r["purity"] for r in records if r["n_segments"] == n_seg and r["class"] == c]
        for c in range(6)
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data_by_class, patch_artist=True, notch=False,
                    medianprops={"color": "black", "linewidth": 2},
                    flierprops={"marker": ".", "markersize": 3, "alpha": 0.5})

    for patch, color in zip(bp["boxes"], CLASS_COLORS):
        patch.set_facecolor(np.array(color) / 255.0)
        patch.set_alpha(0.85)

    ax.set_xticks(range(1, 7))
    ax.set_xticklabels(
        [f"{n}\n(n={len(d)})" for n, d in zip(CLASS_NAMES, data_by_class)],
        fontsize=9,
    )
    ax.set_ylabel("Superpixel purity")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=1.2, alpha=0.7,
               label="purity = 0.80")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(
        f"SLIC superpixel purity by class  (n_segments={n_seg})", fontsize=11
    )
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "figS2_slic_purity_by_class.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def fig_nsegments_sweep(records):
    n_segs = sorted(set(r["n_segments"] for r in records))

    mean_purity, frac_80, mean_coverage = [], [], []
    for n_seg in n_segs:
        subset    = [r for r in records if r["n_segments"] == n_seg]
        purities  = np.array([r["purity"]   for r in subset])
        coverages = np.array([r["coverage"] for r in subset])
        mean_purity.append(purities.mean())
        frac_80.append((purities >= 0.8).mean())
        mean_coverage.append(coverages.mean())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(n_segs, mean_purity, "o-", color="#2a7ae0", linewidth=2,
             markersize=7, label="Mean purity")
    ax1.plot(n_segs, frac_80, "s--", color="#e05c2a", linewidth=1.8,
             markersize=6, label="Frac. purity ≥ 0.80")
    for x, y in zip(n_segs, mean_purity):
        ax1.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)
    ax1.set_xlabel("n_segments")
    ax1.set_ylabel("Purity")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Purity vs superpixel granularity")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.plot(n_segs, mean_coverage, "o-", color="#4caf50", linewidth=2, markersize=7)
    for x, y in zip(n_segs, mean_coverage):
        ax2.annotate(f"{y:.0f} px", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)
    ax2.axhline(1, color="gray", linestyle=":", linewidth=1.2,
                label="1 px (no propagation)")
    ax2.set_xlabel("n_segments")
    ax2.set_ylabel("Mean pixels per superpixel")
    ax2.set_title("Coverage gain vs superpixel granularity")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "SLIC granularity tradeoff: more segments → higher purity, "
        "lower coverage gain",
        fontsize=12,
    )
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "figS3_slic_nsegments_sweep.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def fig_qualitative(sample_indices, seed: int = BASE_SEED,
                    n_seg: int = N_SEGMENTS_DEFAULT):
    """3 example images: image | ground truth | SLIC boundaries | purity map."""
    rng = np.random.default_rng(seed)
    chosen = rng.choice(sample_indices, size=3, replace=False).tolist()

    col_titles = [
        "Image",
        "Ground truth",
        f"SLIC boundaries\n(n_segments={n_seg})",
        "Superpixel purity\n(dots = labeled points)",
    ]

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10)

    for row, idx in enumerate(chosen):
        img, label = load_image_and_label(idx)
        point_seed = seed + idx
        records, segments, point_mask = analyze_image(img, label, n_seg, point_seed)

        # col 0: raw image
        axes[row, 0].imshow(img)

        # col 1: ground truth
        axes[row, 1].imshow(CLASS_COLORS[label.clip(0, 5)])

        # col 2: SLIC boundaries on image
        boundaries = mark_boundaries(
            img.astype(np.float32) / 255.0, segments,
            color=(1, 1, 0), mode="thick",
        )
        axes[row, 2].imshow(boundaries)

        # col 3: purity heatmap per superpixel
        purity_map = np.full(label.shape, np.nan, dtype=np.float32)
        for rec_row, rec_col in np.argwhere(point_mask > 0):
            seg_id   = segments[rec_row, rec_col]
            sp_mask  = segments == seg_id
            pt_class = int(label[rec_row, rec_col])
            purity   = float((label[sp_mask] == pt_class).mean())
            purity_map[sp_mask] = purity

        im = axes[row, 3].imshow(purity_map, cmap="RdYlGn", vmin=0, vmax=1)
        ys, xs = np.where(point_mask > 0)
        pt_colors = [CLASS_COLORS[label[y, x]] / 255.0 for y, x in zip(ys, xs)]
        axes[row, 3].scatter(xs, ys, c=pt_colors, s=18, linewidths=0.5,
                              edgecolors="black")
        plt.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04,
                     label="Purity")

        for ax in axes[row]:
            ax.axis("off")

    patches = [mpatches.Patch(color=np.array(c) / 255.0, label=n)
               for c, n in zip(CLASS_COLORS, CLASS_NAMES)]
    fig.legend(handles=patches, loc="lower center", ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("SLIC purity diagnostic — example images", fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "figS4_slic_qualitative.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SLIC superpixel purity diagnostic for label propagation"
    )
    parser.add_argument("--n_images", type=int, default=N_IMAGES,
                        help="Number of training images to sample (default: 50)")
    parser.add_argument("--seed",     type=int, default=BASE_SEED)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    records, sample_indices = run_analysis(n_images=args.n_images, seed=args.seed)

    save_csv(records, os.path.join(OUT_DIR, "purity_data.csv"))
    print_summary(records)

    print("\nGenerating figures...")
    fig_purity_hist(records)
    fig_purity_by_class(records)
    fig_nsegments_sweep(records)
    fig_qualitative(sample_indices, seed=args.seed)

    print(f"\nDone. Figures in {FIG_DIR}/")


if __name__ == "__main__":
    main()
