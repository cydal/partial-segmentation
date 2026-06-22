"""
analyze_boundary_sampling.py

Pre-training diagnostic: characterises what boundary-biased, interior-biased,
and uniform point sampling strategies actually produce — before any model is
trained.  For a sample of training images the script measures:

  dist_to_boundary  : distance (px) from each sampled point to the nearest
                      class boundary — confirms the sampling is doing what we
                      expect
  local_purity      : fraction of an 11×11 patch centred on each point that
                      shares the point's class — a proxy for label reliability
                      at the click location

These two metrics characterise the tradeoff: boundary clicks are close to
edges (low dist) which gives the model discriminative context, but the patch
around them is more likely to be mixed (lower purity); interior clicks are far
from edges (high dist) and very pure but carry less discriminative information.

Outputs
-------
  results/boundary_analysis/sampling_data.csv    per-point records
  results/figures/figS5_sampling_dist.png        distance distributions (violin)
  results/figures/figS6_sampling_purity.png      local purity distributions
  results/figures/figS7_sampling_scatter.png     purity vs distance scatter
  results/figures/figS8_sampling_qualitative.png example images annotated
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

from data.potsdam_dataset import (
    rgb_to_label, simulate_point_labels, compute_boundary_distance, SPLIT_RANGES
)

# ── settings ──────────────────────────────────────────────────────────────────
DATA_ROOT   = "/home/ubuntu/weak-supervision/patches"
OUT_DIR     = "results/boundary_analysis"
FIG_DIR     = "results/figures"

CLASS_NAMES  = ["Impervious", "Building", "Low veg.", "Tree", "Car", "Clutter"]
CLASS_COLORS = np.array([
    [255, 255, 255],
    [  0,   0, 255],
    [  0, 255, 255],
    [  0, 255,   0],
    [255, 255,   0],
    [255,   0,   0],
], dtype=np.uint8)

STRATEGIES        = ["uniform", "boundary", "interior"]
STRATEGY_COLORS   = {"uniform": "#2a7ae0", "boundary": "#e05c2a", "interior": "#4caf50"}
STRATEGY_LABELS   = {"uniform": "Uniform", "boundary": "Boundary-biased",
                     "interior": "Interior-biased"}
POINTS_PER_CLASS  = 10
PATCH_RADIUS      = 5   # local purity patch = (2r+1)² = 11×11
N_IMAGES          = 50
BASE_SEED         = 42
N_REPEATS         = 5   # sample each image N times per strategy to average out variance


# ── helpers ───────────────────────────────────────────────────────────────────

def load_image_and_label(idx: int):
    img = np.array(
        Image.open(f"{DATA_ROOT}/Images/Image_{idx}.tif").convert("RGB"), dtype=np.uint8)
    lbl_rgb = np.array(
        Image.open(f"{DATA_ROOT}/Labels/Label_{idx}.tif").convert("RGB"), dtype=np.uint8)
    return img, rgb_to_label(lbl_rgb)


def local_purity(label: np.ndarray, row: int, col: int, radius: int = PATCH_RADIUS) -> float:
    """Fraction of (2r+1)² patch that matches label[row, col]."""
    H, W = label.shape
    r0, r1 = max(0, row - radius), min(H, row + radius + 1)
    c0, c1 = max(0, col - radius), min(W, col + radius + 1)
    patch = label[r0:r1, c0:c1]
    return float((patch == label[row, col]).mean())


def analyze_image(label: np.ndarray, dist: np.ndarray,
                  strategy: str, seed: int):
    mask = simulate_point_labels(label, POINTS_PER_CLASS, sampling=strategy)
    records = []
    for row, col in np.argwhere(mask > 0):
        records.append({
            "strategy":       strategy,
            "class":          int(label[row, col]),
            "dist_to_boundary": float(dist[row, col]),
            "local_purity":   local_purity(label, row, col),
        })
    return records


# ── main analysis loop ────────────────────────────────────────────────────────

def run_analysis(n_images: int = N_IMAGES, seed: int = BASE_SEED):
    rng = np.random.default_rng(seed)
    start, end = SPLIT_RANGES["train"]
    sample_indices = rng.choice(range(start, end), size=n_images, replace=False).tolist()

    all_records = []
    print(f"Analysing {n_images} images × {N_REPEATS} repeats × 3 strategies ...")
    for i, idx in enumerate(sample_indices):
        if i % 10 == 0:
            print(f"  {i}/{n_images}  Image_{idx}")
        _, label = load_image_and_label(idx)
        dist = compute_boundary_distance(label)
        for rep in range(N_REPEATS):
            rep_seed = seed + idx * 100 + rep
            for strategy in STRATEGIES:
                records = analyze_image(label, dist, strategy, rep_seed)
                all_records.extend(records)

    return all_records, sample_indices


def save_csv(records, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f,
            fieldnames=["strategy", "class", "dist_to_boundary", "local_purity"])
        w.writeheader()
        w.writerows(records)
    print(f"Saved {path}")


def print_summary(records):
    print("\n=== Summary by strategy ===")
    header = f"{'Strategy':<20} {'N':>6}  {'mean dist':>10}  {'med dist':>9}  "
    header += f"{'mean purity':>12}  {'frac pur≥0.90':>14}"
    print(header)
    print("-" * len(header))
    for s in STRATEGIES:
        sub = [r for r in records if r["strategy"] == s]
        dists   = np.array([r["dist_to_boundary"] for r in sub])
        purities = np.array([r["local_purity"]    for r in sub])
        print(f"  {STRATEGY_LABELS[s]:<18} {len(sub):>6}  "
              f"{dists.mean():>10.2f}  {np.median(dists):>9.2f}  "
              f"{purities.mean():>12.4f}  {(purities >= 0.90).mean():>14.3f}")

    print("\n=== Per-class local purity at n=200 (all strategies) ===")
    for s in STRATEGIES:
        print(f"\n  {STRATEGY_LABELS[s]}:")
        for c, name in enumerate(CLASS_NAMES):
            vals = [r["local_purity"] for r in records
                    if r["strategy"] == s and r["class"] == c]
            if vals:
                print(f"    {name:<15}: mean={np.mean(vals):.4f}  n={len(vals)}")


# ── figures ───────────────────────────────────────────────────────────────────

def fig_distance_distributions(records):
    """Violin plot of distance-to-boundary per strategy."""
    fig, ax = plt.subplots(figsize=(8, 5))

    data = [[r["dist_to_boundary"] for r in records if r["strategy"] == s]
            for s in STRATEGIES]
    parts = ax.violinplot(data, positions=[1, 2, 3], showmedians=True,
                          showextrema=True)
    for pc, s in zip(parts["bodies"], STRATEGIES):
        pc.set_facecolor(STRATEGY_COLORS[s])
        pc.set_alpha(0.75)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    # Overlay median text
    for pos, d in zip([1, 2, 3], data):
        med = np.median(d)
        ax.text(pos, med + 0.5, f"{med:.1f}", ha="center", fontsize=8,
                fontweight="bold")

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGIES])
    ax.set_ylabel("Distance to nearest class boundary (px)")
    ax.set_title("Distribution of sampled point distances to class boundaries\n"
                 "(confirms sampling strategy is functioning as intended)")
    ax.grid(axis="y", alpha=0.3)

    patches = [mpatches.Patch(color=STRATEGY_COLORS[s], label=STRATEGY_LABELS[s])
               for s in STRATEGIES]
    ax.legend(handles=patches, fontsize=9)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "figS5_sampling_dist.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def fig_purity_distributions(records):
    """Violin plot of local purity per strategy, with per-class breakdown inset."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overall purity violin
    ax = axes[0]
    data = [[r["local_purity"] for r in records if r["strategy"] == s]
            for s in STRATEGIES]
    parts = ax.violinplot(data, positions=[1, 2, 3], showmedians=True)
    for pc, s in zip(parts["bodies"], STRATEGIES):
        pc.set_facecolor(STRATEGY_COLORS[s])
        pc.set_alpha(0.75)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)
    for pos, d in zip([1, 2, 3], data):
        med = np.median(d)
        ax.text(pos, med + 0.01, f"{med:.3f}", ha="center", fontsize=8,
                fontweight="bold")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGIES])
    ax.set_ylabel("Local purity  (11×11 patch)")
    ax.set_ylim(0, 1.08)
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=1.2, alpha=0.7,
               label="purity = 0.90")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Local purity by sampling strategy")

    # Right: per-class mean purity grouped bars
    ax = axes[1]
    x = np.arange(6)
    width = 0.25
    for i, s in enumerate(STRATEGIES):
        means = []
        for c in range(6):
            vals = [r["local_purity"] for r in records
                    if r["strategy"] == s and r["class"] == c]
            means.append(np.mean(vals) if vals else 0.0)
        ax.bar(x + (i - 1) * width, means, width,
               label=STRATEGY_LABELS[s], color=STRATEGY_COLORS[s], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Mean local purity")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Per-class local purity by strategy")

    fig.suptitle("Local label purity at sampled points  (11×11 neighbourhood)",
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "figS6_sampling_purity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def fig_scatter(records):
    """Scatter: local purity vs distance-to-boundary, one panel per strategy."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, s in zip(axes, STRATEGIES):
        sub  = [r for r in records if r["strategy"] == s]
        dist = np.array([r["dist_to_boundary"] for r in sub])
        pur  = np.array([r["local_purity"]     for r in sub])
        # Hex-bin for density since N is large
        hb = ax.hexbin(dist, pur, gridsize=35, cmap="Blues", mincnt=1)
        plt.colorbar(hb, ax=ax, label="count")
        # Mean purity in distance bins
        bins = np.arange(0, dist.max() + 5, 5)
        bin_idx = np.digitize(dist, bins) - 1
        bin_means, bin_xs = [], []
        for b in range(len(bins) - 1):
            in_bin = pur[bin_idx == b]
            if len(in_bin) > 5:
                bin_means.append(in_bin.mean())
                bin_xs.append((bins[b] + bins[b+1]) / 2)
        ax.plot(bin_xs, bin_means, color=STRATEGY_COLORS[s],
                linewidth=2.0, label="bin mean")
        ax.set_xlabel("Distance to boundary (px)")
        ax.set_title(STRATEGY_LABELS[s])
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Local purity  (11×11 patch)")
    fig.suptitle("Local purity vs distance to class boundary  —  all three strategies",
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "figS7_sampling_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def fig_qualitative(sample_indices, seed: int = BASE_SEED):
    """3 images × 3 strategies: image overlay with sampled points coloured by class."""
    rng = np.random.default_rng(seed)
    chosen = rng.choice(sample_indices, size=3, replace=False).tolist()

    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    col_titles = ["Image + GT", "Uniform", "Boundary-biased", "Interior-biased"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10)

    for row, idx in enumerate(chosen):
        img, label = load_image_and_label(idx)
        dist = compute_boundary_distance(label)

        # col 0: image blended with GT
        gt_rgb = CLASS_COLORS[label.clip(0, 5)]
        blend  = (img * 0.55 + gt_rgb * 0.45).astype(np.uint8)
        axes[row, 0].imshow(blend)
        axes[row, 0].axis("off")

        for col_i, strategy in enumerate(STRATEGIES, start=1):
            seed_here = seed + idx
            mask = simulate_point_labels(label, POINTS_PER_CLASS, sampling=strategy)
            ys, xs = np.where(mask > 0)
            # Colour each dot by its local purity (green = pure, red = mixed)
            purities = np.array([local_purity(label, y, x) for y, x in zip(ys, xs)])
            axes[row, col_i].imshow(img, alpha=0.6)
            sc = axes[row, col_i].scatter(xs, ys, c=purities, cmap="RdYlGn",
                                          vmin=0, vmax=1, s=25, linewidths=0.5,
                                          edgecolors="black", zorder=5)
            # Draw boundary edge map faintly
            is_boundary = (dist == 0)
            axes[row, col_i].contour(is_boundary, levels=[0.5],
                                      colors=["white"], linewidths=[0.5], alpha=0.5)
            axes[row, col_i].axis("off")

        # shared colorbar on rightmost column
        plt.colorbar(sc, ax=axes[row, 3], fraction=0.046, pad=0.04,
                     label="Local purity")

    patches = [mpatches.Patch(color=np.array(c) / 255.0, label=n)
               for c, n in zip(CLASS_COLORS, CLASS_NAMES)]
    fig.legend(handles=patches, loc="lower center", ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Sampling strategy comparison — dots coloured by local purity\n"
                 "(white contours = class boundaries)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "figS8_sampling_qualitative.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Boundary-biased vs uniform vs interior sampling diagnostic"
    )
    parser.add_argument("--n_images", type=int, default=N_IMAGES)
    parser.add_argument("--seed",     type=int, default=BASE_SEED)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    records, sample_indices = run_analysis(n_images=args.n_images, seed=args.seed)
    save_csv(records, os.path.join(OUT_DIR, "sampling_data.csv"))
    print_summary(records)

    print("\nGenerating figures ...")
    fig_distance_distributions(records)
    fig_purity_distributions(records)
    fig_scatter(records)
    fig_qualitative(sample_indices, seed=args.seed)
    print(f"\nDone. Figures in {FIG_DIR}/")


if __name__ == "__main__":
    main()
