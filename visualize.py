"""Generate all report figures into results/figures/."""

import os
import json
import csv

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image

from data.potsdam_dataset import PotsdamPointDataset, rgb_to_label, simulate_point_labels
from models.segmentation_model import build_model

# ── palette ──────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["Impervious", "Building", "Low veg.", "Tree", "Car", "Clutter"]
CLASS_COLORS = np.array([
    [255, 255, 255],  # 0 impervious
    [0,   0,   255],  # 1 building
    [0,   255, 255],  # 2 low veg
    [0,   255,   0],  # 3 tree
    [255, 255,   0],  # 4 car
    [255,   0,   0],  # 5 clutter
], dtype=np.uint8)

RESULTS_DIR = "results"
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def label_to_rgb(label_idx):
    """H×W int → H×W×3 uint8."""
    return CLASS_COLORS[label_idx.clip(0, 5)]

def load_summary():
    rows = []
    with open(os.path.join(RESULTS_DIR, "summary.csv")) as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if k not in ("run_name","timestamp") else v)
                         for k, v in r.items()})
    return rows

def load_history(run_name):
    path = os.path.join(RESULTS_DIR, "runs", run_name, "history.csv")
    epochs, losses, mious = [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            epochs.append(int(r["epoch"]))
            losses.append(float(r["train_loss"]))
            mious.append(float(r["val_miou"]))
    return epochs, losses, mious

def load_model(run_name):
    ckpt = torch.load(
        os.path.join(RESULTS_DIR, "runs", run_name, "best.pth"),
        map_location=DEVICE)
    cfg = ckpt["config"]
    model = build_model(cfg["num_classes"], cfg["backbone"], pretrained=False).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Baseline training curve (loss + val mIoU)
# ─────────────────────────────────────────────────────────────────────────────
def fig_baseline_curve():
    epochs, losses, mious = load_history("baseline_p10_g2")

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()

    ax1.plot(epochs, losses, color="#e05c2a", linewidth=1.8, label="Train loss")
    ax2.plot(epochs, mious,  color="#2a7ae0", linewidth=1.8, label="Val mIoU")

    best_ep = epochs[mious.index(max(mious))]
    best_v  = max(mious)
    ax2.axvline(best_ep, color="#2a7ae0", linestyle="--", linewidth=1, alpha=0.6)
    ax2.annotate(f"best epoch {best_ep}\nmIoU={best_v:.3f}",
                 xy=(best_ep, best_v), xytext=(best_ep + 2, best_v - 0.04),
                 fontsize=8, color="#2a7ae0",
                 arrowprops=dict(arrowstyle="->", color="#2a7ae0", lw=0.8))

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color="#e05c2a")
    ax2.set_ylabel("Val mIoU",   color="#2a7ae0")
    ax1.tick_params(axis="y", labelcolor="#e05c2a")
    ax2.tick_params(axis="y", labelcolor="#2a7ae0")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=9)
    ax1.set_title("Baseline training curve  (points_per_class=10, γ=2.0)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig1_baseline_curve.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Experiment 1: mIoU vs annotation density
# ─────────────────────────────────────────────────────────────────────────────
def fig_exp1_density():
    rows = load_summary()
    exp1 = sorted([r for r in rows if r["run_name"].startswith("exp1_")],
                  key=lambda r: r["points_per_class"])

    pts   = [int(r["points_per_class"]) for r in exp1]
    mious = [r["test_miou"] for r in exp1]

    per_class = {n: [r[f"iou_{k}"] for r in exp1]
                 for n, k in zip(CLASS_NAMES,
                                 ["impervious","building","low_vegetation",
                                  "tree","car","clutter"])}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: overall mIoU
    ax = axes[0]
    ax.plot(pts, mious, "o-", color="#2a7ae0", linewidth=2, markersize=7)
    for x, y in zip(pts, mious):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks(pts)
    ax.set_xticklabels(pts)
    ax.set_xlabel("points_per_class (log scale)")
    ax.set_ylabel("Test mIoU")
    ax.set_title("Overall mIoU vs annotation density")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(0.50, 0.63)

    # Right: per-class IoU lines
    ax = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 0.6, 6))
    for (name, vals), c in zip(per_class.items(), colors):
        ax.plot(pts, vals, "o-", color=c, linewidth=1.5, markersize=5, label=name)
    ax.set_xscale("log")
    ax.set_xticks(pts)
    ax.set_xticklabels(pts)
    ax.set_xlabel("points_per_class (log scale)")
    ax.set_ylabel("IoU")
    ax.set_title("Per-class IoU vs annotation density")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(0.0, 0.90)

    fig.suptitle("Experiment 1 — Effect of annotation density  (γ = 2.0)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig2_exp1_density.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Experiment 2: mIoU vs focal γ
# ─────────────────────────────────────────────────────────────────────────────
def fig_exp2_gamma():
    rows = load_summary()
    exp2 = sorted([r for r in rows if r["run_name"].startswith("exp2_")],
                  key=lambda r: r["focal_gamma"])

    gammas = [r["focal_gamma"] for r in exp2]
    mious  = [r["test_miou"]   for r in exp2]

    per_class = {n: [r[f"iou_{k}"] for r in exp2]
                 for n, k in zip(CLASS_NAMES,
                                 ["impervious","building","low_vegetation",
                                  "tree","car","clutter"])}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(gammas, mious, "s-", color="#e05c2a", linewidth=2, markersize=7)
    for x, y in zip(gammas, mious):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xticks(gammas)
    ax.set_xlabel("Focal γ")
    ax.set_ylabel("Test mIoU")
    ax.set_title("Overall mIoU vs focal γ")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.555, 0.600)

    ax = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 0.6, 6))
    for (name, vals), c in zip(per_class.items(), colors):
        ax.plot(gammas, vals, "s-", color=c, linewidth=1.5, markersize=5, label=name)
    ax.set_xticks(gammas)
    ax.set_xlabel("Focal γ")
    ax.set_ylabel("IoU")
    ax.set_title("Per-class IoU vs focal γ")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 0.90)

    fig.suptitle("Experiment 2 — Effect of focal weighting  (points_per_class = 10)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig3_exp2_gamma.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Per-class IoU heatmap (all 10 runs)
# ─────────────────────────────────────────────────────────────────────────────
def fig_heatmap():
    rows = load_summary()
    run_order = [
        "exp1_p1_g2","exp1_p5_g2","exp1_p10_g2","exp1_p20_g2","exp1_p50_g2",
        "exp2_p10_g0","exp2_p10_g05","exp2_p10_g1","exp2_p10_g2",
        "baseline_p10_g2",
    ]
    labels_y = [
        "p=1,γ=2","p=5,γ=2","p=10,γ=2","p=20,γ=2","p=50,γ=2",
        "p=10,γ=0","p=10,γ=0.5","p=10,γ=1","p=10,γ=2 (exp2)",
        "baseline",
    ]
    keys = ["impervious","building","low_vegetation","tree","car","clutter","test_miou"]
    col_labels = CLASS_NAMES + ["mIoU"]

    by_name = {r["run_name"]: r for r in rows}
    data = np.array([[by_name[rn][f"iou_{k}"] if k != "test_miou"
                      else by_name[rn]["test_miou"]
                      for k in keys]
                     for rn in run_order])

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=0.85)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="IoU")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels_y)))
    ax.set_yticklabels(labels_y, fontsize=9)

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            color = "black" if 0.25 < v < 0.70 else "white"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.5, color=color)

    # Separate mIoU column with a vertical line
    ax.axvline(5.5, color="white", linewidth=2)

    ax.set_title("Per-class IoU and mIoU — all runs", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig4_heatmap.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — Qualitative prediction grid (4 test images × 4 runs)
# ─────────────────────────────────────────────────────────────────────────────
def fig_qualitative():
    # Pick 4 visually diverse test indices
    test_indices = [2040, 2100, 2200, 2350]
    run_names    = ["exp1_p1_g2", "exp1_p10_g2", "exp1_p50_g2", "baseline_p10_g2"]
    run_labels   = ["p=1, γ=2", "p=10, γ=2", "p=50, γ=2", "baseline (p=10, γ=2)"]

    data_root = "/home/ubuntu/weak-supervision/patches"

    # Load images + GT once
    images_np, gts_np = [], []
    for idx in test_indices:
        img = np.array(Image.open(f"{data_root}/Images/Image_{idx}.tif").convert("RGB"))
        lbl_rgb = np.array(Image.open(f"{data_root}/Labels/Label_{idx}.tif").convert("RGB"))
        images_np.append(img)
        gts_np.append(rgb_to_label(lbl_rgb))

    # Rows: image, GT, then one row per run
    n_rows = 2 + len(run_names)
    n_cols = len(test_indices)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

    row_titles = ["Image", "Ground truth"] + run_labels

    for col, (img, gt) in enumerate(zip(images_np, gts_np)):
        axes[0, col].imshow(img)
        axes[0, col].axis("off")
        axes[1, col].imshow(label_to_rgb(gt))
        axes[1, col].axis("off")

    import torch.nn.functional as FN

    # Load each model and run inference
    for row_i, (run_name, run_label) in enumerate(zip(run_names, run_labels), start=2):
        model, cfg = load_model(run_name)
        for col, img in enumerate(images_np):
            # Preprocess: normalize + pad to 304
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            img_t = (img_t - mean) / std
            # pad to 304
            img_t = FN.pad(img_t.unsqueeze(0), (0, 4, 0, 4), mode="constant", value=0)
            with torch.no_grad():
                logits = model(img_t.to(DEVICE))
            pred = logits.argmax(1)[0, :300, :300].cpu().numpy()
            axes[row_i, col].imshow(label_to_rgb(pred))
            axes[row_i, col].axis("off")

    # Row labels on left
    for row_i, title in enumerate(row_titles):
        axes[row_i, 0].set_ylabel(title, fontsize=9, rotation=0,
                                   labelpad=70, va="center")

    # Legend
    patches = [mpatches.Patch(color=np.array(c)/255, label=n)
               for c, n in zip(CLASS_COLORS, CLASS_NAMES)]
    fig.legend(handles=patches, loc="lower center", ncol=6,
               fontsize=8, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Qualitative segmentation results — test set", fontsize=13)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig5_qualitative.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Point annotation density illustration (1 example image)
# ─────────────────────────────────────────────────────────────────────────────
def fig_point_illustration():
    data_root = "/home/ubuntu/weak-supervision/patches"
    idx = 2100
    img = np.array(Image.open(f"{data_root}/Images/Image_{idx}.tif").convert("RGB"))
    lbl_rgb = np.array(Image.open(f"{data_root}/Labels/Image_{idx}.tif").convert("RGB")
                       if False else
                       Image.open(f"{data_root}/Labels/Label_{idx}.tif").convert("RGB"))
    label = rgb_to_label(lbl_rgb)

    point_counts = [1, 5, 10, 50]
    fig, axes = plt.subplots(1, len(point_counts) + 1, figsize=(14, 3.5))

    axes[0].imshow(img)
    axes[0].set_title("Image", fontsize=10)
    axes[0].axis("off")

    np.random.seed(42)
    for ax, ppc in zip(axes[1:], point_counts):
        mask = simulate_point_labels(label, ppc)
        ys, xs = np.where(mask > 0)
        colors_scatter = [np.array(CLASS_COLORS[label[y, x]]) / 255.0
                          for y, x in zip(ys, xs)]
        ax.imshow(img, alpha=0.5)
        ax.scatter(xs, ys, c=colors_scatter, s=12, linewidths=0.3,
                   edgecolors="white")
        ax.set_title(f"p={ppc}  ({len(ys)} pts)", fontsize=10)
        ax.axis("off")

    patches = [mpatches.Patch(color=np.array(c)/255, label=n)
               for c, n in zip(CLASS_COLORS, CLASS_NAMES)]
    fig.legend(handles=patches, loc="lower center", ncol=6,
               fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Simulated point annotations at different densities", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig6_point_illustration.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 — Experiment 3: SLIC vs. point-only comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig_exp3_slic_comparison():
    """Requires exp1_p10_g2 and exp3_slic_p10_g2 to be complete."""
    import json

    runs = [
        ("exp1_p10_g2",      "Point-only  (p=10, γ=2)",  "#2a7ae0"),
        ("exp3_slic_p10_g2", "SLIC-expanded (p=10, γ=2)", "#e05c2a"),
    ]

    # ── load histories ────────────────────────────────────────────────────────
    histories = {}
    for run_name, _, _ in runs:
        epochs, losses, mious = load_history(run_name)
        histories[run_name] = (epochs, losses, mious)

    # ── load test metrics ─────────────────────────────────────────────────────
    test_metrics = {}
    for run_name, _, _ in runs:
        path = os.path.join(RESULTS_DIR, "runs", run_name, "test_metrics.json")
        with open(path) as f:
            test_metrics[run_name] = json.load(f)

    fig = plt.figure(figsize=(16, 5))
    gs  = fig.add_gridspec(1, 3, wspace=0.35)

    # ── panel 1: val mIoU curves ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    for run_name, label, color in runs:
        epochs, _, mious = histories[run_name]
        ax1.plot(epochs, mious, linewidth=1.8, color=color, label=label)
        best = max(mious)
        best_ep = epochs[mious.index(best)]
        ax1.scatter([best_ep], [best], color=color, s=50, zorder=5)
        ax1.annotate(f"{best:.3f}", (best_ep, best),
                     textcoords="offset points", xytext=(4, 4),
                     fontsize=8, color=color)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val mIoU")
    ax1.set_title("Validation mIoU during training")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # ── panel 2: train loss curves ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    for run_name, label, color in runs:
        epochs, losses, _ = histories[run_name]
        ax2.plot(epochs, losses, linewidth=1.8, color=color, label=label)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Train loss")
    ax2.set_title("Training loss")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # ── panel 3: per-class test IoU bar chart ─────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    cls_keys = ["impervious", "building", "low_vegetation", "tree", "car", "clutter"]
    cls_labels = ["Imperv.", "Building", "Low veg.", "Tree", "Car", "Clutter"]
    x = np.arange(len(cls_keys))
    width = 0.35

    for i, (run_name, label, color) in enumerate(runs):
        pci = test_metrics[run_name]["per_class_iou"]
        vals = [pci[k] for k in cls_keys]
        bars = ax3.bar(x + (i - 0.5) * width, vals, width,
                       label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)

    # mIoU annotations
    for i, (run_name, _, color) in enumerate(runs):
        miou = test_metrics[run_name]["test_miou"]
        ax3.annotate(f"mIoU={miou:.3f}", xy=(0.02 + i * 0.5, 0.97),
                     xycoords="axes fraction", fontsize=8, color=color,
                     va="top", fontweight="bold")

    ax3.set_xticks(x)
    ax3.set_xticklabels(cls_labels, rotation=20, ha="right", fontsize=8)
    ax3.set_ylabel("IoU")
    ax3.set_ylim(0, 1.0)
    ax3.set_title("Test IoU per class")
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Experiment 3 — SLIC label propagation vs. point-only supervision  "
        "(p=10, γ=2.0, 50 epochs)",
        fontsize=12,
    )
    out = os.path.join(FIG_DIR, "fig7_exp3_slic_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8 — Experiment 4: boundary vs interior vs uniform sampling comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig_exp4_sampling_comparison():
    """Requires exp1_p10_g2, exp4_boundary_p10_g2, exp4_interior_p10_g2."""
    import json

    runs = [
        ("exp1_p10_g2",          "Uniform",          "#2a7ae0"),
        ("exp4_boundary_p10_g2", "Boundary-biased",  "#e05c2a"),
        ("exp4_interior_p10_g2", "Interior-biased",  "#4caf50"),
    ]

    histories    = {}
    test_metrics = {}
    for run_name, _, _ in runs:
        histories[run_name]    = load_history(run_name)
        path = os.path.join(RESULTS_DIR, "runs", run_name, "test_metrics.json")
        with open(path) as f:
            test_metrics[run_name] = json.load(f)

    fig = plt.figure(figsize=(16, 5))
    gs  = fig.add_gridspec(1, 3, wspace=0.35)

    # Panel 1: val mIoU curves
    ax1 = fig.add_subplot(gs[0])
    for run_name, label, color in runs:
        epochs, _, mious = histories[run_name]
        ax1.plot(epochs, mious, linewidth=1.8, color=color, label=label)
        best    = max(mious)
        best_ep = epochs[mious.index(best)]
        ax1.scatter([best_ep], [best], color=color, s=50, zorder=5)
        ax1.annotate(f"{best:.3f}", (best_ep, best),
                     textcoords="offset points", xytext=(4, 4),
                     fontsize=8, color=color)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val mIoU")
    ax1.set_title("Validation mIoU during training")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Panel 2: train loss curves
    ax2 = fig.add_subplot(gs[1])
    for run_name, label, color in runs:
        epochs, losses, _ = histories[run_name]
        ax2.plot(epochs, losses, linewidth=1.8, color=color, label=label)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Train loss")
    ax2.set_title("Training loss")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # Panel 3: per-class test IoU grouped bar chart
    ax3 = fig.add_subplot(gs[2])
    cls_keys   = ["impervious", "building", "low_vegetation", "tree", "car", "clutter"]
    cls_labels = ["Imperv.", "Building", "Low veg.", "Tree", "Car", "Clutter"]
    x     = np.arange(len(cls_keys))
    width = 0.25

    for i, (run_name, label, color) in enumerate(runs):
        pci  = test_metrics[run_name]["per_class_iou"]
        vals = [pci[k] for k in cls_keys]
        bars = ax3.bar(x + (i - 1) * width, vals, width,
                       label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=6.0)

    for i, (run_name, _, color) in enumerate(runs):
        miou = test_metrics[run_name]["test_miou"]
        ax3.annotate(f"mIoU={miou:.3f}",
                     xy=(0.02 + i * 0.33, 0.97),
                     xycoords="axes fraction", fontsize=8,
                     color=color, va="top", fontweight="bold")

    ax3.set_xticks(x)
    ax3.set_xticklabels(cls_labels, rotation=20, ha="right", fontsize=8)
    ax3.set_ylabel("IoU")
    ax3.set_ylim(0, 1.0)
    ax3.set_title("Test IoU per class")
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Experiment 4 — Boundary vs interior vs uniform point sampling  "
        "(p=10, γ=2.0, 50 epochs)",
        fontsize=12,
    )
    out = os.path.join(FIG_DIR, "fig8_exp4_sampling_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    print("Generating figures...")
    fig_baseline_curve()
    fig_exp1_density()
    fig_exp2_gamma()
    fig_heatmap()
    fig_qualitative()
    fig_point_illustration()

    # Experiment 3 figure — only generated if the SLIC run is complete
    slic_metrics = os.path.join(RESULTS_DIR, "runs", "exp3_slic_p10_g2", "test_metrics.json")
    if os.path.exists(slic_metrics):
        fig_exp3_slic_comparison()
    else:
        print("Skipping fig7 — exp3_slic_p10_g2 not yet complete")

    # Experiment 4 figure — only generated if both sampling runs are complete
    boundary_metrics = os.path.join(RESULTS_DIR, "runs", "exp4_boundary_p10_g2", "test_metrics.json")
    interior_metrics = os.path.join(RESULTS_DIR, "runs", "exp4_interior_p10_g2", "test_metrics.json")
    if os.path.exists(boundary_metrics) and os.path.exists(interior_metrics):
        fig_exp4_sampling_comparison()
    else:
        print("Skipping fig8 — exp4 runs not yet complete")

    print("Done. All figures in", FIG_DIR)
