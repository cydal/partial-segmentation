"""Article 3 figures — segmentation from sparse point labels.

Regenerates fig-1 … fig-4 (+ fig-2-header) from the v2 reruns and the existing
analysis CSVs. Every mIoU number is read from results/v2/summary.csv; purity
numbers from the analysis CSVs. Nothing is hard-coded to match the article.

Run after the v2 sweep finishes:
    python make_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from PIL import Image
from skimage.segmentation import slic, mark_boundaries

from data.potsdam_dataset import rgb_to_label, simulate_point_labels, _point_rng

# ----------------------------------------------------------------------------- paths
HERE        = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT   = "/home/ubuntu/weak-supervision/patches"
RESULTS_V2  = os.path.join(HERE, "results", "v2")
SUMMARY_CSV = os.path.join(RESULTS_V2, "summary.csv")
PURITY_CSV  = os.path.join(HERE, "results", "slic_analysis", "purity_data.csv")
SAMPLING_CSV= os.path.join(HERE, "results", "boundary_analysis", "sampling_data.csv")
OUT_DIR     = os.path.join(RESULTS_V2, "figures", "article-3")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------- style
# One clean sans-serif, British spelling and sentence case in all text.
mpl.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "axes.edgecolor":   "#333333",
    "axes.linewidth":   0.8,
    "xtick.color":      "#333333",
    "ytick.color":      "#333333",
    "text.color":       "#222222",
    "axes.labelcolor":  "#222222",
    "figure.facecolor": "white",
    "savefig.facecolor":"white",
})

# Potsdam class colours (impervious is white → outline it so it stays visible).
CLASS_NAMES  = ["Impervious surfaces", "Building", "Low vegetation",
                "Tree", "Car", "Clutter"]
CLASS_COLOURS = [
    (1.00, 1.00, 1.00),   # 0 impervious  (white)
    (0.00, 0.00, 1.00),   # 1 building    (blue)
    (0.00, 1.00, 1.00),   # 2 low veg     (cyan)
    (0.00, 1.00, 0.00),   # 3 tree        (green)
    (1.00, 1.00, 0.00),   # 4 car         (yellow)
    (1.00, 0.00, 0.00),   # 5 clutter     (red)
]

GREY      = "#8a8a8a"     # ordinary conditions
SLIC_CLR  = "#5D3A9B"     # violet highlight for SLIC
BAND_CLR  = "#d9d9d9"     # reference noise band (light grey)

DPI = 200                 # 1400 px wide at 7 in → export at 2x via dpi
PX  = 1400


def _figsize(width_px, height_px):
    return (width_px / DPI, height_px / DPI)


# ----------------------------------------------------------------------------- data
def load_summary():
    df = pd.read_csv(SUMMARY_CSV)
    # Normalise dtypes we rely on.
    for c in ["points_per_class", "seed", "point_seed"]:
        if c in df:
            df[c] = df[c].astype(int)
    return df


def agg(df):
    """Return {run_name: (mean, lo, hi, n)} keyed handling of multi-seed groups.

    We identify conditions by (points_per_class, focal_gamma, sampling, use_slic-ish
    via run_name prefix). Simpler: group by the semantic label we assign below.
    """
    return df


def cond_stats(df, mask):
    """mean / min / max / n of test_miou over rows selected by boolean mask."""
    sub = df[mask]
    if len(sub) == 0:
        return None
    v = sub["test_miou"].values
    return dict(mean=float(v.mean()), lo=float(v.min()), hi=float(v.max()), n=len(v))


# ============================================================================= fig 1
def fig1_point_labels():
    """One test patch: image, then 1 / 10 / 50 points per class (nested)."""
    img_idx = 2071  # test patch with 5 well-represented classes incl. car
    img = np.array(Image.open(f"{DATA_ROOT}/Images/Image_{img_idx}.tif").convert("RGB"))
    lab = rgb_to_label(np.array(Image.open(f"{DATA_ROOT}/Labels/Label_{img_idx}.tif").convert("RGB")))

    panels = [("Image", None), ("1 per class", 1),
              ("10 per class", 10), ("50 per class", 50)]
    fig, axes = plt.subplots(1, 4, figsize=_figsize(PX, 430))
    for ax, (title, n) in zip(axes, panels):
        # Light dim so points read clearly, but far less than the old illustration.
        ax.imshow((img.astype(np.float32) * 0.85 + 255 * 0.15).astype(np.uint8))
        if n is not None:
            rng = _point_rng(0, img_idx)         # same seed → nested across panels
            pm = simulate_point_labels(lab, n, "uniform", rng=rng)
            ys, xs = np.where(pm > 0)
            cls = lab[ys, xs]
            cols = np.array([CLASS_COLOURS[c] for c in cls])
            ax.scatter(xs, ys, s=14, c=cols, edgecolors="#1a1a1a",
                       linewidths=0.5, zorder=3)
        ax.set_title(title, fontsize=12, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#bbbbbb")

    handles = [mpatches.Patch(facecolor=CLASS_COLOURS[i], edgecolor="#333333",
                              label=CLASS_NAMES[i]) for i in range(6)]
    fig.legend(handles=handles, ncol=6, loc="lower center", frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.02), handlelength=1.2,
               columnspacing=1.2)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.11, wspace=0.03)
    out = os.path.join(OUT_DIR, "fig-1.png")
    fig.savefig(out, dpi=DPI * 2)
    plt.close(fig)
    print("wrote", out)


# ============================================================================= fig 2
def _fig2_rows(df):
    """Build the ordered list of (group, label, stats, is_slic) for the dot plot."""
    ref = cond_stats(df, (df.run_name.str.startswith("ref_s")))
    rows = []

    # points per class
    rows.append(("group", "Points per class", None, False))
    for p, name in [(1, "1"), (5, "5"), (10, "10"), (20, "20"), (50, "50")]:
        if p == 10:
            rows.append(("row", "10 (reference)", ref, False)); continue
        st = cond_stats(df, (df.points_per_class == p) & (df.focal_gamma == 2.0) &
                            (df.sampling == "uniform") & (df.run_name.str.contains(f"p{p}_g2")))
        rows.append(("row", name, st, False))

    # focal gamma
    rows.append(("group", "Focal γ", None, False))
    for g, rn in [(0.0, "g0_s0"), (0.5, "g05_s0"), (1.0, "g1_s0")]:
        st = cond_stats(df, df.run_name == rn)
        rows.append(("row", f"{g:g}", st, False))
    rows.append(("row", "2 (reference)", ref, False))

    # label spreading
    rows.append(("group", "Label spreading", None, False))
    st = cond_stats(df, df.run_name == "slic_s0")
    rows.append(("row", "SLIC", st, True))

    # point placement
    rows.append(("group", "Point placement", None, False))
    rows.append(("row", "Interior", cond_stats(df, df.run_name == "interior_s0"), False))
    rows.append(("row", "Uniform (reference)", ref, False))
    rows.append(("row", "Boundary", cond_stats(df, df.run_name == "boundary_s0"), False))
    return rows, ref


def fig2_conditions(header=False):
    df = load_summary()
    rows, ref = _fig2_rows(df)

    # Layout: build y positions top-down, with a little gap before each group.
    labels, ys, kinds, stats_list, slic_flags = [], [], [], [], []
    y = 0.0
    for kind, label, st, is_slic in rows:
        if kind == "group":
            y -= 0.6
            labels.append(label); ys.append(y); kinds.append("group")
            stats_list.append(None); slic_flags.append(False)
        else:
            labels.append(label); ys.append(y); kinds.append("row")
            stats_list.append(st); slic_flags.append(is_slic)
        y -= 1.0

    h_px = 788 if header else max(560, int(len(rows) * 46))
    fig, ax = plt.subplots(figsize=_figsize(PX, h_px))

    # Reference noise band (min–max across the reference seed runs).
    if ref is not None:
        ax.axvspan(ref["lo"], ref["hi"], color=BAND_CLR, zorder=0)
        ax.text(ref["hi"], ys[-1] - 1.0, f"  reference, {ref['n']} seeds",
                va="center", ha="left", fontsize=9, color="#666666")

    yticks, yticklabels = [], []
    for label, yy, kind, st, is_slic in zip(labels, ys, kinds, stats_list, slic_flags):
        if kind == "group":
            ax.text(-0.002, yy, label, transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontweight="bold", fontsize=10.5)
            continue
        yticks.append(yy); yticklabels.append(label)
        if st is None:
            continue
        colour = SLIC_CLR if is_slic else GREY
        # min–max bar when several seeds.
        if st["n"] > 1:
            ax.plot([st["lo"], st["hi"]], [yy, yy], color=colour, lw=2.0,
                    solid_capstyle="round", zorder=2, alpha=0.7)
        ax.scatter([st["mean"]], [yy], s=70 if is_slic else 55, color=colour,
                   edgecolors="white", linewidths=1.0, zorder=3)
        ax.annotate(f"{st['mean']:.3f}", (st["mean"], yy),
                    textcoords="offset points", xytext=(0, 8), ha="center",
                    fontsize=8.5, color=colour if is_slic else "#333333",
                    fontweight="bold" if is_slic else "normal")

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=9.5)
    ax.set_ylim(ys[-1] - 1.4, 0.4)
    ax.set_xlabel("Test mIoU")

    # x limits from data with margin.
    allv = [s["lo"] for s in stats_list if s] + [s["hi"] for s in stats_list if s]
    lo, hi = min(allv), max(allv)
    pad = (hi - lo) * 0.12 + 0.005
    ax.set_xlim(lo - pad, hi + pad * 2.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#eeeeee", lw=0.8, zorder=-1)
    fig.subplots_adjust(left=0.20, right=0.975, top=0.97, bottom=0.13 if not header else 0.16)

    name = "fig-2-header.png" if header else "fig-2.png"
    out = os.path.join(OUT_DIR, name)
    fig.savefig(out, dpi=DPI * 2)
    plt.close(fig)
    print("wrote", out)


# ============================================================================= fig 3
def fig3_slic_purity():
    """Image | SLIC boundaries | superpixels with a labelled point, by purity."""
    img_idx = 2071
    img = np.array(Image.open(f"{DATA_ROOT}/Images/Image_{img_idx}.tif").convert("RGB"))
    lab = rgb_to_label(np.array(Image.open(f"{DATA_ROOT}/Labels/Label_{img_idx}.tif").convert("RGB")))
    img_f = img.astype(np.float32) / 255.0
    segments = slic(img_f, n_segments=200, compactness=10.0, start_label=0, channel_axis=2)

    rng = _point_rng(0, img_idx)
    pm = simulate_point_labels(lab, 10, "uniform", rng=rng)

    # Purity per superpixel that contains a labelled point.
    purity_img = np.full(segments.shape, np.nan, dtype=np.float32)
    for row, col in np.argwhere(pm > 0):
        seg = segments[row, col]
        sp = segments == seg
        pc = int(lab[row, col])
        purity_img[sp] = float((lab[sp] == pc).mean())

    fig, axes = plt.subplots(1, 3, figsize=_figsize(PX, 500))
    axes[0].imshow(img); axes[0].set_title("Image", fontsize=12, pad=6)

    axes[1].imshow(mark_boundaries(img_f, segments, color=(1, 1, 0), mode="thin"))
    axes[1].set_title("SLIC superpixels (200)", fontsize=12, pad=6)

    axes[2].imshow(img, alpha=0.35)
    im = axes[2].imshow(np.ma.masked_invalid(purity_img), cmap="viridis",
                        vmin=0.0, vmax=1.0)
    ys, xs = np.where(pm > 0)
    axes[2].scatter(xs, ys, s=12, c="white", edgecolors="#1a1a1a",
                    linewidths=0.5, zorder=3)
    axes[2].set_title("Purity of labelled superpixels", fontsize=12, pad=6)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#bbbbbb")

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.015)
    cbar.set_label("Superpixel purity")
    fig.subplots_adjust(left=0.005, right=0.93, top=0.93, bottom=0.02, wspace=0.03)
    out = os.path.join(OUT_DIR, "fig-3.png")
    fig.savefig(out, dpi=DPI * 2)
    plt.close(fig)
    print("wrote", out)


# ============================================================================= fig 4
def fig4_purity_vs_miou():
    """Scatter: local label purity vs test mIoU for interior / uniform / boundary."""
    df = load_summary()
    samp = pd.read_csv(SAMPLING_CSV)
    purity = samp.groupby("strategy")["local_purity"].mean().to_dict()

    strat_run = {
        "interior": df.run_name == "interior_s0",
        "uniform":  df.run_name.str.startswith("ref_s"),   # reference == uniform 10pt g2
        "boundary": df.run_name == "boundary_s0",
    }

    fig, ax = plt.subplots(figsize=_figsize(PX, 820))
    pts = {}
    for strat in ["interior", "uniform", "boundary"]:
        st = cond_stats(df, strat_run[strat])
        x = purity[strat]
        pts[strat] = (x, st)
        ax.scatter([x], [st["mean"]], s=90, color=GREY,
                   edgecolors="#333333", linewidths=1.0, zorder=3)
        if st["n"] > 1:
            ax.plot([x, x], [st["lo"], st["hi"]], color=GREY, lw=2.0, zorder=2)
        lbl = strat.capitalize()
        # Place interior's label to the left so it doesn't run off the right edge.
        ha, dx = ("right", -10) if strat == "interior" else ("left", 10)
        ax.annotate(f"{lbl}\n({x:.3f}, {st['mean']:.3f})", (x, st["mean"]),
                    textcoords="offset points", xytext=(dx, 8), ha=ha,
                    fontsize=9.5, color="#333333")

    ax.set_xlabel("Local label purity (11×11 patch)")
    ax.set_ylabel("Test mIoU")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#eeeeee", lw=0.8, zorder=-1)

    xs = [x for x, _ in pts.values()]
    ys_all = [s["lo"] for _, s in pts.values()] + [s["hi"] for _, s in pts.values()]
    xpad = (max(xs) - min(xs)) * 0.18
    ypad = (max(ys_all) - min(ys_all)) * 0.18
    ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
    ax.set_ylim(min(ys_all) - ypad, max(ys_all) + ypad)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.13)
    out = os.path.join(OUT_DIR, "fig-4.png")
    fig.savefig(out, dpi=DPI * 2)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["1", "2", "3", "4"]
    if "1" in which: fig1_point_labels()
    if "3" in which: fig3_slic_purity()
    if "4" in which: fig4_purity_vs_miou()
    if "2" in which:
        fig2_conditions(header=False)
        fig2_conditions(header=True)
