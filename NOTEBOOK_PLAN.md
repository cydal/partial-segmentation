# Notebook Implementation Plan

Standalone Jupyter notebook demonstrating the full project without re-running
any of the 50-epoch training experiments. Three execution tiers:

- **Runs live** — loss demos, single-image visualisations, 3-epoch mini-train
- **Loads from disk** — training histories and test metrics from `results/`
- **Embeds figures** — plots re-drawn inline from saved CSVs

---

## Paths (configure once at the top of the notebook)

```python
DATA_ROOT    = "/home/ubuntu/weak-supervision/patches"   # raw images + labels
RESULTS_DIR  = "/home/ubuntu/partial-ce-segmentation/results"
```

Everything else is derived from these two variables. The notebook can live
anywhere on disk.

---

## Section 0 — Setup and Imports

**Execution tier:** live (seconds)

**What it does:**
- `pip install` any missing packages in one cell
- All imports in one cell
- Define DATA_ROOT and RESULTS_DIR
- Quick sanity check: print number of images found, confirm GPU/CPU

**Nothing to copy from the codebase — pure boilerplate.**

---

## Section 1 — Partial Cross-Entropy Loss

**Execution tier:** live (seconds)

**Purpose:** Implement and demonstrate the core loss function.
Show what it does mathematically, then verify it numerically.

**What to copy:**
- `losses/partial_ce.py` — entire file (~35 lines), pasted verbatim into a
  single code cell with a markdown explanation above it.

**Demo cells (no dataset needed):**
1. Create a random `[B=2, C=6, H=32, W=32]` logits tensor, a label tensor,
   and a sparse point mask (only ~1% of pixels set to 1).
2. Call the loss. Print the scalar value.
3. Vary γ ∈ {0, 0.5, 1, 2} and show in a small table how the loss value and
   effective weight on easy vs hard examples changes.
4. Show what happens when point_mask is all-zeros (edge case — loss returns 0).

**Key teaching point:** the loss is identical to focal CE except it ignores
every pixel not in point_mask. Setting γ=0 recovers standard partial CE.

---

## Section 2 — Data and Point Simulation

**Execution tier:** live (seconds)

**Purpose:** Show the Potsdam dataset, the label colour map, and how sparse
point annotations are simulated.

**What to copy from `data/potsdam_dataset.py`:**
- `COLOR_TO_CLASS` dict
- `IMAGENET_MEAN`, `IMAGENET_STD`
- `rgb_to_label()` function
- `simulate_point_labels()` function (the full updated version with
  `sampling` parameter)
- `compute_boundary_distance()` function

**Do NOT copy:** the Dataset class, augmentation pipeline, or
`propagate_points_to_superpixels()` (those appear in later sections).

**Demo cells:**
1. Load one training image and its label with PIL. Display side by side:
   raw image | colour-coded label map with legend.
2. Show point masks at p = 1, 5, 10, 50 overlaid on the image (mirrors
   fig6 from the report, but rendered live).
3. Print a small table: for the displayed image, how many pixels per class
   exist, and how many are sampled at p=10.

**Key teaching point:** the supervision is extremely sparse — at p=10 on a
300×300 image, fewer than 0.2% of pixels carry a label.

---

## Section 3 — The Segmentation Model

**Execution tier:** live (seconds)

**Purpose:** Introduce the model architecture.

**What to copy:**
- `models/segmentation_model.py` — entire file (~15 lines).

**Demo cells:**
1. Instantiate the model (pretrained=False to avoid a download).
2. Print total parameter count.
3. Run one forward pass on a `[1, 3, 304, 304]` dummy tensor.
   Print input shape → output shape. Confirm `[1, 6, 304, 304]`.

**Key teaching point:** DeepLabV3+ takes a full image and outputs a
per-pixel class score map. The loss only reads scores at labeled locations.

---

## Section 4 — Mini Training Demo

**Execution tier:** live (~2–5 minutes depending on GPU)

**Purpose:** Prove the end-to-end pipeline works without running a full
50-epoch experiment.

**What to copy from `train.py`:**
- `train_one_epoch()` function
- `validate()` function
- The DataLoader setup block (4–5 lines)

**What to copy from `data/potsdam_dataset.py`:**
- The full `PotsdamPointDataset` class (now needed for real data loading)

**What to copy from `utils/metrics.py`:**
- `AverageMeter` and `compute_miou()`

**Demo cells:**
1. Instantiate the dataset with a 50-image subset of the train split
   (slice `indices` to the first 50 after construction).
2. Train for 3 epochs with p=10, γ=2. Print loss and val mIoU each epoch.
3. Show a small plot: loss curve and mIoU curve over the 3 epochs.
4. Run inference on one held-out image. Display:
   input image | ground truth | model prediction after 3 epochs.

**Key teaching point:** even after 3 epochs on 50 images the loss is
falling and the model is starting to produce plausible segmentations —
demonstrating that partial CE provides a real learning signal.

**Note:** do not save a checkpoint. This section is demo-only.

---

## Section 5 — Experiment 1: Effect of Annotation Density

**Execution tier:** load from disk (instant)

**Purpose:** Answer "how many clicks per class do you need?"

**What to load:**
- `results/summary.csv` — filter for `run_name.startswith("exp1_")`
- `results/runs/exp1_*/history.csv` — one file per density condition

**Demo cells:**
1. Print the results table (run, pts/class, test mIoU, per-class IoU).
2. Recreate the mIoU-vs-density curve (mirrors fig2 left panel).
3. Recreate the per-class IoU line chart (mirrors fig2 right panel).

**Key teaching point:** annotation density has a clear effect but with
diminishing returns. The plateau around p=10–20 is the practical sweet spot.

---

## Section 6 — Experiment 2: Effect of Focal Weighting (γ)

**Execution tier:** load from disk (instant)

**Purpose:** Answer "does it help to focus on hard examples?"

**What to load:**
- `results/summary.csv` — filter for `run_name.startswith("exp2_")`

**Demo cells:**
1. Print the results table (run, γ, test mIoU, per-class IoU).
2. Recreate the mIoU-vs-γ curve (mirrors fig3 left panel).
3. Brief explanation of what γ does mechanically (with a plot of the
   focal weight `(1-p)^γ` vs confidence `p` for γ ∈ {0, 0.5, 1, 2}).

**Key teaching point:** moderate focal weighting (γ=0.5–1) helps; high
γ (2+) is counterproductive when supervision is already sparse because
it discards too many of the few labeled pixels.

---

## Section 7 — SLIC Label Propagation

**Execution tier:** live for single-image demo, load from disk for aggregates

**Purpose:** Show that each labeled point can supervise ~560 pixels at
74% average purity — and that this dramatically improves training.

### Part A — Live single-image demo

**What to copy from `analyze_slic_purity.py`:**
- `propagate_points_to_superpixels()` (or import from `data/potsdam_dataset.py`)
- The purity measurement loop from `analyze_image()`

**Demo cells:**
1. Load one image. Compute SLIC at n_segments=200.
2. Simulate p=10 points. Expand to superpixels.
3. Display 4-panel figure:
   image | ground truth | SLIC boundaries | purity heatmap with point dots.
4. Print: N points → N superpixels expanded → total supervised pixels
   → mean purity.

### Part B — Aggregate diagnostic (load from disk)

**What to load:**
- `results/slic_analysis/purity_data.csv`

**Demo cells:**
1. Recreate the n_segments sweep table (purity vs coverage).
2. Recreate the granularity tradeoff plot (mirrors figS3).
3. Per-class purity bar chart.

### Part C — Training result (load from disk)

**What to load:**
- `results/runs/exp3_slic_p10_g2/history.csv`
- `results/runs/exp1_p10_g2/history.csv`
- Both `test_metrics.json` files

**Demo cells:**
1. Side-by-side val mIoU curves: point-only vs SLIC-expanded.
2. Results comparison table with Δ column.

**Key teaching point:** one click → 560 supervised pixels at 84% average
purity. This single change (+3.7 pp) outperforms collecting 5× more clicks
(p=50 gains only +2.6 pp over p=10).

---

## Section 8 — Boundary-Biased Sampling

**Execution tier:** live for single-image demo, load from disk for aggregates

**Purpose:** Answer "should annotators click near class edges or safely
in the interior?"

### Part A — Live single-image demo

**What to copy from `analyze_boundary_sampling.py`:**
- `local_purity()` function
- The weighted sampling logic (already in the updated
  `simulate_point_labels()` from Section 2)

**Demo cells:**
1. Load one image and compute `compute_boundary_distance()`.
2. Display the distance map as a heatmap.
3. Sample p=10 with all three strategies. Show the three point overlays
   side by side, dots coloured by local purity.

### Part B — Aggregate diagnostic (load from disk)

**What to load:**
- `results/boundary_analysis/sampling_data.csv`

**Demo cells:**
1. Recreate the summary table: mean distance and mean purity per strategy.
2. Recreate the distance violin plot (mirrors figS5).
3. Recreate the purity violin + per-class bar chart (mirrors figS6).

### Part C — Training result (load from disk)

**What to load:**
- `results/runs/exp4_boundary_p10_g2/history.csv`
- `results/runs/exp4_interior_p10_g2/history.csv`
- All three `test_metrics.json` files (uniform, boundary, interior)

**Demo cells:**
1. Three-way val mIoU curve comparison.
2. Per-class results table with Δ columns.

**Key teaching point:** interior clicks have 97% local purity yet produce
the worst model. Boundary clicks have 85% purity and produce the best.
Discriminative location beats label cleanliness — but the benefit is
class-conditional (helps diffuse classes, hurts sharp ones).

---

## Section 9 — Summary and Conclusions

**Execution tier:** load from disk (instant)

**Demo cells:**
1. Full ranking table: all 13 runs sorted by test mIoU.
2. A single bar chart comparing the five key conditions:
   uniform p=10 | uniform p=50 | SLIC p=10 | boundary p=10 | interior p=10.
3. Written narrative (markdown cell) summarising the three findings:
   - Density matters, but has diminishing returns
   - SLIC expansion is the dominant win: coverage > precision
   - Boundary sampling has a real but class-conditional benefit
   - Interior sampling (highest purity) is the worst performer — the
     purity paradox

---

## Copy-paste reference table

| Notebook section | Source file | What to copy |
|---|---|---|
| 1 — Loss | `losses/partial_ce.py` | Entire file |
| 2 — Data | `data/potsdam_dataset.py` | Lines 1–100: COLOR_TO_CLASS, IMAGENET_MEAN/STD, rgb_to_label, simulate_point_labels, compute_boundary_distance |
| 3 — Model | `models/segmentation_model.py` | Entire file |
| 4 — Mini train | `train.py` | train_one_epoch(), validate() |
| 4 — Mini train | `data/potsdam_dataset.py` | PotsdamPointDataset class |
| 4 — Mini train | `utils/metrics.py` | AverageMeter, compute_miou |
| 7A — SLIC demo | `data/potsdam_dataset.py` | propagate_points_to_superpixels() |
| 7A — SLIC demo | `analyze_slic_purity.py` | analyze_image() purity loop |
| 8A — Boundary demo | `analyze_boundary_sampling.py` | local_purity() |

---

## What is never re-run

| What | Why |
|---|---|
| All 13 fifty-epoch training runs | Already complete; results loaded from CSV/JSON |
| Full SLIC purity sweep (9,225 records) | Loaded from results/slic_analysis/purity_data.csv |
| Full boundary sampling analysis (27,675 records) | Loaded from results/boundary_analysis/sampling_data.csv |
| All report figures (fig1–fig8, figS1–figS8) | Can be embedded with IPython.display if desired |
