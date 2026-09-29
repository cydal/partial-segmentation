# Partial Focal Cross-Entropy for Point-Supervised Segmentation

Weakly supervised semantic segmentation of ISPRS Potsdam aerial imagery from
**sparse point labels** instead of dense pixel masks. A DeepLabV3+ network is
trained with a **Partial Focal Cross-Entropy (pfCE)** loss that backpropagates
only through a handful of labelled pixels per class per image, simulating the
budget of a human annotator who clicks a few points rather than painting every
pixel.

The repository contains the full pipeline — data loading, loss, training,
evaluation — plus the ablation study, pre-training diagnostics, and the figures
for the accompanying article.

---

## Key idea

Full segmentation masks are expensive. Point supervision asks: *if an annotator
only clicks `k` points per class in each image, how good a segmenter can we
train?* The loss is a masked focal cross-entropy — standard focal CE, but summed
only over labelled pixels and normalised by their count:

```
pfCE = Σ( focal_weight · (−log p_t) · point_mask ) / Σ(point_mask)
focal_weight = (1 − p_t)^γ         # γ = 0 recovers plain masked CE
```

Everything else (the other ~99.99% of pixels) contributes no gradient. Evaluation
is always against the **full ground-truth masks**, so mIoU measures real
segmentation quality, not point accuracy.

---

## Results at a glance

From the v2 reruns (`results/v2/`), test mIoU on the 360-image test split:

| Ablation | Finding |
|---|---|
| **Points per class** (1→50) | 0.491 → 0.551 → 0.554 → 0.569 → 0.572. Density helps, saturating above ~10 points. |
| **Focal γ** (0, 0.5, 1, 2) | 0.554–0.565 — all within the seed noise band; γ barely matters. |
| **Label spreading (SLIC)** | **0.603** — the only condition clearly above the noise band. Expanding points to superpixels genuinely helps. |
| **Point placement** | interior 0.532 < boundary 0.553 ≈ uniform 0.554. *Cleaner (interior) labels were **not** better* — boundary pixels drive IoU. |

Reference condition (10 points, γ = 2, uniform) over 3 seeds: mean **0.5545**,
noise band **[0.544, 0.560]**. See [`results/v2/NOTES.md`](results/v2/NOTES.md)
for the full tables and the v1→v2 comparison.

---

## Repository layout

```
configs/
  base.yaml              # default hyperparameters
  v2.yaml                # base.yaml with results_dir = ./results/v2
data/
  potsdam_dataset.py     # PotsdamPointDataset: RGB→class decode, point sampling, SLIC, augment
losses/
  partial_ce.py          # PartialFocalCELoss
models/
  segmentation_model.py  # DeepLabV3+ (ResNet-50) via segmentation-models-pytorch
utils/
  metrics.py             # compute_miou, AverageMeter
train.py                 # training loop (seeded, config + CLI overrides)
evaluate.py              # test-split evaluation, per-class IoU, summary.csv upsert
make_figures.py          # article figures fig-1 … fig-4 (+ fig-2-header)
checks_fixed_points.py   # determinism / nesting / labelled-pixel-count checks
analyze_slic_purity.py       # pre-training diagnostic: SLIC superpixel purity
analyze_boundary_sampling.py # pre-training diagnostic: purity vs placement strategy
visualize.py             # qualitative result plots
run_v2.sh                # orchestrates the v2 experiment sweep
results/
  runs/, summary.csv     # v1 results (original, unseeded points)
  v2/                    # v2 results (fixed seeded points) — see NOTES.md
  slic_analysis/, boundary_analysis/   # diagnostic CSVs
REPORT.md                # write-up of the ablation study
```

---

## Setup

Requires Python 3.10+ and a CUDA GPU (runs were done on an A10G).

```bash
pip install torch torchvision albumentations segmentation-models-pytorch \
            tqdm pyyaml matplotlib scikit-learn pillow numpy pandas scikit-image scipy
```

### Data

ISPRS Potsdam, pre-cropped into 2,400 patches of 300×300 RGB:

```
$DATA_ROOT/
├── Images/   Image_0.tif … Image_2399.tif      # RGB uint8
└── Labels/   Label_0.tif … Label_2399.tif      # RGB uint8, colour-coded
```

Set `data_root` in `configs/base.yaml` (default: `/home/ubuntu/weak-supervision/patches`).

Label colours → class indices:

| RGB | Idx | Class |
|---|---|---|
| (255,255,255) | 0 | Impervious surfaces |
| (0,0,255) | 1 | Building |
| (0,255,255) | 2 | Low vegetation |
| (0,255,0) | 3 | Tree |
| (255,255,0) | 4 | Car |
| (255,0,0) | 5 | Clutter / background |

Split by index: train `0–1679`, val `1680–2039`, test `2040–2399`.

---

## Usage

### Train

```bash
python train.py --config configs/v2.yaml --run_name my_run \
    --points_per_class 10 --focal_gamma 2.0 --sampling uniform \
    --seed 0 --point_seed 0
```

CLI flags override the YAML: `--points_per_class`, `--focal_gamma`, `--epochs`,
`--sampling {uniform,boundary,interior}`, `--use_slic`, `--slic_n_segments`,
`--seed`, `--point_seed`, `--fixed_points/--no_fixed_points`.

Each run writes to `<results_dir>/runs/<run_name>/`: `config.yaml`, `history.csv`
(per-epoch train loss + val mIoU), and `best.pth` (best-val-mIoU checkpoint).

### Evaluate

```bash
python evaluate.py --checkpoint results/v2/runs/my_run/best.pth
```

Prints per-class IoU and mIoU on the test split, writes `test_metrics.json`, and
upserts a row into `<results_dir>/summary.csv`.

### Reproduce the full study

```bash
bash run_v2.sh          # trains + evaluates all conditions into results/v2/
python make_figures.py  # regenerates fig-1 … fig-4 into results/v2/figures/article-3/
```

`run_v2.sh` is idempotent — a run whose `test_metrics.json` already exists is
skipped, so it can be re-invoked safely after an interruption.

### Diagnostics and checks

```bash
python checks_fixed_points.py       # verify point sampling is fixed & nested
python analyze_slic_purity.py       # SLIC superpixel purity sweep
python analyze_boundary_sampling.py # purity vs placement strategy
```

---

## Point sampling (the fixed, seeded scheme)

Points are sampled **once per image**, on the un-augmented label, from a generator
seeded by `(point_seed, image_index)`, then transformed together with the image so
labelled pixels stay put across epochs. This models a fixed annotation budget — a
human clicks once and the labels don't move.

- **Uniform** sampling draws one permutation of each class's pixels and takes the
  first `n`, so densities are **nested**: the 1-point set ⊂ 5-point ⊂ 10-point ⊂ …
  Density comparisons then differ only in the number of points.
- **Boundary / interior** sampling weight pixels by distance to the nearest class
  boundary (near vs far), drawn without replacement from the same seeded generator.
- **SLIC** label spreading computes superpixels once per image (cached to disk) and
  expands each point to its whole superpixel before augmentation.

Set `fixed_points: false` to restore the original v1 behaviour (points re-drawn
every epoch, after augmentation) for reproducing the old results.

> **v1 vs v2.** The original results (`results/runs/`, `results/summary.csv`) used
> unseeded points re-drawn every epoch, which inflated the effective annotation
> budget (a "k-point" model saw up to 50×k distinct pixels over 50 epochs). The v2
> reruns fix this; all v2 mIoU values are correspondingly lower, but the qualitative
> conclusions are unchanged. Details in [`results/v2/NOTES.md`](results/v2/NOTES.md).

---

## Configuration

Defaults in `configs/base.yaml`:

| Key | Default | Meaning |
|---|---|---|
| `num_classes` | 6 | |
| `batch_size` | 64 | |
| `backbone` | resnet50 | DeepLabV3+ encoder |
| `pretrained` | true | ImageNet encoder weights |
| `focal_gamma` | 2.0 | focal exponent (0 = plain masked CE) |
| `points_per_class` | 10 | labelled points sampled per class per image |
| `sampling` | uniform | `uniform` / `boundary` / `interior` |
| `use_slic` / `slic_n_segments` | false / 200 | SLIC label spreading |
| `fixed_points` | true | sample points once per image (seeded) |
| `point_seed` / `seed` | 0 / 0 | point placement seed / training seed |
| `epochs`, `lr`, `weight_decay`, `optimizer` | 50, 1e-4, 1e-4, adam | |

Training seeds `torch`, `numpy`, `random` and the DataLoader workers; `point_seed`
is independent so the effect of the point choice can be separated from
initialisation and shuffling.

---

## See also

- [`REPORT.md`](REPORT.md) — full write-up of the ablation study.
- [`results/v2/NOTES.md`](results/v2/NOTES.md) — v2 code changes, checks, and result tables.
- [`results/v2/figures/article-3/`](results/v2/figures/article-3/) — article figures + `MISMATCHES.md`.
