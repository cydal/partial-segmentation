# v2 reruns — fixed, seeded, nested point sampling

This documents the Part 1 changes from the article-3 brief: fixing how point
labels are sampled, adding reproducibility, and rerunning the ablations. New
results live in `results/v2/`; nothing in `results/runs/` or `results/summary.csv`
(the v1 results) was touched.

## 1. Code changes

All on branch `fixed-points`.

### `data/potsdam_dataset.py`
- **Fixed points per image.** `PotsdamPointDataset.__getitem__` now samples points
  once, on the *un-augmented* label, using a generator seeded by
  `(point_seed, image index)` (`_point_rng`, line 63; used at line 245). Previously
  `simulate_point_labels` used an unseeded `np.random.default_rng()` and was called
  *after* augmentation, so every epoch drew a fresh set of pixels. Controlled by the
  new `fixed_points` flag (default `True`); `fixed_points=false` restores the exact
  v1 behaviour (legacy branch, lines 262–273).
- **Nested densities.** For `uniform` sampling, one permutation of each class's
  pixels is drawn per image and the first *n* are taken (line 104). The permutation
  does not depend on `points_per_class`, so the 1-point set ⊂ 5-point ⊂ 10-point ⊂
  20-point ⊂ 50-point. For `boundary`/`interior`, weighted sampling without
  replacement is drawn from the same seeded generator (lines 106–120).
- **Augmentation.** The fixed point mask (and the SLIC-expanded mask) are passed
  through the same flip/rotate transforms as the image and label
  (`self.aug(image=..., masks=[label, point_mask])`, line 254), so labelled points
  stay on the same pixels.
- **SLIC.** SLIC is computed once per image on the un-augmented image
  (`compute_slic_segments`, line 125), cached to disk under
  `<data_root>/_slic_cache/slic_<idx>_n<N>.npy` (`_slic_segments`, line 210), then
  the point mask is expanded over the superpixels that contain a labelled point
  (`expand_points_over_segments`, line 137) *before* augmentation. Previously SLIC
  was recomputed every epoch on the augmented image.

### `train.py`
- **Training seed.** `set_seed` (line 23) seeds `random`, `numpy`, `torch` and
  `torch.cuda`. DataLoader workers are seeded via `seed_worker` (line 31,
  `worker_init_fn`, line 169) and the train loader's shuffle uses a seeded
  `torch.Generator` (line 167). `cudnn.benchmark` is enabled (line 141) for the
  fixed 304×304 input size. The seed is recorded in each run's `config.yaml`.
- New CLI flags: `--seed`, `--point_seed`, `--fixed_points`/`--no_fixed_points`.

### `evaluate.py`
- `results/v2/summary.csv` gains four columns: `seed`, `point_seed`,
  `fixed_points`, `sampling` (kept before `timestamp`).

### `configs/base.yaml`, `configs/v2.yaml`
- New keys `fixed_points: true`, `point_seed: 0`, `seed: 0`. `configs/v2.yaml` is a
  copy of `base.yaml` with `results_dir: ./results/v2` so all v2 artefacts land under
  `results/v2/` and the v1 results are never overwritten.

## 2. Checks (`checks_fixed_points.py`)

```
=== Check 1: same image → same points (up to augmentation) ===
  no-aug, two loads identical: True  (points=40)
  aug, point counts over 5 loads: [40, 40, 40, 40, 40]  (constant: True)
  identical across fresh dataset instances: True

=== Check 2: nested densities for uniform sampling ===
  image 0:    nested=True  counts={1: 4, 5: 20, 10: 40, 20: 80, 50: 200}
  image 17:   nested=True  counts={1: 3, 5: 15, 10: 30, 20: 60, 50: 150}
  image 123:  nested=True  counts={1: 2, 5: 10, 10: 20, 20: 40, 50: 100}
  image 500:  nested=True  counts={1: 3, 5: 15, 10: 30, 20: 60, 50: 150}
  image 1000: nested=True  counts={1: 4, 5: 20, 10: 40, 20: 80, 50: 200}

=== Check 3: mean labelled pixels per image (train split, 200 images) ===
   1 points/class: mean= 2.75   min=1   max=5
  10 points/class: mean=27.26   min=10  max=50
  50 points/class: mean=135.49  min=50  max=250
```

- **Determinism:** the same image loaded twice (no augmentation) returns a
  bit-identical point mask; with augmentation the count is preserved and only the
  positions move with the flip/rotate. Seed-stability holds across fresh dataset
  instances too.
- **Nesting:** for every probed image the smaller-density masks are strict subsets
  of the larger ones.
- **Mean labelled pixels/image:** fewer than `6 × points_per_class` because not all
  six classes appear in every patch, and rare classes have fewer pixels than the
  requested count.

### Validation / test evaluation
Val and test mIoU are computed against the **full ground-truth label masks**, not
the point masks. In `train.py::validate` and `evaluate.py::evaluate`, predictions
are compared to `batch["label"]` (the dense GT), and the point mask is not read at
all in either function. The point mask only ever gates the *training* loss. So the
reported mIoU measures full-image segmentation quality, as intended.

## 3. Results

Seed budget run: the **minimum** set from the brief — the reference condition
(10 points, γ = 2, uniform, no SLIC) at three training seeds (0, 1, 2), all with
`point_seed = 0`; every other condition at seed 0. 13 unique runs (the
p10 / γ2 / uniform cells in the points, focal and placement groups all reuse
`ref_s0`, as the brief asks). All train / val / test mIoU is evaluated against the
full ground-truth masks.

### Reference noise band

3 seeds → test mIoU **0.5440 / 0.5591 / 0.5603**, mean **0.5545**, band
**[0.544, 0.560]** (spread ≈ 0.016). This is the band drawn in fig-2 and used to
judge whether any single-seed condition is meaningfully different.

### Test mIoU per condition (v2)

| Group | Condition | Test mIoU | Notes |
|---|---|---|---|
| Points per class | 1 | 0.4912 | |
| | 5 | 0.5510 | |
| | 10 (reference) | 0.5545 | mean of 3 seeds; min–max 0.544–0.560 |
| | 20 | 0.5694 | |
| | 50 | 0.5719 | |
| Focal γ | 0 | 0.5610 | |
| | 0.5 | 0.5645 | |
| | 1 | 0.5564 | |
| | 2 (reference) | 0.5545 | = reference |
| Label spreading | SLIC | **0.6031** | only condition clearly above the band |
| Point placement | Interior | 0.5320 | highest label purity, **lowest** mIoU |
| | Uniform (reference) | 0.5545 | |
| | Boundary | 0.5534 | |
| Upper bound | Full supervision | **0.6170** | dense masks, same model/schedule (`full_sup_s0`) |

### Full-supervision upper bound

Run `full_sup_s0`: identical setup (DeepLabV3+/ResNet-50, γ = 2, Adam, 50 epochs,
seed 0) but every pixel labelled — pfCE reduces to ordinary focal CE. Test mIoU
**0.6170**. This is the ceiling the sparse-point conditions are working toward, so
each condition can be read as a fraction of full supervision:

| Condition | Test mIoU | % of full-sup ceiling |
|---|---|---|
| Full supervision | 0.6170 | 100.0% |
| SLIC (10 pts) | 0.6031 | 97.8% |
| 50 points/class | 0.5719 | 92.7% |
| 20 points/class | 0.5694 | 92.3% |
| 10 points/class (reference) | 0.5545 | 89.9% |
| 5 points/class | 0.5510 | 89.3% |
| 1 point/class | 0.4912 | 79.6% |

Takeaways: **10 clicks per class recovers ~90% of dense supervision**, and **SLIC
label spreading closes almost the entire remaining gap (97.8%)** — the sparse-to-
dense gap is smaller than the annotation-effort difference would suggest. The
`full_sup_s0` summary row carries `points_per_class=10`/`sampling=uniform` from the
config, but the operative setting is `full_supervision=True` (recorded in its
`config.yaml`); the point-sampling fields are inert in that run.

### v1 vs v2 per condition

v1 = old `results/summary.csv` (single unseeded run, points re-drawn every epoch).
v2 = this rerun.

| Condition | v2 | v1 | Δ (v2 − v1) |
|---|---|---|---|
| Points 1 | 0.4912 | 0.5422 | −0.0509 |
| Points 5 | 0.5510 | 0.5638 | −0.0129 |
| Points 10 (ref) | 0.5545 | 0.5745 | −0.0200 |
| Points 20 | 0.5694 | 0.5722 | −0.0028 |
| Points 50 | 0.5719 | 0.6005 | −0.0286 |
| γ = 0 | 0.5610 | 0.5738 | −0.0128 |
| γ = 0.5 | 0.5645 | 0.5849 | −0.0205 |
| γ = 1 | 0.5564 | 0.5850 | −0.0286 |
| γ = 2 (ref) | 0.5545 | 0.5797 | −0.0253 |
| SLIC | 0.6031 | 0.6115 | −0.0084 |
| Interior | 0.5320 | 0.5636 | −0.0317 |
| Boundary | 0.5534 | 0.5792 | −0.0258 |

(v1 has no seed replicates, so the reference is compared against v1's
`baseline_p10_g2`.)

## 4. Anything surprising

- **Every condition dropped in v2 (−0.003 to −0.051).** This is the expected direct
  consequence of the fix. In v1 the points were re-drawn every epoch, so over 50
  epochs a "k points per class" model actually saw up to 50×k distinct labelled
  pixels per image. v2 fixes one set of points per image, which is the honest
  annotation budget, so the numbers are lower. The 1-point condition falls the most
  (−0.051): with only fixed points the model can no longer rely on the epoch-to-epoch
  churn that previously gave it extra effective supervision. This is the main reason
  the reruns were requested.
- **The 1-point drop breaks strict monotonicity at the very bottom but the density
  trend is otherwise intact:** 0.491 (1) → 0.551 (5) → 0.554 (10) → 0.569 (20) →
  0.572 (50). Gains flatten above 10 points, as before.
- **SLIC is the only condition clearly outside the noise band** (0.603 vs band top
  0.560), and it survives the fix nearly unchanged (−0.008). Expanding sparse points
  to whole superpixels is doing real work, not just exploiting re-drawn points.
- **"Cleaner labels were not better labels" holds.** Interior sampling has the
  highest local purity (0.974) yet the lowest mIoU (0.532), below both uniform
  (0.554) and boundary (0.553). Concentrating points in class interiors starves the
  model of the boundary pixels that decide most of the IoU.
- **Focal γ barely matters within the noise band.** γ ∈ {0, 0.5, 1, 2} all land in
  0.554–0.565, i.e. within ~1 band-width of each other; γ = 0.5 is nominally best but
  not distinguishable from the reference given the seed spread of ±0.016.
- The reference seed spread (0.016) is wider than several of the between-condition
  gaps in the focal-γ and placement groups, which is exactly why the noise band is
  the right lens for fig-2.
