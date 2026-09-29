# Mismatches

Every number in the figures was re-derived from the source CSVs by
`make_figures.py` (mIoU from `results/v2/summary.csv`; purity from
`results/boundary_analysis/sampling_data.csv` and
`results/slic_analysis/purity_data.csv`). Nothing was hand-typed or adjusted to
match the article.

## Checks against the brief

- **fig-4 local label purity.** The brief states 0.974 (interior), 0.918
  (uniform), 0.848 (boundary). Re-derived means of `local_purity` from
  `sampling_data.csv`: interior 0.974418, uniform 0.918112, boundary 0.848148 →
  **0.974 / 0.918 / 0.848**. **Match.**

- **fig-2 / fig-4 test mIoU.** All read live from `results/v2/summary.csv`. The
  reference dot is the mean of the three seed runs (0.5545), band = min–max
  [0.544, 0.560]. These are the v2 reruns, so they differ from the article's
  current (v1) numbers by design — see `results/v2/NOTES.md` §4. That is a v1→v2
  change, not a figure/CSV mismatch.

## Oddities worth noting (not errors)

- **The article's mIoU numbers will change.** The article currently reflects v1.
  Every v2 mIoU is lower (−0.003 to −0.051) because v1 re-drew points every epoch.
  The brief says the article's numbers will be updated after this work, so this is
  expected, not a mismatch. The qualitative story is unchanged: density helps and
  saturates, SLIC is the clear winner and the only condition outside the noise
  band, and interior (highest purity) is the worst placement.

- **1-point condition (0.491) sits well below the rest.** With genuinely fixed
  points this is the largest v1→v2 drop (−0.051). It is a real effect of the fix,
  not a plotting artefact.

- **fig-1 / fig-3 use test patch `Image_2071`** (five classes present, incl. car),
  with fixed nested sampling at `point_seed = 0`. `sampling_data.csv` /
  `purity_data.csv` were computed on *training* images with their own seeded
  sampler, so the specific superpixels shown in fig-3 are illustrative of the same
  procedure, not a row in those CSVs. The aggregate purity trend in fig-4 is what
  those CSVs quantify.

## fig-5: sparse vs full supervision

- All mIoU read live from `results/v2/summary.csv`. Ceiling = `full_sup_s0`
  test mIoU **0.617**. Percentages are `mean / ceiling`: 1 pt 80%, 5 pt 89%,
  10 pt (reference mean) 90%, 20 pt 92%, 50 pt 93%, SLIC 98%. Re-derived from the
  CSV — **match**.
- The `full_sup_s0` summary row shows `points_per_class=10` / `sampling=uniform`
  (inert config leftovers); the operative flag is `full_supervision=True`, recorded
  in its `config.yaml`. Not a mismatch, just a labelling note.

## Summary

No CSV-vs-figure mismatches found.
