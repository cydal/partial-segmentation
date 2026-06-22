#!/bin/bash
set -e

echo "Starting evaluations — $(date)"

RUNS=(
  baseline_p10_g2
  exp1_p1_g2
  exp1_p5_g2
  exp1_p10_g2
  exp1_p20_g2
  exp1_p50_g2
  exp2_p10_g0
  exp2_p10_g05
  exp2_p10_g1
  exp2_p10_g2
)

for run in "${RUNS[@]}"; do
  echo "--- Evaluating $run ---"
  python evaluate.py \
    --run_name "$run" \
    --checkpoint "results/runs/${run}/best.pth"
done

echo "All evaluations complete — $(date)"
echo "summary.csv rows: $(wc -l < results/summary.csv)"
