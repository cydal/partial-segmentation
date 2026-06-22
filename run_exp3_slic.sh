#!/bin/bash
set -e

# Experiment 3 — SLIC label propagation
# Identical to exp1_p10_g2 (p=10, γ=2.0, 50 epochs) except point_mask is
# expanded to full SLIC superpixels (n_segments=200) before the loss.

echo "=== exp3_slic_p10_g2 — training ==="
python train.py \
    --run_name exp3_slic_p10_g2 \
    --points_per_class 10 \
    --focal_gamma 2.0 \
    --epochs 50 \
    --use_slic \
    --slic_n_segments 200

echo "=== exp3_slic_p10_g2 — evaluation ==="
python evaluate.py \
    --checkpoint results/runs/exp3_slic_p10_g2/best.pth \
    --run_name   exp3_slic_p10_g2

echo "Done — $(date)"
