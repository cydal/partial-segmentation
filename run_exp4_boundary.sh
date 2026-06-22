#!/bin/bash
set -e

# Experiment 4 — Boundary vs interior vs uniform point sampling
# All three runs: p=10, γ=2.0, 50 epochs, same as exp1_p10_g2 except sampling strategy.
# exp1_p10_g2 (uniform) already exists and serves as the control.

echo "=== exp4_boundary_p10_g2 — training ==="
python train.py \
    --run_name exp4_boundary_p10_g2 \
    --points_per_class 10 \
    --focal_gamma 2.0 \
    --epochs 50 \
    --sampling boundary

echo "=== exp4_boundary_p10_g2 — evaluation ==="
python evaluate.py \
    --checkpoint results/runs/exp4_boundary_p10_g2/best.pth \
    --run_name   exp4_boundary_p10_g2

echo "=== exp4_interior_p10_g2 — training ==="
python train.py \
    --run_name exp4_interior_p10_g2 \
    --points_per_class 10 \
    --focal_gamma 2.0 \
    --epochs 50 \
    --sampling interior

echo "=== exp4_interior_p10_g2 — evaluation ==="
python evaluate.py \
    --checkpoint results/runs/exp4_interior_p10_g2/best.pth \
    --run_name   exp4_interior_p10_g2

echo "Done — $(date)"
