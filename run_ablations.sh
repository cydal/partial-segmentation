#!/bin/bash
set -e

echo "Starting sequential ablation runs — $(date)"

# ── Experiment 1: annotation density ──────────────────────────────────────────
echo "=== [1/9] exp1_p1_g2 ==="
python train.py --run_name exp1_p1_g2   --points_per_class 1  --focal_gamma 2.0 --epochs 50
echo "=== [2/9] exp1_p5_g2 ==="
python train.py --run_name exp1_p5_g2   --points_per_class 5  --focal_gamma 2.0 --epochs 50
echo "=== [3/9] exp1_p10_g2 ==="
python train.py --run_name exp1_p10_g2  --points_per_class 10 --focal_gamma 2.0 --epochs 50
echo "=== [4/9] exp1_p20_g2 ==="
python train.py --run_name exp1_p20_g2  --points_per_class 20 --focal_gamma 2.0 --epochs 50
echo "=== [5/9] exp1_p50_g2 ==="
python train.py --run_name exp1_p50_g2  --points_per_class 50 --focal_gamma 2.0 --epochs 50

# ── Experiment 2: focal gamma ──────────────────────────────────────────────────
echo "=== [6/9] exp2_p10_g0 ==="
python train.py --run_name exp2_p10_g0  --points_per_class 10 --focal_gamma 0.0 --epochs 50
echo "=== [7/9] exp2_p10_g05 ==="
python train.py --run_name exp2_p10_g05 --points_per_class 10 --focal_gamma 0.5 --epochs 50
echo "=== [8/9] exp2_p10_g1 ==="
python train.py --run_name exp2_p10_g1  --points_per_class 10 --focal_gamma 1.0 --epochs 50
echo "=== [9/9] exp2_p10_g2 ==="
python train.py --run_name exp2_p10_g2  --points_per_class 10 --focal_gamma 2.0 --epochs 50

echo "All 9 training runs complete — $(date)"
