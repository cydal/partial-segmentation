#!/usr/bin/env bash
# Part 1 reruns → results/v2/runs/. Minimum seed set:
#   - reference (10 pts, gamma=2, uniform, no SLIC) at seeds 0,1,2  (noise band)
#   - every other condition at seed 0
# Idempotent: a run with an existing test_metrics.json is skipped, so this is
# safe to re-invoke after an interruption.
set -euo pipefail
export TQDM_DISABLE=1
cd "$(dirname "$0")"

CFG=configs/v2.yaml
LOG=results/v2/run_v2.log
mkdir -p results/v2
: > "$LOG"

run () {
  local name="$1"; shift
  if [[ -f "results/v2/runs/${name}/test_metrics.json" ]]; then
    echo "[skip] ${name} (already done)" | tee -a "$LOG"
    return
  fi
  echo "=== [train] ${name} :: $* ===" | tee -a "$LOG"
  python train.py --config "$CFG" --run_name "$name" "$@" 2>&1 | tee -a "$LOG"
  echo "=== [eval]  ${name} ===" | tee -a "$LOG"
  python evaluate.py --checkpoint "results/v2/runs/${name}/best.pth" 2>&1 | tee -a "$LOG"
}

# --- Reference condition, three training seeds (point_seed fixed at 0) ---
run ref_s0 --points_per_class 10 --focal_gamma 2.0 --sampling uniform --seed 0 --point_seed 0
run ref_s1 --points_per_class 10 --focal_gamma 2.0 --sampling uniform --seed 1 --point_seed 0
run ref_s2 --points_per_class 10 --focal_gamma 2.0 --sampling uniform --seed 2 --point_seed 0

# --- Points per class (gamma=2, uniform), seed 0; p10 = reference (ref_s0) ---
run p1_g2_s0  --points_per_class 1  --focal_gamma 2.0 --sampling uniform --seed 0 --point_seed 0
run p5_g2_s0  --points_per_class 5  --focal_gamma 2.0 --sampling uniform --seed 0 --point_seed 0
run p20_g2_s0 --points_per_class 20 --focal_gamma 2.0 --sampling uniform --seed 0 --point_seed 0
run p50_g2_s0 --points_per_class 50 --focal_gamma 2.0 --sampling uniform --seed 0 --point_seed 0

# --- Focal gamma (10 pts, uniform), seed 0; gamma=2 = reference (ref_s0) ---
run g0_s0  --points_per_class 10 --focal_gamma 0.0 --sampling uniform --seed 0 --point_seed 0
run g05_s0 --points_per_class 10 --focal_gamma 0.5 --sampling uniform --seed 0 --point_seed 0
run g1_s0  --points_per_class 10 --focal_gamma 1.0 --sampling uniform --seed 0 --point_seed 0

# --- Label spreading: SLIC (10 pts, gamma=2), seed 0 ---
run slic_s0 --points_per_class 10 --focal_gamma 2.0 --sampling uniform --seed 0 --point_seed 0 --use_slic

# --- Point placement (10 pts, gamma=2), seed 0; uniform = reference (ref_s0) ---
run interior_s0 --points_per_class 10 --focal_gamma 2.0 --sampling interior --seed 0 --point_seed 0
run boundary_s0 --points_per_class 10 --focal_gamma 2.0 --sampling boundary --seed 0 --point_seed 0

echo "=== all v2 runs complete ===" | tee -a "$LOG"
