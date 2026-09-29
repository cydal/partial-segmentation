import argparse
import csv
import json
import os
from datetime import datetime

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.potsdam_dataset import PotsdamPointDataset
from models.segmentation_model import build_model
from utils.metrics import compute_miou

CLASS_NAMES = [
    "Impervious surfaces",
    "Building",
    "Low vegetation",
    "Tree",
    "Car",
    "Clutter/background",
]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default="configs/base.yaml")
    parser.add_argument("--run_name",   default=None,
                        help="Override run name (default: inferred from checkpoint path)")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    all_preds, all_targets = [], []
    for batch in tqdm(loader, desc="evaluating"):
        images  = batch["image"].to(device)
        targets = batch["label"]
        logits  = model(images)
        preds   = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_targets.append(targets)

    preds_cat   = torch.cat(all_preds,   dim=0)
    targets_cat = torch.cat(all_targets, dim=0)

    per_class_iou = []
    for cls in range(num_classes):
        pred_cls   = preds_cat == cls
        target_cls = targets_cat == cls
        if not target_cls.any():
            per_class_iou.append(None)
            continue
        intersection = (pred_cls & target_cls).sum().item()
        union        = (pred_cls | target_cls).sum().item()
        per_class_iou.append(intersection / union if union > 0 else 0.0)

    valid = [v for v in per_class_iou if v is not None]
    miou  = sum(valid) / len(valid) if valid else 0.0
    return per_class_iou, miou


def save_test_metrics(run_dir: str, run_name: str, config: dict,
                      per_class_iou: list, miou: float, best_val_miou: float):
    safe_iou = [v if v is not None else 0.0 for v in per_class_iou]

    metrics = {
        "run_name":    run_name,
        "test_miou":   round(miou, 6),
        "best_val_miou": round(best_val_miou, 6),
        "per_class_iou": {
            "impervious":     round(safe_iou[0], 6),
            "building":       round(safe_iou[1], 6),
            "low_vegetation": round(safe_iou[2], 6),
            "tree":           round(safe_iou[3], 6),
            "car":            round(safe_iou[4], 6),
            "clutter":        round(safe_iou[5], 6),
        },
        "points_per_class": config["points_per_class"],
        "focal_gamma":      config["focal_gamma"],
        "epochs_trained":   config["epochs"],
        "batch_size":       config["batch_size"],
        "timestamp":        datetime.utcnow().isoformat(),
    }
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    summary_path = os.path.join(config["results_dir"], "summary.csv")
    fieldnames = [
        "run_name", "points_per_class", "focal_gamma",
        "best_val_miou", "test_miou",
        "iou_impervious", "iou_building", "iou_low_vegetation",
        "iou_tree", "iou_car", "iou_clutter",
        "epochs_trained", "batch_size",
        "seed", "point_seed", "fixed_points", "sampling",
        "timestamp",
    ]
    row = {
        "run_name":           run_name,
        "points_per_class":   config["points_per_class"],
        "focal_gamma":        config["focal_gamma"],
        "best_val_miou":      round(best_val_miou, 6),
        "test_miou":          round(miou, 6),
        "iou_impervious":     round(safe_iou[0], 6),
        "iou_building":       round(safe_iou[1], 6),
        "iou_low_vegetation": round(safe_iou[2], 6),
        "iou_tree":           round(safe_iou[3], 6),
        "iou_car":            round(safe_iou[4], 6),
        "iou_clutter":        round(safe_iou[5], 6),
        "epochs_trained":     config["epochs"],
        "batch_size":         config["batch_size"],
        "seed":               config.get("seed", 0),
        "point_seed":         config.get("point_seed", 0),
        "fixed_points":       config.get("fixed_points", True),
        "sampling":           config.get("sampling", "uniform"),
        "timestamp":          datetime.utcnow().isoformat(),
    }
    existing = []
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            existing = list(csv.DictReader(f))
    existing = [r for r in existing if r["run_name"] != run_name]
    existing.append(row)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg  = ckpt.get("config", load_config(args.config))

    # Infer run_name and run_dir from checkpoint path if not provided
    run_name = args.run_name or os.path.basename(os.path.dirname(args.checkpoint))
    run_dir  = os.path.dirname(args.checkpoint)

    model = build_model(cfg["num_classes"], cfg["backbone"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_ds = PotsdamPointDataset(cfg["data_root"], "test",
                                  points_per_class=cfg["points_per_class"], augment=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"],
                             shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)

    per_class_iou, miou = evaluate(model, test_loader, device, cfg["num_classes"])
    best_val_miou = ckpt.get("val_miou", 0.0)

    print(f"\nTest results  (run: {run_name})")
    print("-" * 40)
    for cls, iou in enumerate(per_class_iou):
        tag = f"{iou:.4f}" if iou is not None else "N/A"
        print(f"  {CLASS_NAMES[cls]:<24} {tag}")
    print("-" * 40)
    print(f"  {'mIoU':<24} {miou:.4f}")

    save_test_metrics(run_dir, run_name, cfg, per_class_iou, miou, best_val_miou)
    print(f"\nSaved test_metrics.json → {run_dir}")
    print(f"Upserted row  → {cfg['results_dir']}/summary.csv")


if __name__ == "__main__":
    main()
