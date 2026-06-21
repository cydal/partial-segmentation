import argparse

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


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_cfg = ckpt.get("cfg", cfg)

    model = build_model(saved_cfg["num_classes"], saved_cfg["backbone"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])

    test_ds = PotsdamPointDataset(saved_cfg["data_root"], "test",
                                  points_per_class=saved_cfg["points_per_class"], augment=False)
    test_loader = DataLoader(test_ds, batch_size=saved_cfg["batch_size"],
                             shuffle=False, num_workers=saved_cfg["num_workers"], pin_memory=True)

    per_class_iou, miou = evaluate(model, test_loader, device, saved_cfg["num_classes"])

    print(f"\nTest results (checkpoint: {args.checkpoint})")
    print("-" * 40)
    for cls, iou in enumerate(per_class_iou):
        if iou is None:
            print(f"  {CLASS_NAMES[cls]:<24} N/A")
        else:
            print(f"  {CLASS_NAMES[cls]:<24} {iou:.4f}")
    print("-" * 40)
    print(f"  {'mIoU':<24} {miou:.4f}")


if __name__ == "__main__":
    main()
