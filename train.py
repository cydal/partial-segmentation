import argparse
import csv
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.potsdam_dataset import PotsdamPointDataset
from losses.partial_ce import PartialFocalCELoss
from models.segmentation_model import build_model
from utils.metrics import AverageMeter, compute_miou


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Seed torch, numpy and random for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """Give each DataLoader worker a deterministic (but distinct) seed."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",           default="configs/base.yaml")
    parser.add_argument("--points_per_class", type=int,   default=None)
    parser.add_argument("--focal_gamma",      type=float, default=None)
    parser.add_argument("--run_name",         type=str,   default="run")
    parser.add_argument("--epochs",           type=int,   default=None)
    parser.add_argument("--use_slic",         action="store_true", default=None)
    parser.add_argument("--slic_n_segments",  type=int,   default=None)
    parser.add_argument("--sampling",         type=str,   default=None,
                        choices=["uniform", "boundary", "interior"])
    parser.add_argument("--seed",             type=int,   default=None)
    parser.add_argument("--point_seed",       type=int,   default=None)
    parser.add_argument("--fixed_points",     dest="fixed_points", action="store_true",  default=None)
    parser.add_argument("--no_fixed_points",  dest="fixed_points", action="store_false")
    parser.add_argument("--full_supervision", action="store_true", default=None,
                        help="dense-mask upper bound: label every pixel (pfCE → CE)")
    return parser.parse_args()


def setup_run_dir(run_name: str, config: dict) -> str:
    run_dir = os.path.join(config["results_dir"], "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return run_dir


def append_history(run_dir: str, epoch: int, train_loss: float, val_miou: float):
    path = os.path.join(run_dir, "history.csv")
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "train_loss", "val_miou"])
        writer.writerow([epoch, round(train_loss, 6), round(val_miou, 6)])


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    meter = AverageMeter()
    for batch in tqdm(loader, desc="  train", leave=False):
        images     = batch["image"].to(device)
        labels     = batch["label"].to(device)
        point_mask = batch["point_mask"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels, point_mask)
        loss.backward()
        optimizer.step()

        meter.update(loss.item(), images.size(0))
    return meter.avg


@torch.no_grad()
def validate(model, loader, device, num_classes):
    model.eval()
    all_preds, all_targets = [], []
    for batch in tqdm(loader, desc="  val  ", leave=False):
        images  = batch["image"].to(device)
        targets = batch["label"].to(device)

        logits = model(images)
        preds  = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

    preds_cat   = torch.cat(all_preds,   dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    return compute_miou(preds_cat, targets_cat, num_classes)


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # CLI overrides
    if args.points_per_class is not None:
        cfg["points_per_class"] = args.points_per_class
    if args.focal_gamma is not None:
        cfg["focal_gamma"] = args.focal_gamma
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.use_slic:
        cfg["use_slic"] = True
    if args.slic_n_segments is not None:
        cfg["slic_n_segments"] = args.slic_n_segments
    if args.sampling is not None:
        cfg["sampling"] = args.sampling
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.point_seed is not None:
        cfg["point_seed"] = args.point_seed
    if args.fixed_points is not None:
        cfg["fixed_points"] = args.fixed_points
    if args.full_supervision:
        cfg["full_supervision"] = True

    # Defaults for keys that may be absent in older configs
    cfg.setdefault("seed", 0)
    cfg.setdefault("point_seed", 0)
    cfg.setdefault("fixed_points", True)
    cfg.setdefault("full_supervision", False)

    seed = cfg["seed"]
    set_seed(seed)
    torch.backends.cudnn.benchmark = True   # autotune conv kernels (fixed input size)

    os.makedirs(cfg["results_dir"], exist_ok=True)
    run_dir = setup_run_dir(args.run_name, cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Run directory: {run_dir}")
    print(f"seed={seed}  point_seed={cfg['point_seed']}  fixed_points={cfg['fixed_points']}  "
          f"full_supervision={cfg['full_supervision']}  "
          f"sampling={cfg.get('sampling','uniform')}  use_slic={cfg.get('use_slic', False)}")

    ds_kwargs = dict(
        use_slic=cfg.get("use_slic", False),
        slic_n_segments=cfg.get("slic_n_segments", 200),
        sampling=cfg.get("sampling", "uniform"),
        fixed_points=cfg["fixed_points"],
        point_seed=cfg["point_seed"],
        full_supervision=cfg["full_supervision"],
    )
    train_ds = PotsdamPointDataset(cfg["data_root"], "train",
                                   points_per_class=cfg["points_per_class"],
                                   augment=True, **ds_kwargs)
    val_ds   = PotsdamPointDataset(cfg["data_root"], "val",
                                   points_per_class=cfg["points_per_class"],
                                   augment=False, **ds_kwargs)

    g = torch.Generator()
    g.manual_seed(seed)
    loader_kwargs = dict(num_workers=cfg["num_workers"], pin_memory=True,
                         worker_init_fn=seed_worker, persistent_workers=cfg["num_workers"] > 0)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, generator=g, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, **loader_kwargs)

    model = build_model(cfg["num_classes"], cfg["backbone"], cfg["pretrained"]).to(device)
    criterion = PartialFocalCELoss(gamma=cfg["focal_gamma"], num_classes=cfg["num_classes"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best_miou = 0.0
    ckpt_path = os.path.join(run_dir, "best.pth")

    for epoch in range(1, cfg["epochs"] + 1):
        print(f"Epoch {epoch}/{cfg['epochs']}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_miou   = validate(model, val_loader, device, cfg["num_classes"])

        print(f"  train_loss={train_loss:.4f}  val_mIoU={val_miou:.4f}")
        append_history(run_dir, epoch, train_loss, val_miou)

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_miou": val_miou,
                "config": cfg,
            }, ckpt_path)
            print(f"  -> saved best checkpoint (mIoU={best_miou:.4f})")

    print(f"Training complete. Best val mIoU: {best_miou:.4f}")
    print(f"Artifacts in: {run_dir}")


if __name__ == "__main__":
    main()
