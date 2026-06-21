import torch
import yaml
from data.potsdam_dataset import PotsdamPointDataset
from models.segmentation_model import build_model
from losses.partial_ce import PartialFocalCELoss
from torch.utils.data import DataLoader

config = yaml.safe_load(open("configs/base.yaml"))
config["augment"] = False

model = build_model().cuda()
criterion = PartialFocalCELoss(gamma=2.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

batch_size = 8
while True:
    try:
        print(f"Trying batch_size={batch_size} ...", end=" ", flush=True)
        dataset = PotsdamPointDataset(
            config["data_root"], split="train",
            points_per_class=10, augment=False
        )
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
        batch = next(iter(loader))
        images = batch["image"].cuda()
        labels = batch["label"].cuda()
        masks  = batch["point_mask"].cuda()

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels, masks)
        loss.backward()
        optimizer.step()

        print(f"OK — VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB "
              f"/ {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        torch.cuda.empty_cache()
        batch_size *= 2

    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            best = batch_size // 2
            print(f"OOM. Safe batch_size = {best}")
            break
        else:
            raise
