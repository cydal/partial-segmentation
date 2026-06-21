import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = 0.0
        self._count = 0

    def update(self, val: float, n: int = 1):
        self._sum += val * n
        self._count += n

    @property
    def avg(self) -> float:
        return self._sum / self._count if self._count > 0 else 0.0


def compute_miou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 6) -> float:
    """
    preds:   [B, H, W] int64
    targets: [B, H, W] int64
    Returns mean IoU over classes that appear in targets.
    """
    ious = []
    for cls in range(num_classes):
        pred_cls   = preds == cls
        target_cls = targets == cls

        if not target_cls.any():
            continue

        intersection = (pred_cls & target_cls).sum().item()
        union        = (pred_cls | target_cls).sum().item()
        ious.append(intersection / union if union > 0 else 0.0)

    return float(sum(ious) / len(ious)) if ious else 0.0
