import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialFocalCELoss(nn.Module):
    def __init__(self, gamma: float = 2.0, num_classes: int = 6):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
        """
        logits:     [B, C, H, W] float32
        targets:    [B, H, W]    int64
        point_mask: [B, H, W]    float32 binary
        """
        total_points = point_mask.sum()
        if total_points == 0:
            return torch.tensor(0.0, requires_grad=True, device=logits.device)

        # Softmax probabilities: [B, C, H, W]
        probs = F.softmax(logits, dim=1)

        # Gather p_t for the target class: [B, H, W]
        targets_clamped = targets.clamp(0, self.num_classes - 1)
        p_t = probs.gather(dim=1, index=targets_clamped.unsqueeze(1)).squeeze(1)

        # Focal weight and per-pixel loss
        focal_weight = (1.0 - p_t) ** self.gamma
        pixel_loss = -focal_weight * torch.log(p_t + 1e-8)

        # Mask to labeled points only and normalise
        loss = (pixel_loss * point_mask).sum() / total_points.clamp(min=1)
        return loss
