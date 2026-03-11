import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryDiceLoss(nn.Module):
    """
    Binary Dice Loss with proper ignore_index handling.
    Assumes inputs are probabilities in [0, 1].
    """

    def __init__(self, smooth=1e-5, ignore_index=-100, eps=1e-8):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, probs, targets):
        """
        Args:
            probs:   (B, 1, D, H, W)  -- sigmoid already applied
            targets: (B, D, H, W) or (B, 1, D, H, W)
                     values ∈ {0, 1, ignore_index}
        """
        # Shape handling
        if targets.dim() == 5:
            targets = targets.squeeze(1)

        probs = probs.squeeze(1)
        targets = targets.float()

        # Mask out ignored pixels
        mask = targets != self.ignore_index

        if mask.sum() == 0:
            # keep graph alive, avoid NaNs
            return (probs ** 2).mean() * self.eps

        # Select valid pixels only
        probs_valid = probs[mask]
        targets_valid = targets[mask]

        # Clamp probabilities for numerical stability
        probs_valid = probs_valid.clamp(self.eps, 1.0 - self.eps)

        # Dice computation
        intersection = (probs_valid * targets_valid).sum()
        union = probs_valid.sum() + targets_valid.sum()

        dice = (2.0 * intersection + self.smooth) / (
            union + self.smooth + self.eps
        )

        dice = torch.clamp(dice, 0.0, 1.0)

        return 1.0 - dice
