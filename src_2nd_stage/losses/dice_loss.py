import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Dice Loss that properly handles ignore_index."""

    def __init__(self, smooth=1e-5, ignore_index=2, eps=1e-8):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.eps = eps  # Small bias for numerical stability

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C, D, H, W) - logits (should already be clamped)
            targets: (B, D, H, W) - class indices
        """
        # Ensure targets is 4D (B, D, H, W)
        if targets.dim() == 5 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        # Apply softmax to get probabilities
        probs = F.softmax(inputs, dim=1)  # (B, C, D, H, W)

        # Get probability for class 1 (positive class)
        prob_positive = probs[:, 1]  # (B, D, H, W)

        # Convert targets to binary (1 for class 1, 0 for class 0)
        # Note: label 2 (ignore_index) will be filtered out by mask below
        targets_binary = (targets == 1).float()

        # Create mask to exclude ignore_index pixels from loss
        mask = (targets != self.ignore_index)  # (B, D, H, W)

        # Check if all pixels are ignored
        if mask.sum() == 0:
            # Return small bias loss to maintain gradients (prevents NaN)
            return (inputs ** 2).mean() * self.eps

        # Flatten
        prob_flat = prob_positive.reshape(-1)
        target_flat = targets_binary.reshape(-1)
        mask_flat = mask.reshape(-1)

        # Filter out ignored pixels
        prob_valid = prob_flat[mask_flat]
        target_valid = target_flat[mask_flat]

        # Calculate intersection and union on valid pixels only
        intersection = (prob_valid * target_valid).sum()
        union = prob_valid.sum() + target_valid.sum()

        # Dice coefficient with numerical stability
        # Add eps to both numerator and denominator to prevent division by zero
        dice = (2. * intersection + self.smooth) / (union + self.smooth + self.eps)
        dice = torch.clamp(dice, min=0.0, max=1.0)
        return 1 - dice

