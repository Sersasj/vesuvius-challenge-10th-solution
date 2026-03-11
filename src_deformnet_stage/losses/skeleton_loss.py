import torch
import torch.nn as nn
import torch.nn.functional as F


class BinarySkeletonRecallLoss(nn.Module):
    """
    Encourages high recall on thin GT skeleton / centerline voxels.
    Binary-only, probability-based, ignore_index aware.
    """

    def __init__(self, ignore_index=-100, smooth=1e-5):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(
        self,
        probs: torch.Tensor,
        target_skeleton: torch.Tensor,
        original_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            probs:            (B, 1, D, H, W)  -- probabilities in [0,1]
            target_skeleton: (B, D, H, W) or (B, 1, D, H, W) -- binary
            original_labels: (B, D, H, W) -- used for ignore masking
        """

        # Shape handling
        probs = probs.squeeze(1)

        if target_skeleton.dim() == 5:
            target_skeleton = target_skeleton.squeeze(1)

        target_skeleton = target_skeleton.float()

        # Mask ignore regions
        mask = (original_labels != self.ignore_index).float()

        # Skeleton voxels that matter
        active_skeleton = target_skeleton * mask

        # If no skeleton present → no penalty
        denom = active_skeleton.sum(dim=(1, 2, 3))

        valid_batch = denom > 0
        if valid_batch.sum() == 0:
            return torch.tensor(
                0.0, device=probs.device, requires_grad=True
            )

        # Recall on skeleton voxels
        numerator = (probs * active_skeleton).sum(dim=(1, 2, 3))

        recall = (numerator + self.smooth) / (denom + self.smooth)

        # Average only over batches with skeleton
        recall = recall[valid_batch]

        return 1.0 - recall.mean()

