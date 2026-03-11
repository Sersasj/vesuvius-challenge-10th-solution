import torch
import torch.nn as nn
import torch.nn.functional as F

class SkeletonRecallLoss(nn.Module):
    """
    Ensures the model 'recalls' the thin centerline/skeleton of the sheet.    
    """
    def __init__(self, ignore_index=2, smooth=1e-5):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, target_skeleton, original_labels):
        """
        Args:
            logits: (B, C, D, H, W) - model predictions (logits)
            target_skeleton: (B, D, H, W) - the thin 1-voxel GT skeleton (binary)
            original_labels: (B, D, H, W) - used for masking ignore regions
        """
        # Get probability for the positive class (class 1)
        # logits shape: (B, 2, D, H, W) or (B, 1, D, H, W)
        if logits.shape[1] == 2:
            probs = F.softmax(logits, dim=1)[:, 1]
        else:
            probs = torch.sigmoid(logits).squeeze(1)
            
        # Create mask to exclude ignore_index pixels from loss
        mask = (original_labels != self.ignore_index).float()
        
        # We only care about the recall on the skeleton voxels
        # skeleton is 1-voxel thin, so we want the model to be high there
        # We multiply by mask to ensure we don't penalize in ignore regions
        active_skeleton = target_skeleton * mask
        
        # Weighted recall: focus only on skeleton points
        numerator = torch.sum(probs * active_skeleton, dim=(1, 2, 3))
        denominator = torch.sum(active_skeleton, dim=(1, 2, 3))
        
        # Avoid division by zero if a patch has no skeleton
        recall = (numerator + self.smooth) / (denominator + self.smooth)
        
        return 1.0 - recall.mean()

