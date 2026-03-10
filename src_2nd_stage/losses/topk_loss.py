import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKCrossEntropyLoss(nn.Module):
    """
    Top-k Cross Entropy Loss (Hard Example Mining).
    
    Computes the Cross Entropy loss and backpropagates only for the top k% 
    pixels with the highest loss.
    """
    def __init__(self, top_k_percent=1.0, ignore_index=-100):
        super(TopKCrossEntropyLoss, self).__init__()
        self.top_k_percent = top_k_percent
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits, target):
        """
        Args:
            logits: (B, C, D, H, W)
            target: (B, D, H, W)
        """
        # Ensure target is 4D (B, D, H, W) and Long for CrossEntropy
        if target.dim() == 5 and target.shape[1] == 1:
            target = target.squeeze(1)
        target = target.long()

        # 1. Compute pixel-wise CE loss
        # Output shape: (B, D, H, W)
        pixel_losses = self.ce(logits, target)
        
        if self.top_k_percent >= 1.0:
            return pixel_losses.mean()
            
        # 2. Flatten losses
        pixel_losses = pixel_losses.view(-1)
        target_flat = target.view(-1)
        
        # 3. Filter out ignore_index pixels (if any) so they don't count towards the top k
        # The CE loss already handles ignore_index by setting loss to 0 for those pixels,
        # but we want to exclude them from the sorting/counting to be precise.
        # However, nn.CrossEntropyLoss(reduction='none') returns 0 for ignored targets.
        # So we can just filter non-zero losses or filter by target.
        valid_mask = target_flat != self.ignore_index
        valid_losses = pixel_losses[valid_mask]
        
        if valid_losses.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        # 4. Select top k%
        num_valid = valid_losses.numel()
        k = int(self.top_k_percent * num_valid)
        k = max(1, k) # Ensure at least 1 pixel is selected
        
        topk_losses, _ = torch.topk(valid_losses, k)
        
        return topk_losses.mean()

