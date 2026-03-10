"""PyTorch Lightning module for 3D segmentation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from monai.inferers import sliding_window_inference
import numpy as np
from pathlib import Path

from .residual_unet import create_residual_unet
from ..losses import DiceLoss, SurfaceDiceLoss, SkeletonRecallLoss
from ..utils.ema import EMA


def strip_metatensor(t):
    """Strip MONAI MetaTensor metadata if present."""
    return t.as_tensor() if hasattr(t, "as_tensor") else t


def create_model(in_channels, out_channels, channels, strides,
                 use_deep_supervision, n_blocks_per_stage=None,
                 ):
    """Factory function to create the segmentation model."""
    return create_residual_unet(in_channels, out_channels, channels, strides,
                                    n_blocks_per_stage=n_blocks_per_stage,
                                    deep_supervision=use_deep_supervision)


class SegmentationModule(pl.LightningModule):
    """Lightning module for 3D surface segmentation."""

    def __init__(
        self,
        in_channels=3, out_channels=2,  # Default: 1 image + 2 OOF channels
        channels=(32, 64, 128, 128), strides=(1, 2, 2, 2),
        lr=1e-3, weight_decay=1e-4, ignore_index=2,
        # (CE, Dice, SurfaceDice, SkeletonRecall)
        loss_weights=(0.3, 0.3, 0.2, 0.2),
        surface_dice_iterations=5,
        use_deep_supervision=False,
        deep_supervision_weights=None,
        n_blocks_per_stage=(1, 3, 4, 6, 6, 6),
        debug_mode=False, compute_leaderboard_metrics=False,
        leaderboard_max_val_samples=0, leaderboard_surface_tolerance=2.0,
        ce_top_k=1.0,
        use_ema=False, ema_decay=0.999, ema_warmup=0,
        patch_size=128,
        use_iterative_training=False, iterative_training_prob=0.5, iterative_training_threshold=0.3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.debug_samples = []
        self.val_debug_samples = []
        self._val_cache = []

        # Model - always uses residual UNet
        self.model = create_model(
            in_channels, out_channels, channels, strides,
            use_deep_supervision, n_blocks_per_stage,
        )

        # EMA (optional)
        self.ema = None
        if use_ema:
            self.ema = EMA(self.model, decay=ema_decay, warmup=ema_warmup)

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss(smooth=1e-5, ignore_index=ignore_index)
        self.surface_dice_loss = SurfaceDiceLoss(
            ignore_label=ignore_index, soft_skel_iterations=surface_dice_iterations, smooth=1.0
        )
        self.skeleton_loss = SkeletonRecallLoss(ignore_index=ignore_index)

    def forward(self, x):
        # Use EMA for inference, original model for training
        model = self.ema.module if (self.ema is not None and not self.training) else self.model
        out = model(x)
        if not self.training and isinstance(out, (list, tuple)):
            return out[0]
        return out

    def _get_loss_weights(self):
        """Get loss weights with backward compatibility for multiple formats."""
        weights = self.hparams.loss_weights

        # Backwards compat:
        # - (ce, dice) => no surface losses
        # - (ce, dice, surface_dice) => SurfaceDice only
        # - (ce, dice, surface_dice, skeleton_recall) => All
        if len(weights) == 2:
            return weights[0], weights[1], 0.0, 0.0
        if len(weights) == 3:
            return weights[0], weights[1], weights[2], 0.0
        if len(weights) == 4:
            return weights[0], weights[1], weights[2], weights[3]
        return weights[0], weights[1], weights[2], 0.0

    def _downsample_labels(self, labels, target_size):
        """Downsample labels to target size using nearest neighbor interpolation.

        Args:
            labels: (B, D, H, W) - class indices or binary masks
            target_size: Tuple of (D, H, W)

        Returns:
            Downsampled labels of shape (B, *target_size)
        """
        if labels.shape[1:] == target_size:
            return labels

        # Ensure input is 5D for interpolation (B, 1, D, H, W)
        if labels.dim() == 4:
            labels_5d = labels.unsqueeze(1).float()
        else:
            # If already 5D, assume channel is at index 1
            labels_5d = labels.float()

        # Downsample using nearest neighbor (preserves class indices)
        labels_downsampled = F.interpolate(
            labels_5d,
            size=target_size,
            mode='nearest'
        )

        # Return as 4D (B, D, H, W) for CrossEntropy
        return labels_downsampled.squeeze(1)

    def _prepare_tensors(self, logits, labels, skeletons=None):
        """Prepare tensors for loss computation."""
        logits = strip_metatensor(logits)
        labels = strip_metatensor(labels)
        if labels.dim() == 5 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        if skeletons is not None:
            skeletons = strip_metatensor(skeletons)
        return logits, labels, skeletons

    def _safe_loss(self, loss_val, logits):
        """Replace NaN losses with small regularization term."""
        return (logits ** 2).mean() * 1e-8 if torch.isnan(loss_val) else loss_val

    def _compute_single_scale_loss(self, logits, labels, skeletons):
        """Compute loss at a single scale."""
        logits, labels, skeletons = self._prepare_tensors(
            logits, labels, skeletons)
        ce_w, dice_w, surf_w, skel_w = self._get_loss_weights()

        loss_ce = self._safe_loss(self.ce_loss(logits, labels.long()), logits)
        loss_dice = self._safe_loss(
            self.dice_loss(logits, labels.long()), logits)

        # Surface Dice (only if weight > 0)
        if surf_w > 0:
            logits_binary = logits[:, 1:2] - logits[:, 0:1]
            loss_surf_dice = self.surface_dice_loss(
                logits_binary, labels.unsqueeze(1)).mean()
            loss_surf_dice = self._safe_loss(loss_surf_dice, logits)
        else:
            loss_surf_dice = loss_ce * 0.0  # Zero loss with grad_fn

        # Skeleton Recall (only if weight > 0 and skeletons provided)
        if skel_w > 0 and skeletons is not None:
            loss_skel = self.skeleton_loss(logits, skeletons, labels)
        else:
            loss_skel = loss_ce * 0.0  # Zero loss with grad_fn

        loss = ce_w * loss_ce + dice_w * loss_dice + \
            surf_w * loss_surf_dice + skel_w * loss_skel
        return {'loss': loss, 'loss_ce': loss_ce, 'loss_dice': loss_dice,
                'loss_surf_dice': loss_surf_dice, 'loss_skel': loss_skel}

    def _compute_deep_supervision_loss(self, logits_list, labels, skeletons):
        """Compute multi-scale deep supervision loss (nnUNet-style)."""
        num_scales = len(logits_list)

        # nnUNet-style weights: exponential decay, skip last scale
        ds_weights = [1.0 / (2 ** i) for i in range(num_scales)]
        ds_weights[-1] = 0.0
        total_w = sum(ds_weights)
        ds_weights = [w / total_w for w in ds_weights]

        ce_w, dice_w, surf_w, skel_w = self._get_loss_weights()
        total_loss = total_ce = total_dice = 0
        # Initialize to None, will be set on first iteration
        total_surf_dice = total_skel = None
        first_loss_ce = None

        for i, (logits, weight) in enumerate(zip(logits_list, ds_weights)):
            if weight == 0.0:
                continue

            target_size = logits.shape[2:]
            labels_ds = self._downsample_labels(labels, target_size)
            logits, labels_ds, _ = self._prepare_tensors(logits, labels_ds)

            loss_ce = self._safe_loss(self.ce_loss(
                logits, labels_ds.long()), logits)
            loss_dice = self._safe_loss(
                self.dice_loss(logits, labels_ds.long()), logits)

            # Store first loss_ce for creating zero losses
            if first_loss_ce is None:
                first_loss_ce = loss_ce

            # Expensive losses only at full resolution
            if i == 0:
                if surf_w > 0:
                    logits_binary = logits[:, 1:2] - logits[:, 0:1]
                    total_surf_dice = self._safe_loss(
                        self.surface_dice_loss(
                            logits_binary, labels_ds.unsqueeze(1)).mean(), logits
                    )
                else:
                    total_surf_dice = loss_ce * 0.0

                if skel_w > 0 and skeletons is not None:
                    skeletons_ds = strip_metatensor(
                        self._downsample_labels(skeletons, target_size))
                    total_skel = self.skeleton_loss(
                        logits, skeletons_ds, labels_ds)
                else:
                    total_skel = loss_ce * 0.0

            if i == 0:
                scale_loss = ce_w * loss_ce + dice_w * loss_dice + \
                    surf_w * total_surf_dice + skel_w * total_skel
            else:
                scale_loss = (ce_w * loss_ce + dice_w *
                              loss_dice) / (ce_w + dice_w + 1e-8)

            total_loss += weight * scale_loss
            total_ce += weight * loss_ce
            total_dice += weight * loss_dice

        # Ensure surface losses are initialized (fallback if never set)
        if total_surf_dice is None:
            total_surf_dice = first_loss_ce * 0.0 if first_loss_ce is not None else torch.tensor(0.0, device=labels.device)
        if total_skel is None:
            total_skel = first_loss_ce * 0.0 if first_loss_ce is not None else torch.tensor(0.0, device=labels.device)

        return {'loss': total_loss, 'loss_ce': total_ce, 'loss_dice': total_dice,
                'loss_surf_dice': total_surf_dice, 'loss_skel': total_skel}

    def compute_metrics(self, pred, target):
        """Compute metrics ignoring class 2."""
        # pred: (B, C, D, H, W), target: (B, D, H, W)
        pred_class = torch.argmax(pred, dim=1)  # (B, D, H, W)

        # Mask out ignore_index
        mask = (target != self.hparams.ignore_index)

        # We need to handle the case where mask is empty, but doing it with purely tensor ops to avoid sync
        # Sum of mask gives number of valid pixels
        valid_count = mask.sum()

        pred_masked = pred_class * mask
        target_masked = target * mask

        # Binary dice (class 1 vs class 0) - only valid pixels matter
        # Since we masked with 0, and 0 is background, we need to be careful.
        # Actually, pred_masked has 0 where mask is False. Target too.
        # Class 1 is what we care about. 0 is background.
        # So (pred_masked == 1) filters out the ignored regions (which are 0) correctly?
        # No, if ignore_index was 2, we replaced it with 0? No, we didn't replace.
        # We just need to compute intersection on the mask.

        pred_binary = (pred_class == 1) & mask
        target_binary = (target == 1) & mask

        intersection = (pred_binary & target_binary).sum().float()
        union = pred_binary.sum().float() + target_binary.sum().float()

        dice = (2.0 * intersection) / (union + 1e-8)
        
        # If union is 0, dice should be 1.0. 
        # We can use torch.where to handle this without .item()
        dice = torch.where(union > 0, dice, torch.tensor(1.0, device=dice.device))

        # Accuracy
        # Only on mask
        correct = (pred_class == target) & mask
        accuracy = correct.sum().float() / (valid_count.float() + 1e-8)
        
        return {
            'dice': dice,
            'accuracy': accuracy
        }

    def _unpack_deep_supervision_output(self, output):
        """Convert deep supervision output to list of tensors (or None if single output)."""
        if isinstance(output, (list, tuple)):
            return list(output)
        if hasattr(output, 'dim') and output.dim() == 6:  # Stacked MONAI format: [B, num_outputs, C, D, H, W]
            return [output[:, i] for i in range(output.shape[1])]
        return None

    def _process_output(self, output, labels, skeletons):
        """Process model output and compute losses."""
        ds_outputs = self._unpack_deep_supervision_output(output)
        if ds_outputs is not None:
            losses = self._compute_deep_supervision_loss(
                ds_outputs, labels, skeletons)
            logits = ds_outputs[0]
        else:
            losses = self._compute_single_scale_loss(output, labels, skeletons)
            logits = output
        return losses, logits

    def _log_losses(self, prefix, losses, metrics, on_step=False):
        """Log loss components and metrics."""
        self.log(f'{prefix}/loss', losses['loss'],
                 on_step=on_step, on_epoch=True, prog_bar=True)
        self.log(f'{prefix}/loss_ce',
                 losses['loss_ce'], on_step=False, on_epoch=True)
        self.log(f'{prefix}/loss_dice',
                 losses['loss_dice'], on_step=False, on_epoch=True)
        self.log(f'{prefix}/loss_surf_dice',
                 losses['loss_surf_dice'], on_step=False, on_epoch=True)
        self.log(f'{prefix}/loss_skel',
                 losses['loss_skel'], on_step=False, on_epoch=True)
        self.log(f'{prefix}/dice', metrics['dice'],
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'{prefix}/accuracy',
                 metrics['accuracy'], on_step=False, on_epoch=True)

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            images, labels, skeletons = batch
        else:
            images, labels = batch
            skeletons = None

        # Iterative refinement: use model's own predictions to refine OOF channels
        if self.hparams.use_iterative_training and torch.rand(1).item() < self.hparams.iterative_training_prob:
            with torch.no_grad():
                pred1 = self(images)
                pred1 = pred1[0] if isinstance(pred1, (list, tuple)) else pred1
                oof_refined = (torch.softmax(pred1, dim=1)[:, 1:2] > self.hparams.iterative_training_threshold).float()
            # Replace all OOF channels (channels 1:) with refined predictions
            num_oof_channels = images.shape[1] - 1
            oof_refined_expanded = oof_refined.expand(-1, num_oof_channels, -1, -1, -1)
            images = torch.cat([images[:, 0:1], oof_refined_expanded], dim=1)

        # Ensure gradients are enabled for the main forward pass even if an outer
        # context accidentally disabled them (e.g., from a callback).
        with torch.enable_grad():
            losses, logits = self._process_output(self(images), labels, skeletons)
        metrics = self.compute_metrics(logits.detach(), labels)

        self._log_losses('train', losses, metrics, on_step=True)
        self.log(
            'train/lr', self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=False)

        if self.hparams.debug_mode and batch_idx == 0:
            self._store_debug_sample(images, labels, logits)

        loss = losses['loss']
        if not loss.requires_grad:
            # Attach a zero-valued term to keep autograd happy if loss is detached.
            # This avoids crashes on edge cases (e.g., all-ignore patches).
            param = next((p for p in self.model.parameters() if p.requires_grad), None)
            if param is not None:
                loss = loss + 0.0 * param.sum()
        return loss

    def on_before_zero_grad(self, optimizer):
        """Update EMA after optimizer step, before zeroing gradients."""
        if self.ema is not None:
            self.ema.update(self.model)

    def _store_debug_sample(self, images, labels, logits, target_list=None):
        """Store a sample for visualization."""
        pred_class = torch.argmax(logits[0], dim=0)
        target = self.debug_samples if target_list is None else target_list
        target.append({
            'image': images[0].detach().cpu(),
            'label': labels[0].detach().cpu(),
            'prediction': pred_class.detach().cpu(),
        })

    def on_train_epoch_end(self):
        """Visualize predictions at the end of each epoch."""
        if not self.hparams.debug_mode or not self.debug_samples:
            return

        output_dir = Path(self.trainer.log_dir) / 'debug_visualizations'
        output_dir.mkdir(parents=True, exist_ok=True)
        self._visualize_predictions(
            self.debug_samples[:3], output_dir /
            f'epoch_{self.current_epoch:03d}.png'
        )
        self.debug_samples = []

    def on_validation_epoch_end(self):
        """Visualize validation predictions at the end of each epoch."""
        if not self.hparams.debug_mode or not self.val_debug_samples:
            return

        output_dir = Path(self.trainer.log_dir) / 'debug_visualizations'
        output_dir.mkdir(parents=True, exist_ok=True)
        self._visualize_predictions(
            self.val_debug_samples[:3], output_dir /
            f'val_epoch_{self.current_epoch:03d}.png'
        )
        self.val_debug_samples = []

    def _visualize_predictions(self, samples, output_path):
        """Create visualization of predictions vs ground truth, including OOF predictions."""
        import matplotlib.pyplot as plt

        n = len(samples)
        num_channels = samples[0]['image'].shape[0]
        num_oof_channels = num_channels - 1  # First channel is image, rest are OOF

        # Calculate number of columns: image + OOF channels + GT + pred + overlays
        num_cols = 1 + num_oof_channels + 1 + 1 + 2  # image, oofs, gt, pred, oof1_overlay, pred_overlay
        fig, axes = plt.subplots(n, num_cols, figsize=(4 * num_cols, 4 * n))
        if n == 1:
            axes = axes.reshape(1, -1)

        for i, s in enumerate(samples):
            mid = s['image'].shape[1] // 2
            # Image has multiple channels: [0]=image, [1:]=binarized OOF channels
            img = s['image'][0, mid].numpy()
            oofs = [s['image'][c, mid].numpy() for c in range(1, num_channels)]
            lbl = s['label'][mid].numpy()
            pred = s['prediction'][mid].numpy()
            valid = lbl != 2

            col = 0

            # Column 0: Input image
            axes[i, col].imshow(img, cmap='gray')
            axes[i, col].set_title('Input Image')
            axes[i, col].axis('off')
            col += 1

            # OOF channels
            for oof_idx, oof in enumerate(oofs):
                axes[i, col].imshow(oof, cmap='RdYlGn', vmin=0, vmax=1)
                axes[i, col].set_title(f'OOF {oof_idx+1}')
                axes[i, col].axis('off')
                col += 1

            # Ground truth
            gt_vis = lbl.astype(float)
            gt_vis[lbl == 2] = 0.5
            axes[i, col].imshow(gt_vis, cmap='RdYlGn', vmin=0, vmax=1)
            axes[i, col].set_title('Ground Truth')
            axes[i, col].axis('off')
            col += 1

            # 2nd stage prediction
            axes[i, col].imshow(pred, cmap='RdYlGn', vmin=0, vmax=1)
            axes[i, col].set_title('Pred (2nd stage)')
            axes[i, col].axis('off')
            col += 1

            # OOF1 vs GT overlay (compare 1st OOF to GT)
            oof1 = oofs[0] if oofs else np.zeros_like(lbl)
            oof1_bin = (oof1 > 0.5).astype(float)
            oof_overlay = np.zeros((*oof1.shape, 3))
            oof_overlay[(oof1_bin == 1) & (lbl == 1) & valid] = [1, 1, 1]  # TP: white
            oof_overlay[(oof1_bin == 1) & (lbl == 0) & valid] = [0, 1, 0]  # FP: green
            oof_overlay[(oof1_bin == 0) & (lbl == 1) & valid] = [1, 0, 0]  # FN: red
            oof_overlay[~valid] = [0.5, 0.5, 0.5]  # Unlabeled: gray
            axes[i, col].imshow(oof_overlay)
            axes[i, col].set_title('OOF1 Overlay\n(TP/FP/FN)')
            axes[i, col].axis('off')
            col += 1

            # Prediction vs GT overlay (compare 2nd stage to GT)
            pred_overlay = np.zeros((*pred.shape, 3))
            pred_overlay[(pred == 1) & (lbl == 1) & valid] = [1, 1, 1]  # TP: white
            pred_overlay[(pred == 1) & (lbl == 0) & valid] = [0, 1, 0]  # FP: green
            pred_overlay[(pred == 0) & (lbl == 1) & valid] = [1, 0, 0]  # FN: red
            pred_overlay[~valid] = [0.5, 0.5, 0.5]  # Unlabeled: gray
            axes[i, col].imshow(pred_overlay)
            axes[i, col].set_title('Pred Overlay\n(TP/FP/FN)')
            axes[i, col].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            images, labels, skeletons = batch
        else:
            images, labels = batch
            skeletons = None
        ps = self.hparams.patch_size

        # Use EMA model for validation if available
        predictor = self.ema.module if self.ema is not None else self.model

        with torch.no_grad():
            output = sliding_window_inference(
                images, roi_size=(ps, ps, ps), sw_batch_size=2,
                predictor=predictor, overlap=0.125, mode='gaussian', progress=False
            )

        ds_outputs = self._unpack_deep_supervision_output(output)
        logits = ds_outputs[0] if ds_outputs else output
        losses = self._compute_single_scale_loss(logits, labels, skeletons)
        metrics = self.compute_metrics(logits, labels)

        self._log_losses('val', losses, metrics)
        self.log('val_loss', losses['loss'],
                 on_step=False, on_epoch=True, logger=False)
        self.log('val_dice', metrics['dice'],
                 on_step=False, on_epoch=True, logger=False)

        if self.hparams.debug_mode and batch_idx == 0:
            self._store_debug_sample(images, labels, logits, self.val_debug_samples)

        return losses['loss']

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}
        }
