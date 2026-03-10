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
from .primus import create_primus, create_primus_v2


def strip_metatensor(t):
    """Strip MONAI MetaTensor metadata if present."""
    return t.as_tensor() if hasattr(t, "as_tensor") else t


def create_model(in_channels, out_channels, channels, strides,
                 use_deep_supervision, n_blocks_per_stage=None):
    """Factory function to create the segmentation model."""
    return create_residual_unet(in_channels, out_channels, channels, strides,
                                n_blocks_per_stage=n_blocks_per_stage,
                                deep_supervision=use_deep_supervision)


class SegmentationModule(pl.LightningModule):
    """Lightning module for 3D surface segmentation."""

    def __init__(
        self,
        model_type = 'unet',
        in_channels=1, out_channels=2,
        channels=(32, 64, 128, 128), strides=(1, 2, 2, 2),
        lr=1e-3, weight_decay=1e-4, ignore_index=2,
        # (CE, Dice, SurfaceDice, SkeletonRecall)
        loss_weights=(0.3, 0.3, 0.2, 0.2),
        patch_size=128, surface_dice_iterations=5,
        val_check_interval=1,
        use_deep_supervision=False,
        deep_supervision_weights=(1.0, 0.5, 0.25),
        n_blocks_per_stage=(1, 3, 4, 6, 6, 6),
        debug_mode=False, compute_leaderboard_metrics=False,
        leaderboard_max_val_samples=0, leaderboard_surface_tolerance=2.0,
        ce_top_k=1.0,
        use_ema=False, ema_decay=0.999, ema_warmup=0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.debug_samples = []
        self._val_cache = []
        print(f"Model type: {model_type}")
        # Model
        if model_type == 'res_unet' or model_type == 'unet':
            self.model = create_model(
                in_channels, out_channels, channels, strides,
                use_deep_supervision, n_blocks_per_stage
            )
        elif model_type == 'primus':
            self.model = create_primus(
                in_channels=in_channels,
                out_channels=out_channels,
                input_shape=patch_size,
                drop_path_rate=drop_path_rate,
            )

        elif model_type == 'primus_v2':
            self.model = create_primus_v2(
                in_channels=in_channels,
                out_channels=out_channels,
                input_shape=patch_size,
                drop_path_rate=drop_path_rate,
            )

        else:
            raise Exception('invalid model type')

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
            loss_surf_dice = torch.tensor(0.0, device=logits.device)

        # Skeleton Recall (only if weight > 0 and skeletons provided)
        loss_skel = (self.skeleton_loss(logits, skeletons, labels) if (skel_w > 0 and skeletons is not None)
                     else torch.tensor(0.0, device=logits.device))

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
        total_surf_dice = total_skel = torch.tensor(0.0, device=labels.device)

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

            # Expensive losses only at full resolution
            if i == 0 and surf_w > 0:
                logits_binary = logits[:, 1:2] - logits[:, 0:1]
                total_surf_dice = self._safe_loss(
                    self.surface_dice_loss(
                        logits_binary, labels_ds.unsqueeze(1)).mean(), logits
                )
            if i == 0 and skel_w > 0 and skeletons is not None:
                skeletons_ds = strip_metatensor(
                    self._downsample_labels(skeletons, target_size))
                total_skel = self.skeleton_loss(
                    logits, skeletons_ds, labels_ds)

            if i == 0:
                scale_loss = ce_w * loss_ce + dice_w * loss_dice + \
                    surf_w * total_surf_dice + skel_w * total_skel
            else:
                scale_loss = (ce_w * loss_ce + dice_w *
                              loss_dice) / (ce_w + dice_w + 1e-8)

            total_loss += weight * scale_loss
            total_ce += weight * loss_ce
            total_dice += weight * loss_dice

        return {'loss': total_loss, 'loss_ce': total_ce, 'loss_dice': total_dice,
                'loss_surf_dice': total_surf_dice, 'loss_skel': total_skel}

    def compute_metrics(self, pred, target):
        """Compute metrics ignoring class 2."""
        # pred: (B, C, D, H, W), target: (B, D, H, W)
        pred_class = torch.argmax(pred, dim=1)  # (B, D, H, W)

        # Mask out ignore_index
        mask = (target != self.hparams.ignore_index)

        valid_count = mask.sum()


        pred_binary = (pred_class == 1) & mask
        target_binary = (target == 1) & mask

        intersection = (pred_binary & target_binary).sum().float()
        union = pred_binary.sum().float() + target_binary.sum().float()

        dice = (2.0 * intersection) / (union + 1e-8)

        dice = torch.where(union > 0, dice, torch.tensor(1.0, device=dice.device))


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
        losses, logits = self._process_output(self(images), labels, skeletons)
        metrics = self.compute_metrics(logits.detach(), labels)

        self._log_losses('train', losses, metrics, on_step=True)
        self.log(
            'train/lr', self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=False)

        if self.hparams.debug_mode and batch_idx == 0:
            self._store_debug_sample(images, labels, logits)

        return losses['loss']

    def on_before_zero_grad(self, optimizer):
        """Update EMA after optimizer step, before zeroing gradients."""
        if self.ema is not None:
            self.ema.update(self.model)

    def _store_debug_sample(self, images, labels, logits):
        """Store a sample for visualization."""
        pred_class = torch.argmax(logits[0], dim=0)
        self.debug_samples.append({
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

    def _visualize_predictions(self, samples, output_path):
        """Create visualization of predictions vs ground truth."""
        import matplotlib.pyplot as plt

        n = len(samples)
        fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
        if n == 1:
            axes = axes.reshape(1, -1)

        for i, s in enumerate(samples):
            mid = s['image'].shape[1] // 2
            img, lbl, pred = s['image'][0, mid].numpy(
            ), s['label'][mid].numpy(), s['prediction'][mid].numpy()
            valid = lbl != 2

            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title('Input')
            axes[i, 0].axis('off')

            gt_vis = lbl.astype(float)
            gt_vis[lbl == 2] = 0.5
            axes[i, 1].imshow(gt_vis, cmap='RdYlGn', vmin=0, vmax=1)
            axes[i, 1].set_title('GT')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred, cmap='RdYlGn', vmin=0, vmax=1)
            axes[i, 2].set_title('Pred')
            axes[i, 2].axis('off')

            # Overlay: TP=white, FP=green, FN=red, unlabeled=gray
            overlay = np.zeros((*pred.shape, 3))
            overlay[(pred == 1) & (lbl == 1) & valid] = [1, 1, 1]
            overlay[(pred == 1) & (lbl == 0) & valid] = [0, 1, 0]
            overlay[(pred == 0) & (lbl == 1) & valid] = [1, 0, 0]
            overlay[~valid] = [0.5, 0.5, 0.5]
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title('Overlay')
            axes[i, 3].axis('off')

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


        return losses['loss']

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss',
                'interval': 'epoch',
                'frequency': self.hparams.val_check_interval
            }
        }