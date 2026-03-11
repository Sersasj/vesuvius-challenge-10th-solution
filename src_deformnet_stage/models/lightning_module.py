"""PyTorch Lightning module for 3D segmentation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from monai.inferers import sliding_window_inference
import numpy as np
from pathlib import Path
from .deformnet import DiffeomorphicNetwork, warp_vol_using_disp
from ..losses import SurfaceDiceLoss
from ..utils.ema import EMA
from ..losses.deformnet_losses import (
    gaussian_blur_3d,
    svf_smoothness,
    jacobian_log_barrier,
    topo_sparsity,
    topo_tv,
    topo_boundary,
    SkeletonRecallLoss,
)
from monai.losses import DiceCELoss


def strip_metatensor(t):
    """Strip MONAI MetaTensor metadata if present."""
    return t.as_tensor() if hasattr(t, "as_tensor") else t


class SegmentationModule(pl.LightningModule):
    def __init__(self, in_channels=2, out_channels=4,
                 n_steps=6, max_v=1.5, max_topo_offset=1.0,
                 lambda_jac=0.3, lambda_smooth=0.05, lambda_sparse=0.1,
                 lambda_tv = 0.02, lambda_boundary =0.1,
                 use_ema=True, ema_decay=0.999, ema_warmup=0,
                 kernel_size = 3, sigma = 5,
                 surface_dice_iterations = 5,
                 ce_top_k=1.0, ignore_index=2, oof_threshold = 0.3, oof_aug_sigma = 0.1,
                 apply_gaussian_oof = True, threshold = 0.5,
                 lr=1e-3, weight_decay=1e-4, loss_weights=(0.3, 0.3, 0.2, 0.2),
                 debug_mode=False, compute_leaderboard_metrics=False,
                 leaderboard_max_val_samples=0, leaderboard_surface_tolerance=2.0,
                 patch_size = 160
                 ):
        super().__init__()
        #segmentation module setup
        self.save_hyperparameters()
        self.debug_samples = []
        self.val_debug_samples = []
        self._val_cache = []
        self.ignore_index = ignore_index

        #basic losses
        self.seg_loss = DiceCELoss(
            sigmoid=False,
            to_onehot_y=True,
            softmax=False,
            reduction="mean",
            squared_pred=True,
            lambda_ce= 0.4,
            lambda_dice= 0.4,
        )
        self.surface_dice_loss = SurfaceDiceLoss(
            ignore_label=ignore_index, soft_skel_iterations=surface_dice_iterations, smooth=1.0
        )
        self.skel_loss = SkeletonRecallLoss()

        #deformnet
        self.model = DiffeomorphicNetwork(
            in_channels, out_channels, n_steps, max_v, max_topo_offset)
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.oof_threshold = oof_threshold
        self.oof_aug_sigma = oof_aug_sigma
        self.apply_gaussian_oof = apply_gaussian_oof
        self.threshold = threshold

        #loss configs
        self.lambda_jac = lambda_jac
        self.lambda_smooth = lambda_smooth
        self.lambda_sparse = lambda_sparse
        self.lambda_tv = lambda_tv
        self.lambda_boundary = lambda_boundary

        # EMA (optional)
        self.ema = None
        if use_ema:
            self.ema = EMA(self.model, decay=ema_decay, warmup=ema_warmup)

    def forward(self, x, return_params=False):
        model = self.ema.module if (self.ema is not None and not self.training) else self.model
        out = model(x, return_params=return_params)
        return out

    def training_step(self, batch, batch_idx):
        #images: Cat([vol, oof1]); shape = (B, 2, D, W, D)
        if len(batch) == 3:
            images, labels, skeletons = batch
        else:
            images, labels = batch
            skeletons = None

        # ---- OOF augmentation ----
        vol = images[:, 0:1]
        mask_oof = images[:, 1:2]
        #mask_oof = apply_random_thickness_augmentation(mask_oof)
        if self.apply_gaussian_oof:
            mask_oof = gaussian_blur_3d(
                mask_oof, self.kernel_size, self.sigma
            )
        _input = torch.cat([vol, mask_oof], dim=1)

        #forward
        prediction, v, phi, t  = self(_input, return_params=True)

        #compute basic loss
        losses, logits = self._process_output(prediction, labels, skeletons)

        #compute warp loss
        warp_losses = self.comppute_warp_loss(mask_oof, v, phi, t)

        #compute total loss
        losses = self.compute_total_loss(losses, warp_losses)

        #metric & log
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

    def compute_total_loss(self, losses, warp_losses):
        ce_w, dice_w, surf_w, skel_w = self._get_loss_weights()
        total_loss = losses['loss_dice_ce'] + \
               0.1 * losses['loss_skel']+ \
                    0.1 * losses['loss_surf_dice'] +\
               self.lambda_jac * warp_losses['loss_jac']\
                + self.lambda_smooth * warp_losses['loss_smooth']\
                + self.lambda_sparse * warp_losses['loss_sparse']\
                + self.lambda_tv * warp_losses['loss_tv']\
                + self.lambda_boundary * warp_losses['loss_boundary'] \

        losses['loss'] = total_loss

        for k, v in warp_losses.items():
            losses[k] = v
        return losses

    def comppute_warp_loss(self, mask_oof, v, phi, t):
        L_smooth = svf_smoothness(v)
        L_jac = jacobian_log_barrier(phi)

        # ---------------- topo-gate regularization ----------------
        with torch.no_grad():
            warped = warp_vol_using_disp(mask_oof, phi)

        L_sparse = topo_sparsity(t)
        L_tv = topo_tv(t)
        L_boundary = topo_boundary(t, warped)
        return {'loss_smooth': L_smooth, 'loss_jac': L_jac,
                'loss_sparse': L_sparse, 'loss_tv': L_tv, 'loss_boundary': L_boundary}

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

    def _compute_single_scale_loss(self, logits, labels, skeletons):
        """Compute loss at a single scale."""
        logits, labels, skeletons = self._prepare_tensors(
            logits, labels, skeletons)
        ce_w, dice_w, surf_w, skel_w = self._get_loss_weights()

        _labels = labels.float()[:, None]
        valid_mask = _labels != self.ignore_index
        pred_warped = logits * valid_mask
        mask = _labels * valid_mask
        loss_dice_ce = self.seg_loss(logits * valid_mask,
                                     _labels * valid_mask)
        loss_skel = self.skel_loss(pred_warped, skeletons * valid_mask, mask)
        eps = 1e-6
        logits_binary = torch.log(pred_warped.clamp(eps, 1 - eps)) \
                        - torch.log((1 - pred_warped).clamp(eps, 1 - eps))
        loss_surf_dice = self.surface_dice_loss(logits_binary, mask).mean()
        return {'loss_dice_ce': loss_dice_ce,
                'loss_skel': loss_skel,
                'loss_surf_dice': loss_surf_dice
                }

    def compute_metrics(self, pred, target):
        """Compute metrics ignoring class 2."""
        # pred: (B, C, D, H, W), target: (B, D, H, W)
        pred_class = (pred[:, 0] > self.threshold).float()

        # Mask out ignore_index
        mask = (target != self.hparams.ignore_index)

        # We need to handle the case where mask is empty, but doing it with purely tensor ops to avoid sync
        # Sum of mask gives number of valid pixels
        valid_count = mask.sum()

        pred_masked = pred_class * mask
        target_masked = target * mask

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


    def _process_output(self, output, labels, skeletons):
        losses = self._compute_single_scale_loss(output, labels, skeletons)
        logits = output
        return losses, logits

    def _log_losses(self, prefix, losses, metrics, on_step=False):
        for k, v in losses.items():
            self.log(f'{prefix}/{k}', v,
                     on_step=on_step, on_epoch=True, prog_bar=True)
        self.log(f'{prefix}/dice', metrics['dice'],
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'{prefix}/accuracy',
                 metrics['accuracy'], on_step=False, on_epoch=True)

    def on_before_zero_grad(self, optimizer):
        """Update EMA after optimizer step, before zeroing gradients."""
        if self.ema is not None:
            self.ema.update(self.model)

    def _store_debug_sample(self, images, labels, logits, target_list=None):
        """Store a sample for visualization."""
        pred_class = (logits[0, 0] > self.threshold).float()
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

        # ---- OOF augmentation ----
        vol = images[:, 0:1]
        mask_oof = images[:, 1:2]
        if self.apply_gaussian_oof:
            mask_oof = gaussian_blur_3d(
                mask_oof, self.kernel_size, self.sigma
            )
        _input = torch.cat([vol, mask_oof], dim=1)

        # Use EMA model for validation if available
        predictor = self.ema.module if self.ema is not None else self.model

        with torch.no_grad():
            output = sliding_window_inference(
                _input, roi_size=(ps, ps, ps), sw_batch_size=2,
                predictor=predictor, overlap=0.125, mode='gaussian', progress=False
            )

        logits = output
        losses = self._compute_single_scale_loss(logits, labels, skeletons)
        metrics = self.compute_metrics(logits, labels)

        self._log_losses('val', losses, metrics)
        self.log('val_loss', losses['loss_dice_ce'],
                 on_step=False, on_epoch=True, logger=False)
        self.log('val_dice', metrics['dice'],
                 on_step=False, on_epoch=True, logger=False)

        if self.hparams.debug_mode and batch_idx == 0:
            self._store_debug_sample(images, labels, logits, self.val_debug_samples)

        return losses['loss_dice_ce']

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