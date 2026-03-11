"""K-fold cross-validation training script for Vesuvius segmentation."""
import argparse
import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import torch
import numpy as np
import random
from sklearn.model_selection import KFold
import wandb
from monai.utils import set_determinism

from src_2nd_4th_stages.models.lightning_module import SegmentationModule
from src_2nd_4th_stages.data.dataset import VesuviusDataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

torch.set_float32_matmul_precision('medium')


def get_folds(df, n_folds=5, seed=42):
    """Split dataset into K folds."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(df.iloc[train_idx], df.iloc[val_idx]) for train_idx, val_idx in kf.split(df)]


def worker_init_fn(worker_id):
    """Initialize worker with deterministic seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def create_dataloaders(args, fold_dir):
    """Create train and validation dataloaders."""
    common_args = dict(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        skeleton_dir=args.skeleton_dir,
        oof_dirs=args.oof_dirs,
        patch_size=args.patch_size,
    )

    train_ds = VesuviusDataset(str(fold_dir / 'train.csv'), is_train=True, augment=True, **common_args)
    val_ds = VesuviusDataset(str(fold_dir / 'val.csv'), is_train=False, augment=False, **common_args)

    loader_args = dict(num_workers=args.num_workers, pin_memory=True,
                       prefetch_factor=args.prefetch_factor, worker_init_fn=worker_init_fn, persistent_workers=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, **loader_args)

    return train_loader, val_loader


def create_model(args):
    """Create and optionally load pretrained model."""
    # in_channels = 1 (image) + number of OOF directories
    num_oof_channels = len(args.oof_dirs)
    in_channels = 1 + num_oof_channels

    model = SegmentationModule(
        in_channels=in_channels, out_channels=2,
        channels=tuple(args.channels), strides=tuple(args.strides),
        lr=args.lr, weight_decay=args.weight_decay,
        loss_weights=tuple(args.loss_weights),
        debug_mode=getattr(args, 'debug', False),
        use_deep_supervision=getattr(args, 'use_deep_supervision', False),
        deep_supervision_weights=tuple(getattr(args, 'deep_supervision_weights', None)) if getattr(args, 'use_deep_supervision', False) else None,
        n_blocks_per_stage=tuple(args.n_blocks_per_stage),
        compute_leaderboard_metrics=args.compute_leaderboard_metrics,
        leaderboard_max_val_samples=args.leaderboard_max_val_samples,
        use_ema=getattr(args, 'use_ema', False),
        ema_decay=getattr(args, 'ema_decay', 0.999),
        ema_warmup=getattr(args, 'ema_warmup', 0),
        patch_size=args.patch_size,
        use_iterative_training=getattr(args, 'use_iterative_training', False),
        iterative_training_prob=getattr(args, 'iterative_training_prob', 0.5),
        iterative_training_threshold=getattr(args, 'iterative_training_threshold', 0.3),
    )

    # Only load weights manually if NOT resuming.
    # If resuming, trainer.fit(..., ckpt_path=...) handles everything.
    if args.pretrained_ckpt and not getattr(args, 'resume', False):
        print(f"Loading pretrained weights (fine-tuning mode) from: {args.pretrained_ckpt}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(args.pretrained_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f"Loaded pretrained weights from: {args.pretrained_ckpt}")

    return model


def get_checkpoint_config(args):
    """Determine checkpoint monitoring configuration."""
    monitor = args.monitor
    mode = args.monitor_mode

    if monitor == "auto":
        # Use val/dice as default metric (more stable than leaderboard_score)
        monitor = "val/dice"
        mode = "max"
    if mode == "auto":
        mode = "min" if "loss" in monitor.lower() else "max"

    filename = args.ckpt_filename
    if filename == "auto":
        filename = "best-{epoch:02d}-{val_dice:.4f}-{val_loss:.4f}"

    return monitor, mode, filename


def train_fold(args, fold_idx, train_df, val_df):
    """Train a single fold."""
    print(f"\n{'='*40}\nTraining Fold {fold_idx}/{args.n_folds - 1}\n{'='*40}")

    # Small dataset mode
    if args.small_dataset:
        frac = 0.1
        train_df = train_df.sample(n=max(1, int(len(train_df) * frac)), random_state=42 + fold_idx)
        val_df = val_df.sample(n=max(1, int(len(val_df) * frac)), random_state=42 + fold_idx)
        print(f"Small dataset mode: {len(train_df)} train, {len(val_df)} val")

    fold_dir = Path(args.output_dir) / f'fold_{fold_idx}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(fold_dir / 'train.csv', index=False)
    val_df.to_csv(fold_dir / 'val.csv', index=False)

    # WandB
    if wandb.run:
        wandb.finish()
    wandb_logger = (WandbLogger(project=args.wandb_project, name=f"refine_v1_all_losses_fold_{fold_idx}",
                                save_dir=str(fold_dir), config=vars(args),
                                tags=[f"fold_{fold_idx}", "cv"], group="cv_run")
                    if args.wandb_project else None)

    # Reproducibility
    pl.seed_everything(args.seed, workers=True)
    set_determinism(seed=args.seed)

    train_loader, val_loader = create_dataloaders(args, fold_dir)
    # Create model
    model = create_model(args)
    monitor, mode, filename = get_checkpoint_config(args)

    callbacks = [
        ModelCheckpoint(dirpath=fold_dir, filename=filename, monitor=monitor, mode=mode,
                        save_top_k=args.save_top_k, verbose=True),
        EarlyStopping(monitor=monitor, mode=mode, patience=args.early_stop_patience, verbose=True),
    ]

    trainer = pl.Trainer(
        default_root_dir=fold_dir, max_epochs=args.epochs, accelerator='auto', devices=1,
        precision='16-mixed' if args.use_amp else 32, callbacks=callbacks, logger=wandb_logger,
        log_every_n_steps=10, gradient_clip_val=1.0, check_val_every_n_epoch=args.val_interval,
        deterministic=False,
    )

    # Use ckpt_path for full restoration (optimizer, epoch, etc.)
    ckpt_path = args.pretrained_ckpt if args.resume else None
    if ckpt_path:
        print(f"Resuming training from checkpoint: {ckpt_path}")

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    if wandb_logger:
        wandb.finish()

    return callbacks[0].best_model_path


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="K-fold CV training for Vesuvius segmentation")

    # Data
    p.add_argument('--full_csv', required=True, help="Path to CSV with all training data")
    p.add_argument('--image_dir', default='train_images_npy')
    p.add_argument('--label_dir', default='train_labels_npy')
    p.add_argument('--skeleton_dir', default='train_skeletons_npy')
    p.add_argument('--oof_dirs', nargs='+', default=['1st_stage_cache', 'Primus_1st_stage_cache', 'PrimusV2_1st_stage_cache'],
                   help="Directory(ies) with OOF predictions (resEnc, primus, primusV2)")

    # Training
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--fold_idx', type=int, default=None, help="Train only this fold (0-based)")
    p.add_argument('--resume', action='store_true', help="Resume full trainer state from pretrained_ckpt")
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--patch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--prefetch_factor', type=int, default=2)
    p.add_argument('--use_amp', action='store_true', default=True)
    p.add_argument('--val_interval', type=int, default=5)
    p.add_argument('--small_dataset', action='store_true', help="Use 10% of data for testing")

    # Model (Residual UNet only)
    p.add_argument('--channels', type=int, nargs='+', default=[32, 64, 128, 256, 320, 320])
    p.add_argument('--strides', type=int, nargs='+', default=[1, 2, 2, 2, 2, 2])
    p.add_argument('--n_blocks_per_stage', type=int, nargs='+', default=[1, 3, 4, 6, 6, 6])
    p.add_argument('--pretrained_ckpt', default=None)

    # Loss
    p.add_argument('--loss_weights', type=float, nargs='+', default=[0.4, 0.4, 0.1, 0.1],
                   help="(CE, Dice, SurfaceDice, SkeletonRecall)")


    # Deep supervision
    p.add_argument('--use_deep_supervision', action='store_true')
    p.add_argument('--deep_supervision_num', type=int, default=4)
    p.add_argument('--deep_supervision_weights', type=float, nargs='+', default=[1.0, 0.5, 0.25, 0.125, 0.0625])

    # EMA
    p.add_argument('--use_ema', action='store_true', help="Use Exponential Moving Average of weights")
    p.add_argument('--ema_decay', type=float, default=0.999, help="EMA decay rate (0.999=slow, 0.99=fast)")
    p.add_argument('--ema_warmup', type=int, default=0, help="Steps before starting EMA")

    # Iterative Training
    p.add_argument('--use_iterative_training', action='store_true', help="Use iterative refinement during training")
    p.add_argument('--iterative_training_prob', type=float, default=0.5, help="Probability of applying iterative refinement")
    p.add_argument('--iterative_training_threshold', type=float, default=0.3, help="Threshold for binarizing refined predictions")

    # Checkpointing
    p.add_argument('--output_dir', default='cv_outputs')
    p.add_argument('--monitor', default="auto")
    p.add_argument('--monitor_mode', default="auto", choices=["auto", "min", "max"])
    p.add_argument('--ckpt_filename', default="auto")
    p.add_argument('--save_top_k', type=int, default=2)
    p.add_argument('--early_stop_patience', type=int, default=15)

    # Metrics
    p.add_argument('--compute_leaderboard_metrics', action='store_true')
    p.add_argument('--leaderboard_max_val_samples', type=int, default=0)

    # Logging
    p.add_argument('--wandb_project', default=None)
    p.add_argument('--debug', action='store_true')

    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.full_csv)
    folds = get_folds(df, n_folds=args.n_folds, seed=args.seed)

    # Train specified fold or all folds
    fold_indices = [args.fold_idx] if args.fold_idx is not None else range(len(folds))

    for i in fold_indices:
        train_fold(args, i, folds[i][0], folds[i][1])

if __name__ == '__main__':
    main()
