#!/bin/bash
# Step 4: Pretrain ResEncL, Primus, PrimusV2 on additional data
set -e

COMMON="--full_csv additional_data/samples.csv --image_dir additional_data/images --label_dir additional_data/labels --skeleton_dir additional_data/skeletons --epochs 100 --batch_size 2 --patch_size 128 --lr 1e-4 --n_folds 1 --fold_idx 0 --use_amp --use_deep_supervision --use_ema --ema_decay 0.999 --use_cutmix --loss_weights 0.4 0.4 0.1 0.1 --val_interval 5 --early_stop_patience 20"

# ResEncL
python3 -m src.training.train_cv $COMMON \
    --model_type unet --channels 32 64 128 256 320 320 --strides 1 2 2 2 2 2 --n_blocks_per_stage 1 3 4 6 6 6 \
    --output_dir pretrained_checkpoints/resencl --wandb_project vesuvius-pretrain

# Primus
python3 -m src.training.train_cv $COMMON \
    --model_type primus --drop_path_rate 0.0 \
    --output_dir pretrained_checkpoints/primus --wandb_project vesuvius-pretrain

# PrimusV2
python3 -m src.training.train_cv $COMMON \
    --model_type primus_v2 --drop_path_rate 0.2 \
    --output_dir pretrained_checkpoints/primus_v2 --wandb_project vesuvius-pretrain

echo "Step 4 complete."
