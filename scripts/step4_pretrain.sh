#!/bin/bash
# Step 4: Pretrain ResEncL, Primus, PrimusV2 on additional data
set -e

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

COMMON="--full_csv additional_data/samples.csv --image_dir additional_data/images --label_dir additional_data/labels --skeleton_dir additional_data/skeletons --n_folds 5 --patch_size 160 --batch_size 2 --epochs 400 --lr 5e-5 --use_amp --use_deep_supervision --deep_supervision_num 4 --loss_weights 0.4 0.4 0.1 0.1 --use_ema --ema_decay 0.999 --ema_warmup 1000 --val_interval 5 --seed 42"

# ResEncL
python3 src/training/train_cv.py $COMMON \
    --model_type unet --channels 32 64 128 256 320 320 --strides 1 2 2 2 2 2 --n_blocks_per_stage 1 3 4 6 6 6 \
    --fold_idx 0 --output_dir pretrained_checkpoints/resencl --wandb_project vesuvius-pretrain

# Primus
python3 src/training/train_cv.py $COMMON \
    --model_type primus --drop_path_rate 0.0 \
    --fold_idx 0 --output_dir pretrained_checkpoints/primus --wandb_project vesuvius-pretrain

# PrimusV2
python3 src/training/train_cv.py $COMMON \
    --model_type primus_v2 --drop_path_rate 0.2 \
    --fold_idx 0 --output_dir pretrained_checkpoints/primus_v2 --wandb_project vesuvius-pretrain

echo "Step 4 complete."
