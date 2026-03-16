#!/bin/bash
# Step 5: Train 5-fold CV for each model (fine-tune from pretrained)
set -e

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

COMMON="--full_csv train.csv --image_dir train_images_npy --label_dir train_labels_npy --skeleton_dir train_skeletons_npy --n_folds 5 --patch_size 160 --batch_size 2 --epochs 400 --lr 5e-5 --use_amp --use_deep_supervision --deep_supervision_num 4 --loss_weights 0.4 0.4 0.1 0.1 --use_ema --ema_decay 0.999 --ema_warmup 1000 --val_interval 5 --seed 42"

RESENCL_CKPT=$(ls pretrained_checkpoints/resencl/fold_0/best*.ckpt 2>/dev/null | head -1)
PRIMUS_CKPT=$(ls pretrained_checkpoints/primus/fold_0/best*.ckpt 2>/dev/null | head -1)
PRIMUSV2_CKPT=$(ls pretrained_checkpoints/primus_v2/fold_0/best*.ckpt 2>/dev/null | head -1)

# ResEncL 5-fold
for FOLD in 0 1 2 3 4; do
    python3 src/training/train_cv.py $COMMON \
        --model_type unet --channels 32 64 128 256 320 320 --strides 1 2 2 2 2 2 --n_blocks_per_stage 1 3 4 6 6 6 \
        --pretrained_ckpt "$RESENCL_CKPT" \
        --fold_idx $FOLD --output_dir cv_outputs_resencl --wandb_project vesuvius-cv
done

# Primus 5-fold
for FOLD in 0 1 2 3 4; do
    python3 src/training/train_cv.py $COMMON \
        --model_type primus --drop_path_rate 0.0 \
        --pretrained_ckpt "$PRIMUS_CKPT" \
        --fold_idx $FOLD --output_dir cv_outputs_primus --wandb_project vesuvius-cv
done

# PrimusV2 5-fold
for FOLD in 0 1 2 3 4; do
    python3 src/training/train_cv.py $COMMON \
        --model_type primus_v2 --drop_path_rate 0.2 \
        --pretrained_ckpt "$PRIMUSV2_CKPT" \
        --fold_idx $FOLD --output_dir cv_outputs_primus_v2 --wandb_project vesuvius-cv
done

echo "Step 5 complete."
