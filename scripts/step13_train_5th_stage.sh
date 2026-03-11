#!/bin/bash
# Step 13: Train 5th stage DeformNet (2-channel input: image + 4th stage OOF)
set -e

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

COMMON="--full_csv train.csv --image_dir train_images_npy --label_dir train_labels_npy --skeleton_dir train_skeletons_npy --oof_dirs 4th_stage_cache --patch_size 160 --batch_size 2 --epochs 200 --lr 1e-4 --weight_decay 1e-4 --use_amp --use_ema --ema_decay 0.999 --ema_warmup 1000 --val_interval 5 --early_stop_patience 15 --save_top_k 2 --seed 42"

for FOLD in 0 1 2 3 4; do
    python3 src_deformnet_stage/training/train_cv.py $COMMON \
        --fold_idx $FOLD \
        --output_dir cv_outputs_5th_stage \
        --wandb_project vesuvius-5th-stage
done

echo "Step 13 complete."
