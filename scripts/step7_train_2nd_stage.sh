#!/bin/bash
# Step 7: Train 2nd stage (4-channel input: image + resEnc OOF + Primus OOF + PrimusV2 OOF)
set -e

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

COMMON="--full_csv train.csv --image_dir train_images_npy --label_dir train_labels_npy --skeleton_dir train_skeletons_npy --oof_dirs 1st_stage_cache Primus_1st_stage_cache PrimusV2_1st_stage_cache --channels 32 64 128 256 320 320 --strides 1 2 2 2 2 2 --n_blocks_per_stage 1 3 4 6 6 6 --patch_size 160 --batch_size 2 --epochs 200 --lr 1e-4 --weight_decay 1e-4 --use_amp --use_deep_supervision --deep_supervision_num 4 --loss_weights 0.4 0.4 0.1 0.1 --n_folds 5 --use_ema --ema_decay 0.999 --ema_warmup 1000 --val_interval 5 --early_stop_patience 15 --save_top_k 2 --seed 42"

for FOLD in 0 1 2 3 4; do
    python3 src_2nd_4th_stages/training/train_cv.py $COMMON \
        --fold_idx $FOLD \
        --output_dir cv_outputs_2nd_stage \
        --wandb_project vesuvius-2nd-stage
done

echo "Step 7 complete."
