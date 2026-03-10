#!/bin/bash
# Step 6: Generate 1st-stage OOF predictions for each model
set -e

COMMON="--num_stages 1 --folds all"

# ResEncL OOF
python3 generate_oof.py $COMMON \
    --checkpoint_dir cv_outputs_resencl \
    --ckpt_pattern "fold_{fold}/best*.ckpt" \
    --save_preds_dir 1st_stage_cache

# Primus OOF
python3 generate_oof.py $COMMON \
    --checkpoint_dir cv_outputs_primus \
    --ckpt_pattern "fold_{fold}/best*.ckpt" \
    --save_preds_dir Primus_1st_stage_cache

# PrimusV2 OOF
python3 generate_oof.py $COMMON \
    --checkpoint_dir cv_outputs_primus_v2 \
    --ckpt_pattern "fold_{fold}/best*.ckpt" \
    --save_preds_dir PrimusV2_1st_stage_cache

echo "Step 6 complete."
