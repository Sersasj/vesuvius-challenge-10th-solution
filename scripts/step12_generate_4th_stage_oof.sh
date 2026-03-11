#!/bin/bash
# Step 12: Generate 4th stage OOF predictions (probability maps for 5th stage training)
set -e

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python3 generate_oof.py \
    --num_stages 4 \
    --folds all \
    --cache_3rd_stage_dir 3rd_stage_cache \
    --checkpoint_dir_4th cv_outputs_4th_stage \
    --save_probs_dir 4th_stage_cache

echo "Step 12 complete. 4th stage OOF saved to 4th_stage_cache/"
