#!/bin/bash
# Step 10: Generate 3rd stage OOF predictions (probability maps for 4th stage training)
set -e

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python3 generate_oof.py \
    --num_stages 3 \
    --folds all \
    --cache_2nd_stage_dir 2nd_stage_cache \
    --checkpoint_dir_3rd cv_outputs_3rd_stage \
    --save_probs_dir 3rd_stage_cache

echo "Step 10 complete. 3rd stage OOF saved to 3rd_stage_cache/"
