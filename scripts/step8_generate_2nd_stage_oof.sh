#!/bin/bash
# Step 8: Generate 2nd stage OOF predictions (probability maps for 3rd stage training)
set -e

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python3 generate_oof.py \
    --num_stages 2 \
    --folds all \
    --checkpoint_dir_2nd cv_outputs_2nd_stage \
    --save_probs_dir 2nd_stage_cache

echo "Step 8 complete. 2nd stage OOF saved to 2nd_stage_cache/"
