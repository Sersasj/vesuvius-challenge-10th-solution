#!/bin/bash
# Step 2: Convert TIF to NPY, then remove deprecated samples
set -e

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python3 src/data/convert_to_npy.py --output_dir .
python3 src/data/remove_deprecated.py

echo "Step 2 complete."
