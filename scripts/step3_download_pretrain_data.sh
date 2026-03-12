#!/bin/bash
# Step 3: Download additional labeled data for pretraining
set -e

python3 download_all_data.py

echo "Step 3 complete."
