#!/bin/bash
# Step 1: Download competition data from Kaggle
set -e

kaggle competitions download -c vesuvius-challenge-surface-detection
unzip -o vesuvius-challenge-surface-detection.zip -d .

echo "Step 1 complete."
