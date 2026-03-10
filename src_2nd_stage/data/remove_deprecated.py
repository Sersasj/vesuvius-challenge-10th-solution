import os
import pandas as pd
from pathlib import Path

# Paths
DEPRECATED_DIR = Path("deprecated_train_images")
TRAIN_CSV = Path("train.csv")
IMAGES_DIR = Path("train_images_npy")
LABELS_DIR = Path("train_labels_npy")

def main():
    # 1. Get list of deprecated IDs
    if not DEPRECATED_DIR.exists():
        print(f"Error: {DEPRECATED_DIR} does not exist.")
        return

    # Extract IDs from filenames (e.g., "123.tif" -> "123")
    deprecated_ids = [f.stem for f in DEPRECATED_DIR.glob("*.tif")]
    print(f"Found {len(deprecated_ids)} deprecated files.")
    
    if not deprecated_ids:
        print("No deprecated files found.")
        return

    # 2. Update train.csv
    if TRAIN_CSV.exists():
        df = pd.read_csv(TRAIN_CSV)
        initial_count = len(df)
        
        # Ensure ID column is string for comparison
        df['id'] = df['id'].astype(str)
        
        # Filter out deprecated IDs
        df_clean = df[~df['id'].isin(deprecated_ids)]
        removed_count = initial_count - len(df_clean)
        
        if removed_count > 0:
            df_clean.to_csv(TRAIN_CSV, index=False)
            print(f"Removed {removed_count} rows from {TRAIN_CSV}.")
        else:
            print(f"No rows matching deprecated IDs found in {TRAIN_CSV}.")
    else:
        print(f"Warning: {TRAIN_CSV} not found.")

    # 3. Remove .npy files
    for file_id in deprecated_ids:
        # Check and remove image
        img_path = IMAGES_DIR / f"{file_id}.npy"
        if img_path.exists():
            try:
                os.remove(img_path)
                print(f"Deleted {img_path}")
            except OSError as e:
                print(f"Error deleting {img_path}: {e}")
        
        # Check and remove label
        lbl_path = LABELS_DIR / f"{file_id}.npy"
        if lbl_path.exists():
            try:
                os.remove(lbl_path)
                print(f"Deleted {lbl_path}")
            except OSError as e:
                print(f"Error deleting {lbl_path}: {e}")

if __name__ == "__main__":
    main()

