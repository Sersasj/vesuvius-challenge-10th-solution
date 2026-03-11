"""
Convert TIF dataset to .npy format for faster loading.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

def load_volume(path):
    """Load multi-frame TIF as 3D volume."""
    img = Image.open(path)
    n_frames = getattr(img, 'n_frames', 1)
    frames = []
    for i in range(n_frames):
        img.seek(i)
        frames.append(np.array(img))
    return np.stack(frames, axis=0)

def convert_dataset(args):
    """Convert train_images and train_labels to .npy."""
    
    splits = ['train.csv']
    all_ids = set()
    
    # Collect all IDs from splits
    for split in splits:
        if Path(split).exists():
            df = pd.read_csv(split)
            all_ids.update(df['id'].astype(str).tolist())
            
    print(f"Found {len(all_ids)} unique samples to convert.")
    
    # Directories
    dirs = {
        'train_images': args.output_dir / 'train_images_npy',
        'train_labels': args.output_dir / 'train_labels_npy'
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        
    # Convert
    for sample_id in tqdm(all_ids):
        # Image
        img_path = Path('train_images') / f'{sample_id}.tif'
        if img_path.exists():
            vol = load_volume(str(img_path))
            np.save(dirs['train_images'] / f'{sample_id}.npy', vol)
            
        # Label
        lbl_path = Path('train_labels') / f'{sample_id}.tif'
        if lbl_path.exists():
            vol = load_volume(str(lbl_path))
            np.save(dirs['train_labels'] / f'{sample_id}.npy', vol)

    print("\nConversion complete!")
    print(f"Images saved to: {dirs['train_images']}")
    print(f"Labels saved to: {dirs['train_labels']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=Path, default=Path('.'))
    args = parser.parse_args()
    
    convert_dataset(args)

