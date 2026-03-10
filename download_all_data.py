import numpy as np
import pandas as pd
from pathlib import Path
from vesuvius import Volume
import random
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import tifffile
from io import BytesIO
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Labeled data
ASH2TXT_BASE_URL = "https://dl.ash2txt.org/datasets/seg-derived-recto-surfaces/"
ASH2TXT_IMAGES_URL = ASH2TXT_BASE_URL + "imagesTr/"
ASH2TXT_LABELS_URL = ASH2TXT_BASE_URL + "labelsTr/"

# Unlabeled data 
PHERC_0139_URL = "https://data.aws.ash2txt.org/samples/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr/0"
PHERC_0009B_URL = "https://data.aws.ash2txt.org/samples/PHerc0009B/volumes/20250521125136-8.640um-1.2m-116keV-masked.zarr/0"

OUTPUT_IMAGE_DIR = Path("additional_data/images")
OUTPUT_LABEL_DIR = Path("additional_data/labels")
CSV_PATH = Path("additional_data/samples.csv")

CHUNK_SIZE = 320
NUM_CHUNKS_PER_SCROLL = 100

# Limits (set to None to download all)
MAX_LABELED_SAMPLES = 1800  # Set to a number  to limit labeled data downloads
MAX_UNLABELED_SAMPLES = 0  # Set to a number to limit unlabeled data downloads

# Visualization
VISUALIZE_SAMPLES = False  # Set to True to plot sample images and labels
NUM_SAMPLES_TO_PLOT = 100   # Number of samples to visualize

OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Helper Functions
# ============================================================================

def list_files_from_url(url):
    """Scrape directory listing to get file names."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        files = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and not href.startswith('?') and href not in ['../', '../', '/']:
                filename = href.rstrip('/')
                if filename and filename != '..' and not filename.endswith('/'):
                    files.append(filename)
        
        return files
    except Exception as e:
        print(f"Error listing files from {url}: {e}")
        return []


def download_tif(url, target_path):
    """Download TIF file and convert to NPY."""
    try:
        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()
        
        # Read TIF from memory
        tif_data = BytesIO(response.content)
        volume = tifffile.imread(tif_data)
        
        # Save as NPY
        np.save(target_path, volume.astype(np.uint8))
        return True
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        return False


# ============================================================================
# Download Labeled Data
# ============================================================================

def download_labeled_data(existing_ids):
    """Download labeled data from ash2txt.org."""
    print("\n" + "="*60)
    print("DOWNLOADING LABELED DATA from ash2txt.org")
    print("="*60)
    
    samples = []
    
    # Get list of files
    print("\nFetching file list from server...")
    image_files = list_files_from_url(ASH2TXT_IMAGES_URL)
    label_files = list_files_from_url(ASH2TXT_LABELS_URL)
    
    # Filter for .tif files
    image_files = [f for f in image_files if f.endswith('.tif') or f.endswith('.tiff')]
    label_files = [f for f in label_files if f.endswith('.tif') or f.endswith('.tiff')]
    
    print(f"Found {len(image_files)} images and {len(label_files)} labels")
    
    if not image_files:
        print("⚠ No files found. The website may require manual download or different access method.")
        return samples
    
    # Match image and label pairs
    # Images have _0000 suffix (e.g., s1_z10240_y2560_x2560_0000.tif)
    # Labels don't have it (e.g., s1_z10240_y2560_x2560.tif)
    image_dict = {}
    for f in image_files:
        # Remove .tif/.tiff extension and _0000 suffix
        base_name = f.replace('.tif', '').replace('.tiff', '')
        if base_name.endswith('_0000'):
            base_name = base_name[:-5]  # Remove _0000
        image_dict[base_name] = f
    
    label_dict = {f.replace('.tif', '').replace('.tiff', ''): f for f in label_files}
    
    matched_pairs = []
    for sample_name in image_dict.keys():
        if sample_name in label_dict:
            matched_pairs.append((sample_name, image_dict[sample_name], label_dict[sample_name]))
    
    print(f"Found {len(matched_pairs)} matching image-label pairs")
    
    if not matched_pairs:
        print("⚠ No matching pairs found")
        return samples
    
    # Limit number of samples if specified
    if MAX_LABELED_SAMPLES is not None:
        matched_pairs = matched_pairs[:MAX_LABELED_SAMPLES]
        print(f"Limiting to {len(matched_pairs)} samples (MAX_LABELED_SAMPLES={MAX_LABELED_SAMPLES})")
    
    # Download each pair
    downloaded = 0
    skipped = 0
    
    for sample_name, image_file, label_file in tqdm(matched_pairs, desc="Downloading"):
        # Generate unique ID
        sample_id = random.randint(1000000000, 9999999999)
        while sample_id in existing_ids:
            sample_id = random.randint(1000000000, 9999999999)
        existing_ids.add(sample_id)
        
        # Download paths
        image_url = ASH2TXT_IMAGES_URL + image_file
        label_url = ASH2TXT_LABELS_URL + label_file
        
        image_path = OUTPUT_IMAGE_DIR / f"{sample_id}.npy"
        label_path = OUTPUT_LABEL_DIR / f"{sample_id}.npy"
        
        # Download image and label
        if download_tif(image_url, image_path):
            if download_tif(label_url, label_path):
                samples.append({
                    'id': sample_id,
                    'source': 'ash2txt_labeled',
                    'original_name': sample_name,
                    'has_label': True
                })
                downloaded += 1
            else:
                image_path.unlink(missing_ok=True)
                skipped += 1
        else:
            skipped += 1
    
    print(f"\n✓ Downloaded {downloaded} labeled samples")
    if skipped > 0:
        print(f"⚠ Skipped {skipped} samples due to errors")
    
    return samples



def extract_chunks_from_scroll(volume, scroll_name, existing_ids, num_chunks):
    """Extract random chunks from scroll volume (unlabeled)."""
    vol_shape = volume.shape()
    print(f"  Volume shape: {vol_shape}")
    
    # Limit number of chunks if specified
    if MAX_UNLABELED_SAMPLES is not None:
        num_chunks = min(num_chunks, MAX_UNLABELED_SAMPLES)
        print(f"  Limiting to {num_chunks} samples (MAX_UNLABELED_SAMPLES={MAX_UNLABELED_SAMPLES})")
    
    z_start = vol_shape[0] // 2 - CHUNK_SIZE // 2
    y_start = vol_shape[1] // 2 - CHUNK_SIZE // 2
    x_start = vol_shape[2] // 2 - CHUNK_SIZE // 2
    
    samples = []
    skipped = 0
    
    for i in tqdm(range(num_chunks), desc=f"  {scroll_name}"):
        z_offset = random.randint(-1000, 1000)
        y_offset = random.randint(-500, 500)
        x_offset = random.randint(-500, 500)
        
        z = max(0, min(vol_shape[0] - CHUNK_SIZE, z_start + z_offset))
        y = max(0, min(vol_shape[1] - CHUNK_SIZE, y_start + y_offset))
        x = max(0, min(vol_shape[2] - CHUNK_SIZE, x_start + x_offset))
        
        chunk = volume[z:z+CHUNK_SIZE, y:y+CHUNK_SIZE, x:x+CHUNK_SIZE]
        
        # Skip if >20% black
        if (chunk == 0).sum() / chunk.size > 0.2:
            skipped += 1
            continue
        
        sample_id = random.randint(10000000, 99999999)
        while sample_id in existing_ids:
            sample_id = random.randint(10000000, 99999999)
        existing_ids.add(sample_id)
        
        # Save image only (no label - will be pseudo-labeled later)
        np.save(OUTPUT_IMAGE_DIR / f"{sample_id}.npy", chunk.astype(np.uint8))
        
        samples.append({
            'id': sample_id,
            'source': scroll_name,
            'has_label': False
        })
    
    print(f"  ✓ Saved {len(samples)} chunks (skipped {skipped})")
    return samples


def download_unlabeled_data(existing_ids):
    """Download unlabeled data from scrolls."""
    print("\n" + "="*60)
    print("DOWNLOADING UNLABELED DATA from scrolls")
    print("="*60)
    
    samples = []
    
    print("\n--- PHerc 0139 ---")
    try:
        scroll_0139 = Volume(type="zarr", path=PHERC_0139_URL)
        samples.extend(extract_chunks_from_scroll(
            scroll_0139, "PHerc0139", existing_ids, NUM_CHUNKS_PER_SCROLL
        ))
    except Exception as e:
        print(f"  ⚠ Error: {e}")
    
    print("\n--- PHerc 0009B ---")
    try:
        scroll_0009B = Volume(type="zarr", path=PHERC_0009B_URL)
        samples.extend(extract_chunks_from_scroll(
            scroll_0009B, "PHerc0009B", existing_ids, NUM_CHUNKS_PER_SCROLL
        ))
    except Exception as e:
        print(f"  ⚠ Error: {e}")
    
    return samples


def main():
    print("\n" + "="*80)
    print("DOWNLOADING ALL TRAINING DATA")
    print("="*80)
    print(f"\nOutput:")
    print(f"  Images: {OUTPUT_IMAGE_DIR}")
    print(f"  Labels: {OUTPUT_LABEL_DIR}")
    print(f"  CSV: {CSV_PATH}")
    
    # Track existing IDs
    existing_ids = set()
    if CSV_PATH.exists():
        existing_df = pd.read_csv(CSV_PATH)
        existing_ids = set(existing_df['id'].values)
        print(f"\nFound {len(existing_ids)} existing samples")
    
    all_samples = []
    
    # 1. Download labeled data
    labeled_samples = download_labeled_data(existing_ids)
    all_samples.extend(labeled_samples)
    
    # 2. Download unlabeled data
    #unlabeled_samples = download_unlabeled_data(existing_ids)
    #all_samples.extend(unlabeled_samples)
    
    # Save CSV
    if all_samples:
        df = pd.DataFrame(all_samples)
        df.to_csv(CSV_PATH, index=False)
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total: {len(df)} samples")
        print(f"  With labels: {df['has_label'].sum()} (ground truth)")
        print(f"  Without labels: {(~df['has_label']).sum()}")   
        




if __name__ == "__main__":
    main()