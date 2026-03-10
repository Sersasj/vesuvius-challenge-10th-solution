import numpy as np
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = Path("path/to/input")  # Directory containing .npz files
OUTPUT_DIR = None  # Output directory (None = same as input)
REMOVE_ORIGINAL = True  # Remove original .npz files after conversion


def convert_npz_to_npy_float16(input_dir, output_dir=None, remove_original=False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir

    npz_files = list(input_dir.glob('*.npz'))
    if not npz_files:
        print(f"No .npz files found in {input_dir}")
        return

    print(f"Found {len(npz_files)} NPZ files. Extracting to NPY format...")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_original_size = total_new_size = 0

    for npz_path in tqdm(npz_files, desc="Converting"):
        prob = np.load(str(npz_path))['prob']  # Already float16

        npy_path = output_dir / npz_path.name.replace('.npz', '.npy')
        np.save(str(npy_path), prob)

        total_original_size += npz_path.stat().st_size
        total_new_size += npy_path.stat().st_size

        if remove_original:
            npz_path.unlink()

    print(f"Original size: {total_original_size / 1024**3:.2f} GB")
    print(f"New size: {total_new_size / 1024**3:.2f} GB")
    print(f"Space saved: {(total_original_size - total_new_size) / 1024**3:.2f} GB ({(1 - total_new_size/total_original_size)*100:.1f}%)")
    if remove_original:
        print("Original .npz files have been removed")


if __name__ == '__main__':
    convert_npz_to_npy_float16(INPUT_DIR, OUTPUT_DIR, REMOVE_ORIGINAL)

