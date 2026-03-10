"""I/O utilities for loading volumes and common file operations."""
import numpy as np
from pathlib import Path
from PIL import Image, ImageSequence


def load_volume(path, mmap_mode=None):
    """Load 3D volume from .npy or .tif file.

    Args:
        path: Path to volume file (.npy or .tif)
        mmap_mode: Memory-map mode for .npy files (None, 'r', 'r+', 'w+', 'c')

    Returns:
        np.ndarray: 3D volume array (D, H, W)
    """
    path = str(path)
    if path.endswith('.npy'):
        return np.load(path, mmap_mode=mmap_mode)

    # Handle .tif files
    im = Image.open(path)
    slices = [np.array(page) for page in ImageSequence.Iterator(im)]
    return np.stack(slices, axis=0)
