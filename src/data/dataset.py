import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import random
import math
from typing import Tuple
import scipy.ndimage as ndimage
from skimage import morphology
from skimage.morphology import skeletonize
from monai.transforms import (
    Compose, RandFlipd, RandRotate90d, EnsureTyped, RandShiftIntensityd,
    RandScaleIntensityd, RandAdjustContrastd, RandGaussianNoised,
    RandGaussianSmoothd, RandZoomd, RandSimulateLowResolutiond,
)

from ..utils.io import load_volume


def compute_skeleton_3d(label: np.ndarray) -> np.ndarray:
    """Compute 3D slice-wise skeletonization with smoothing to reduce spurs."""
    # label shape: (D, H, W)
    skel = np.zeros_like(label, dtype=np.uint8)
    for z in range(label.shape[0]):
        mask = label[z] == 1
        if np.any(mask):
         
            smoothed = ndimage.median_filter(mask.astype(np.uint8), size=5).astype(bool)
            
            skel[z] = skeletonize(smoothed).astype(np.uint8)
    return skel


class VesuviusDataset(Dataset):
    """3D patch-based dataset for scroll surface detection.

    Args:
        csv_path: Path to train_split.csv or val_split.csv
        image_dir: Directory containing images (TIF or NPY)
        label_dir: Directory containing labels (TIF or NPY)
        patch_size: Size of 3D patches (default: 128)
        is_train: If True, extract random patches. If False, use center crop.
        augment: Apply augmentations (train only)
        normalize: Normalize images to [0, 1]
        additional_image_dir: Directory for additional/pseudo-labeled images (optional)
        additional_label_dir: Directory for additional/pseudo-labeled labels (optional)
    """

    def __init__(
        self,
        csv_path,
        image_dir='train_images_npy',
        label_dir='train_labels_npy',
        skeleton_dir='train_skeletons_npy',
        patch_size=128,
        is_train=True,
        augment=True,
        normalize=True,
    ):
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.skeleton_dir = Path(skeleton_dir)

        self.patch_size = patch_size
        self.is_train = is_train
        self.augment = augment and is_train
        self.normalize = normalize

        # Define MONAI transforms
        self.transforms = self._get_transforms()

        print(f"Dataset: {len(self.df)} samples, patch_size={patch_size}, train={is_train}")

    def _get_transforms(self):
        keys = ["image", "label", "skeleton"]
        transforms = []

        if self.is_train and self.augment:
            # Spatial augmentations
            transforms += [
                Compose([RandFlipd(keys=keys, prob=0.5, spatial_axis=i) for i in range(3)] + [RandRotate90d(keys=keys, prob=0.2, max_k=3, spatial_axes=(1, 2))]
            , lazy=True)
            ]
            transforms += [
                RandZoomd(keys=keys, prob=0.2, min_zoom=0.7, max_zoom=1.4,
                          mode=["trilinear", "nearest", "nearest"]),
            ]
            # Intensity augmentations (image only)
            transforms += [
                RandGaussianNoised(keys=["image"], prob=0.1, std=0.1),
                RandGaussianSmoothd(keys=["image"], sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0),
                                    sigma_z=(0.5, 1.0), prob=0.2),
                RandScaleIntensityd(keys=["image"], factors=0.25, prob=0.15),
                RandAdjustContrastd(keys=["image"], gamma=(0.75, 1.25), prob=0.15),
                RandSimulateLowResolutiond(keys=["image"], prob=0.25, zoom_range=(0.5, 1.0)),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
            ]

        transforms.append(EnsureTyped(keys=keys, track_meta=False))
        return Compose(transforms)

    def __len__(self):
        return len(self.df)

    def _get_paths(self, sample_id):
        """Get image and label paths for a sample."""
        return self.image_dir / f'{sample_id}.npy', self.label_dir / f'{sample_id}.npy', self.skeleton_dir / f'{sample_id}.npy'

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path, label_path, skeleton_path = self._get_paths(row['id'])

        image = load_volume(str(image_path))
        label = load_volume(str(label_path))
        
        # Load or compute skeleton
        if skeleton_path.exists():
            skeleton = load_volume(str(skeleton_path))
        else:
            # Compute skeleton on-the-fly if not present
            print(f"Computing skeleton on-the-fly for {row['id']} (file not found at {skeleton_path})")
            skeleton = compute_skeleton_3d(label)
            
            # Save computed skeleton to disk for future use
            try:
                skeleton_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(skeleton_path), skeleton)
                print(f"Saved computed skeleton to {skeleton_path}")
            except Exception as e:
                print(f"Warning: Could not save skeleton to {skeleton_path}: {e}")

        if self.is_train:
            image, label, skeleton = self._random_crop(image, label, skeleton)

        image = (image.astype(np.float32) / 255.0) if self.normalize else image.astype(np.float32)

        data = self.transforms({"image": image[None], "label": label[None], "skeleton": skeleton[None]})
        return data["image"], data["label"].squeeze(0).long(), data["skeleton"].squeeze(0).float()

    def _random_crop(self, image: np.ndarray, label: np.ndarray, skeleton: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract a patch with smart sampling (70% surface-focused, 30% random)."""
        D, H, W = image.shape
        ps = self.patch_size

        def random_start(dim): return np.random.randint(0, max(1, dim - ps + 1))

        # 70% chance: sample near surface (where label == 1)
        surface_coords = np.argwhere(label == 1) if np.random.rand() < 0.7 else []
        if len(surface_coords) > 0:
            cd, ch, cw = surface_coords[np.random.randint(len(surface_coords))]
            jitter = ps // 4
            d = np.clip(cd - ps//2 + np.random.randint(-jitter, jitter), 0, max(0, D - ps))
            h = np.clip(ch - ps//2 + np.random.randint(-jitter, jitter), 0, max(0, H - ps))
            w = np.clip(cw - ps//2 + np.random.randint(-jitter, jitter), 0, max(0, W - ps))
        else:
            d, h, w = random_start(D), random_start(H), random_start(W)

        return image[d:d+ps, h:h+ps, w:w+ps], label[d:d+ps, h:h+ps, w:w+ps], skeleton[d:d+ps, h:h+ps, w:w+ps]




class CutMixCollator:
    """Collate function that applies 3D CutMix augmentation."""

    def __init__(self, prob=0.5, max_cube_ratio=0.4):
        self.prob = prob
        self.max_cube_ratio = max_cube_ratio

    def __call__(self, batch):
        images = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])
        skeletons = torch.stack([b[2] for b in batch])

        if len(batch) >= 2:
            images, labels, skeletons = cutmix_3d(images, labels, skeletons, self.prob, self.max_cube_ratio)
        return images, labels, skeletons


def cutmix_3d(images, labels, skeletons, prob=0.5, max_cube_ratio=0.4):
    """Apply 3D CutMix: swap cubic patches between random pairs in batch."""
    B = images.shape[0]
    if B < 2 or random.random() > prob:
        return images, labels, skeletons

    perm = torch.randperm(B, device=images.device)
    spatial = images.shape[2:]
    frac = random.uniform(0.3, max_cube_ratio)
    cube_sizes = [max(1, int(math.ceil(frac * s))) for s in spatial]

    images_out, labels_out, skeletons_out = images.clone(), labels.clone(), skeletons.clone()

    for i in range(B):
        j = perm[i].item()
        if j == i:
            continue

        starts = [random.randint(0, max(0, dim - sz)) for dim, sz in zip(spatial, cube_sizes)]
        img_sl = (slice(None),) + tuple(slice(s, s + sz) for s, sz in zip(starts, cube_sizes))
        lbl_sl = tuple(slice(s, s + sz) for s, sz in zip(starts, cube_sizes))

        images_out[i][img_sl] = images[j][img_sl]
        labels_out[i][lbl_sl] = labels[j][lbl_sl]
        skeletons_out[i][lbl_sl] = skeletons[j][lbl_sl]

    return images_out, labels_out, skeletons_out