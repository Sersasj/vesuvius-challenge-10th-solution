import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from typing import Tuple
import scipy.ndimage as ndimage
from skimage.morphology import skeletonize
from monai.transforms import (
    Compose, RandFlipd, RandRotate90d, EnsureTyped, RandShiftIntensityd,
    RandScaleIntensityd, RandAdjustContrastd, RandGaussianNoised,
    RandGaussianSmoothd, RandSimulateLowResolutiond,
    RandCoarseDropoutd, MapTransform,
)

from ..utils.io import load_volume


class RandCoarseDropoutdWithRanges(MapTransform):
    """Coarse dropout with shared holes (applied to all keys) and independent holes (per key).

    Args:
        keys: Keys to apply dropout to
        prob: Probability of applying dropout
        shared_holes_range: Range for number of holes applied to ALL keys (aligned)
        independent_holes_range: Range for number of holes applied to EACH key independently
        spatial_size_range: Range for hole sizes
        fill_value: Value to fill dropped regions
    """
    def __init__(
        self,
        keys,
        prob=1.0,
        shared_holes_range=(10, 15),
        independent_holes_range=(5, 10),
        spatial_size_range=(10, 30),
        fill_value=0.0,
    ):
        super().__init__(keys)
        self.prob = prob
        self.shared_holes_range = shared_holes_range
        self.independent_holes_range = independent_holes_range
        self.spatial_size_range = spatial_size_range
        self.fill_value = fill_value

    def __call__(self, data):
        if np.random.rand() > self.prob:
            return data

        # 1. Apply shared holes to ALL keys (same positions)
        num_shared = np.random.randint(self.shared_holes_range[0], self.shared_holes_range[1] + 1)
        hole_d = np.random.randint(self.spatial_size_range[0], self.spatial_size_range[1] + 1)
        hole_h = np.random.randint(self.spatial_size_range[0], self.spatial_size_range[1] + 1)
        hole_w = np.random.randint(self.spatial_size_range[0], self.spatial_size_range[1] + 1)

        shared_transform = RandCoarseDropoutd(
            keys=self.keys,
            prob=1.0,
            holes=num_shared,
            spatial_size=(hole_d, hole_h, hole_w),
            fill_value=self.fill_value,
        )
        data = shared_transform(data)

        # 2. Apply independent holes to EACH key separately
        for key in self.keys:
            num_indep = np.random.randint(self.independent_holes_range[0], self.independent_holes_range[1] + 1)
            hole_d = np.random.randint(self.spatial_size_range[0], self.spatial_size_range[1] + 1)
            hole_h = np.random.randint(self.spatial_size_range[0], self.spatial_size_range[1] + 1)
            hole_w = np.random.randint(self.spatial_size_range[0], self.spatial_size_range[1] + 1)

            indep_transform = RandCoarseDropoutd(
                keys=[key],
                prob=1.0,
                holes=num_indep,
                spatial_size=(hole_d, hole_h, hole_w),
                fill_value=self.fill_value,
            )
            data = indep_transform(data)

        return data


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
        oof_dirs: List of directories containing OOF predictions (supports 1 or 2 OOF masks)
    """

    def __init__(
        self,
        csv_path,
        image_dir='train_images_npy',
        label_dir='train_labels_npy',
        skeleton_dir='train_skeletons_npy',
        oof_dirs=None,
        patch_size=128,
        is_train=True,
        augment=True,
        normalize=True,
    ):
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.skeleton_dir = Path(skeleton_dir)

        # Handle oof_dirs - support both single dir (str) and multiple dirs (list)
        if oof_dirs is None:
            oof_dirs = ['1st_stage_cache', 'Primus_1st_stage_cache', 'PrimusV2_1st_stage_cache']
        elif isinstance(oof_dirs, str):
            oof_dirs = [oof_dirs]
        self.oof_dirs = [Path(d) for d in oof_dirs]
        self.num_oof_channels = len(self.oof_dirs)

        self.patch_size = patch_size
        self.is_train = is_train
        self.augment = augment and is_train
        self.normalize = normalize

        # Filter out samples without OOF files in ALL directories
        def has_all_oofs(sample_id):
            """Check if OOF files exist for this sample in all OOF directories."""
            for oof_dir in self.oof_dirs:
                # Support both {id}.npy and {id}_probs.npy patterns
                if not (oof_dir / f'{sample_id}.npy').exists() and not (oof_dir / f'{sample_id}_probs.npy').exists():
                    return False
            return True

        initial_count = len(self.df)
        self.df = self.df[self.df['id'].apply(has_all_oofs)].reset_index(drop=True)
        filtered_count = len(self.df)

        if initial_count > filtered_count:
            print(f"Dataset: Filtered {initial_count} -> {filtered_count} samples (removed {initial_count - filtered_count} without OOF files)")

        # Define MONAI transforms
        self.transforms = self._get_transforms()

        print(f"Dataset: {len(self.df)} samples, patch_size={patch_size}, train={is_train}")
        print(f"Dataset: {self.num_oof_channels} OOF channel(s) from: {[str(d) for d in self.oof_dirs]}")

    def _get_transforms(self):
        # Build keys based on number of OOF channels
        oof_keys = [f"oof_{i}" for i in range(self.num_oof_channels)]
        keys = ["image", "label", "skeleton"] + oof_keys
        transforms = []

        if self.is_train and self.augment:
            # Spatial augmentations (applied to all keys to keep them aligned)
            transforms += [
                Compose([RandFlipd(keys=keys, prob=0.5, spatial_axis=i) for i in range(3)] + [RandRotate90d(keys=keys, prob=0.2, max_k=3, spatial_axes=(1, 2))]
            , lazy=True)
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

            # Coarse dropout on OOF masks
            if oof_keys:
                # Large holes (10-30 voxels)
                transforms.append(
                    RandCoarseDropoutdWithRanges(
                        keys=oof_keys,
                        prob=1,
                        shared_holes_range=(10, 15),
                        independent_holes_range=(15, 15),
                        spatial_size_range=(10, 30),
                        fill_value=0.0,
                    )
                )
                # Microscopic holes (1-5 voxels)
                transforms.append(
                    RandCoarseDropoutdWithRanges(
                        keys=oof_keys,
                        prob=0.8,
                        shared_holes_range=(20, 40),
                        independent_holes_range=(30, 60),
                        spatial_size_range=(1, 3),
                        fill_value=0.0,
                    )
                )

        transforms.append(EnsureTyped(keys=keys, track_meta=False))
        return Compose(transforms)

    def __len__(self):
        return len(self.df)

    def _get_paths(self, sample_id):
        """Get image, label, skeleton, and OOF paths for a sample."""
        # Support both {id}.npy and {id}_probs.npy patterns for OOF files
        oof_paths = []
        for oof_dir in self.oof_dirs:
            path = oof_dir / f'{sample_id}.npy'
            if not path.exists():
                path = oof_dir / f'{sample_id}_probs.npy'
            oof_paths.append(path)
        return (self.image_dir / f'{sample_id}.npy',
                self.label_dir / f'{sample_id}.npy',
                self.skeleton_dir / f'{sample_id}.npy',
                oof_paths)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path, label_path, skeleton_path, oof_paths = self._get_paths(row['id'])

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

        # Load all OOF predictions
        oofs = []
        for oof_path in oof_paths:
            if not oof_path.exists():
                raise FileNotFoundError(f"OOF prediction file not found: {oof_path}")
            oofs.append(np.load(str(oof_path)).astype(np.float32))  # Shape: (D, H, W)

        if self.is_train:
            image, label, skeleton, oofs = self._random_crop(image, label, skeleton, oofs)

        image = (image.astype(np.float32) / 255.0) if self.normalize else image.astype(np.float32)

        # Binarize OOF predictions
        if self.is_train:
            threshold = 0.3 + np    .random.uniform(-0.2, 0.5)
        else:
            threshold = 0.3
        oofs_binary = [(oof > threshold).astype(np.float32) for oof in oofs]

        # Apply transforms (image, label, skeleton, and oofs are kept separate)
        # OOF dropout is applied in the transform pipeline
        data_dict = {
            "image": image[None],  # Add channel dim: (1, D, H, W)
            "label": label[None],  # Add channel dim: (1, D, H, W)
            "skeleton": skeleton[None],  # Add channel dim: (1, D, H, W)
        }
        # Add each OOF channel with its key
        for i, oof_binary in enumerate(oofs_binary):
            data_dict[f"oof_{i}"] = oof_binary[None]  # Add channel dim: (1, D, H, W)

        data = self.transforms(data_dict)

        # Stack image and all OOF channels: (1 + num_oof_channels, D, H, W)
        # After EnsureTyped, data is already tensors, so use torch.cat
        oof_tensors = [data[f"oof_{i}"] for i in range(self.num_oof_channels)]
        combined_input = torch.cat([data["image"]] + oof_tensors, dim=0)

        return combined_input, data["label"].squeeze(0).long(), data["skeleton"].squeeze(0).float()


    def _random_crop(self, image: np.ndarray, label: np.ndarray, skeleton: np.ndarray, oofs: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """Extract a patch with smart sampling (70% surface-focused, 30% random).

        Args:
            image: (D, H, W) image array
            label: (D, H, W) label array
            skeleton: (D, H, W) skeleton array
            oofs: List of (D, H, W) OOF arrays

        Returns:
            Cropped image, label, skeleton, and list of cropped OOF arrays
        """
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

        cropped_oofs = [oof[d:d+ps, h:h+ps, w:w+ps] for oof in oofs]
        return (image[d:d+ps, h:h+ps, w:w+ps],
                label[d:d+ps, h:h+ps, w:w+ps],
                skeleton[d:d+ps, h:h+ps, w:w+ps],
                cropped_oofs)


