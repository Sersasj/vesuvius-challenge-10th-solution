from __future__ import annotations

import sys
import os
import argparse

# Set allocator to avoid fragmentation (must be done before CUDA init)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from sklearn.model_selection import KFold

# Ensure repo root is on sys.path so `import src...` works even when running via an absolute script path.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.lightning_module import SegmentationModule
from src_2nd_4th_stages.models.lightning_module import SegmentationModule as SegmentationModule2ndStage
from src.utils.metric import load_volume



def _load_model_with_ema(module_class, ckpt_path: str, device: torch.device):
    """Load a Lightning checkpoint, preferring EMA weights if available."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    has_ema = any(k.startswith('ema.module.') for k in state_dict.keys())

    model = module_class.load_from_checkpoint(ckpt_path, map_location=device, strict=False)

    if has_ema:
        ema_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('ema.module.'):
                new_key = 'model.' + k[len('ema.module.'):]
                ema_state_dict[new_key] = v
        model.load_state_dict(ema_state_dict, strict=False)
        print(f"    Loaded EMA weights from {Path(ckpt_path).name}")

    del checkpoint, state_dict
    model.eval().to(device)
    return model


# ==========================
# Configuration
# ==========================

# Base directory containing fold checkpoints (fold_0/, fold_1/, etc.)
CHECKPOINT_BASE_DIR = Path("1st_stage")

# Pattern to find best checkpoint in each fold directory
# Will glob for files matching this pattern and pick the best one
CHECKPOINT_PATTERN = "best-*.ckpt"

# Checkpoint path dicts — populated via CLI args (--ckpt_pattern, --checkpoint_dir_2nd, etc.)
CHECKPOINT_PATHS_1ST_STAGE = {}
CHECKPOINT_PATHS_2ND_STAGE = {}
CHECKPOINT_PATHS_3RD_STAGE = {}
CHECKPOINT_PATHS_4TH_STAGE = {}

# Main training CSV used to determine fold splits
TRAIN_CSV_PATH = Path("train.csv")
IMAGE_DIR = Path("train_images_npy")
LABEL_DIR = Path("train_labels_npy")

# Fold configuration
N_FOLDS = 5
FOLD_SEED = 42
MAX_SAMPLES = 2000

PATCH_SIZE_1ST_STAGE: Optional[int] = 160
PATCH_SIZE_2ND_STAGE: Optional[int] = 160
PATCH_SIZE_3RD_STAGE: Optional[int] = 160
PATCH_SIZE_4TH_STAGE: Optional[int] = 160
SW_BATCH_SIZE = 1
OVERLAP_1ST_STAGE = 0.5
OVERLAP_2ND_STAGE = 0.5
OVERLAP_3RD_STAGE = 0.5
OVERLAP_4TH_STAGE = 0.5
SW_MODE = "gaussian"
SW_SIGMA_SCALE = 0.125  # Gaussian blending sigma = sigma_scale * roi_dim (used when mode="gaussian")
PADDING_MODE = "reflect"

PRED_METHOD = "threshold"
# Float: same threshold for all OOF channels. Tuple: per-channel thresholds for 2nd stage (one per CACHE_1ST_STAGE_DIRS).
PRED_THRESHOLD_1ST_STAGE: float | Tuple[float, ...] = (0.3, 0.3, 0.3)  # (resEnc, primus, primusV2)
#PRED_THRESHOLD_1ST_STAGE: float | Tuple[float, ...] = (0.6, 0.45, 0.3)  # (resEnc, primus, primusV2)

PRED_THRESHOLD_2ND_STAGE = 0.3
PRED_THRESHOLD_3RD_STAGE = 0.3
PRED_THRESHOLD_4TH_STAGE = 0.3

USE_TTA = True
USE_POST_PROCESSING = True
POST_PROCESS_MIN_CC_VOLUME = 3000

# How many stages to run: 1 = 1st only, ..., 4 = +4th refinement
NUM_STAGES = 4

# 2nd stage specific settings
NUM_ITERATIONS = 1  # Number of refinement iterations for 2nd stage
NUM_ITERATIONS_3RD_STAGE = 2  # Number of refinement iterations for 3rd stage
NUM_ITERATIONS_4TH_STAGE = 1  # Number of refinement iterations for 4th stage
SAVE_PREDS_DIR: Optional[Path] = None
SAVE_2ND_STAGE_PROBS: bool = False  # Save 2nd stage probability masks
SAVE_2ND_STAGE_PROBS_DIR: Optional[Path] = "2nd_stage_probs"  # Directory for 2nd stage probabilities (defaults to SAVE_PREDS_DIR/probs if None)

# 1st stage caching – load cached probabilities from multiple dirs (no inference needed)
# Each dir becomes a separate OOF channel for the 2nd stage input: [image, oof_0, oof_1, ...]
CACHE_1ST_STAGE_DIRS: List[Path] = [
    Path("1st_stage_cache"),           # resEnc OOF
    Path("Primus_1st_stage_cache"),    # primus OOF
    Path("PrimusV2_1st_stage_cache"),  # primusV2 OOF
]
CACHE_1ST_STAGE_ONLY: bool = True  # If True, skip 1st stage inference entirely

# 2nd stage caching – load pre-computed 2nd stage OOF probabilities (skip 1st/2nd inference)
# Set to a directory containing fold_0/, fold_1/, ... with {sample_id}_probs.npy files
CACHE_2ND_STAGE_DIR: Optional[Path] = None  # Set to None to run inference

# 3rd stage caching – load pre-computed 3rd stage OOF probabilities (skip 1st–3rd inference, run 4th stage)
CACHE_3RD_STAGE_DIR: Optional[Path] = None  # Set to None to run inference

# Generic: save the final stage's probability maps to this directory
SAVE_FINAL_PROBS_DIR: Optional[Path] = None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate folds using official metrics")
    parser.add_argument(
        "--folds",
        nargs="+",
        default=["all"],
        help="Folds to evaluate. Use integers (e.g., 0 1 2) or 'all' for all folds"
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=N_FOLDS,
        help=f"Total number of folds (default: {N_FOLDS})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=FOLD_SEED,
        help=f"Random seed for fold splitting (default: {FOLD_SEED})"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(CHECKPOINT_BASE_DIR),
        help=f"Base directory containing fold checkpoints (default: {CHECKPOINT_BASE_DIR})"
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default=str(TRAIN_CSV_PATH),
        help=f"Path to training CSV (default: {TRAIN_CSV_PATH})"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=MAX_SAMPLES,
        help=f"Maximum samples per fold to evaluate (default: {MAX_SAMPLES})"
    )
    parser.add_argument(
        "--num_stages",
        type=int,
        default=None,
        help="Number of stages to run (1-4). Overrides NUM_STAGES global."
    )
    parser.add_argument(
        "--save_preds_dir",
        type=str,
        default=None,
        help="Directory to save OOF predictions. Overrides SAVE_PREDS_DIR global."
    )
    parser.add_argument(
        "--ckpt_pattern",
        type=str,
        default=None,
        help="Glob pattern for 1st stage checkpoints within --checkpoint_dir, e.g. 'fold_{fold}/best*.ckpt'. "
             "Use {fold} placeholder for fold index. Overrides CHECKPOINT_PATHS_1ST_STAGE."
    )
    parser.add_argument(
        "--checkpoint_dir_2nd",
        type=str,
        default=None,
        help="Directory with fold_0/, fold_1/, ... containing 2nd stage checkpoints. Auto-discovers best*.ckpt."
    )
    parser.add_argument(
        "--checkpoint_dir_3rd",
        type=str,
        default=None,
        help="Directory with fold_0/, fold_1/, ... containing 3rd stage checkpoints."
    )
    parser.add_argument(
        "--checkpoint_dir_4th",
        type=str,
        default=None,
        help="Directory with fold_0/, fold_1/, ... containing 4th stage checkpoints."
    )
    parser.add_argument(
        "--oof_dirs",
        nargs="+",
        default=None,
        help="Override CACHE_1ST_STAGE_DIRS: list of OOF cache directories for 1st stage."
    )
    parser.add_argument(
        "--cache_2nd_stage_dir",
        type=str,
        default=None,
        help="Load pre-computed 2nd stage OOF probs from this directory (skip 1st+2nd stage inference)."
    )
    parser.add_argument(
        "--cache_3rd_stage_dir",
        type=str,
        default=None,
        help="Load pre-computed 3rd stage OOF probs from this directory (skip 1st–3rd stage inference)."
    )
    parser.add_argument(
        "--save_probs_dir",
        type=str,
        default=None,
        help="Save probability maps from the final stage to this directory."
    )
    return parser.parse_args()


def _discover_fold_checkpoints(base_dir: str, n_folds: int) -> dict:
    """Auto-discover best checkpoints from fold_0/, fold_1/, ... directories."""
    ckpt_dir = Path(base_dir)
    result = {}
    for fold_idx in range(n_folds):
        fold_path = ckpt_dir / f"fold_{fold_idx}"
        if not fold_path.exists():
            continue
        matches = sorted(fold_path.glob("best*.ckpt"))
        if not matches:
            matches = sorted(fold_path.glob("*.ckpt"))
        if matches:
            # Pick the one with highest val_dice if available
            best = matches[-1]
            for m in matches:
                if "val_dice" in m.name:
                    best = m
                    break
            result[f"fold{fold_idx}"] = [str(best)]
            print(f"  fold{fold_idx}: {best.name}")
        else:
            print(f"  fold{fold_idx}: no checkpoints found in {fold_path}")
    return result


def _save_stage_probs(stage_num: int, fold_idx: int, sample_id: str, probs: np.ndarray):
    """Save probability maps if SAVE_FINAL_PROBS_DIR is set and this is the target stage."""
    if SAVE_FINAL_PROBS_DIR is None:
        return
    if NUM_STAGES != stage_num:
        return
    save_dir = Path(SAVE_FINAL_PROBS_DIR) / f"fold_{fold_idx}"
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / f"{sample_id}_probs.npy", probs.astype(np.float16))
    print(f"      Saved stage {stage_num} probs to fold_{fold_idx}/{sample_id}_probs.npy")


def get_folds(df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Split dataset into K folds."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(df.iloc[train_idx], df.iloc[val_idx]) for train_idx, val_idx in kf.split(df)]



import cupy as cp
import cupyx.scipy.ndimage as cndi
import numpy as np


def postprocess_mask_voxel(mask, min_cc_volume=3000, median_iters=7, verbose=True):
    """
    GPU version:
    1. Remove CC < min_cc_volume
    2. Apply iterative 3x3x3 median per surviving CC
    3. Reassemble volume
    Returns stats for compatibility.
    """

    if mask.sum() == 0:
        if verbose:
            return mask, {
                "original_cc": 0,
                "final_cc": 0,
                "removed_cc": 0
            }
        return mask

    try:
        # Move mask to GPU
        mask_gpu = cp.asarray(mask).astype(bool)

        # Count original CCs (GPU)
        labeled_before, n_before = cndi.label(mask_gpu)

        final_mask = cp.zeros_like(mask_gpu, dtype=bool)

        # Process each CC
        for cc_id in range(1, int(n_before) + 1):
            component = (labeled_before == cc_id)
            comp_size = int(component.sum())

            if comp_size < min_cc_volume:
                continue

            comp_uint8 = component.astype(cp.uint8)

            # Median smoothing
            for _ in range(median_iters):
                comp_uint8 = (
                    cndi.median_filter(comp_uint8, size=3) > 0
                ).astype(cp.uint8)

            final_mask |= comp_uint8.astype(bool)

        # Count final CCs
        labeled_after, n_after = cndi.label(final_mask)

        # Move result back to CPU
        final_mask_cpu = cp.asnumpy(final_mask).astype(np.uint8)

        if verbose:
            return final_mask_cpu, {
                "original_cc": int(n_before),
                "final_cc": int(n_after),
                "removed_cc": max(0, int(n_before) - int(n_after))
            }

        return final_mask_cpu

    except Exception as e:
        import traceback
        traceback.print_exc()

        if verbose:
            return mask, {
                "original_cc": 0,
                "final_cc": 0,
                "removed_cc": 0,
                "error": str(e)
            }

        return mask

def apply_tta_transform(volume: np.ndarray, flip_dims: Optional[list] = None, rotation_k: int = 0) -> np.ndarray:
    """Apply TTA transform to volume."""
    vol_torch = torch.from_numpy(volume.copy())
    dim_map = {2: 0, 3: 1, 4: 2}
    if flip_dims:
        for dim_5d in flip_dims:
            if dim_5d in dim_map:
                vol_torch = torch.flip(vol_torch, dims=[dim_map[dim_5d]])
    if rotation_k > 0:
        vol_torch = torch.rot90(vol_torch, k=rotation_k, dims=[1, 2])
    return vol_torch.numpy()


def reverse_tta_transform(prediction: np.ndarray, flip_dims: Optional[list] = None, rotation_k: int = 0) -> np.ndarray:
    """Reverse TTA transformation."""
    pred = torch.from_numpy(prediction.copy())
    dim_map = {2: 0, 3: 1, 4: 2}
    if rotation_k > 0:
        pred = torch.rot90(pred, k=(4 - rotation_k), dims=[1, 2])
    if flip_dims:
        for dim_5d in flip_dims:
            if dim_5d in dim_map:
                pred = torch.flip(pred, dims=[dim_map[dim_5d]])
    return pred.numpy()


def get_tta_transforms():
    """Return list of (flip_dims, rotation_k) tuples for TTA."""
    return [
        (None, 0),   # Original
        ([2], 0),      # Flip D
        ([3], 0),      # Flip H
        ([4], 0),      # Flip W
        #([2, 3], 0),   # Flip D+H
        #([2, 4], 0),   # Flip D+W
        #(None, 1),   # Rotate 90°
    ]



@torch.no_grad()
def predict_volume(model: SegmentationModule, image_3d: np.ndarray, *, device: torch.device, roi_size: Tuple[int, int, int], sw_batch_size: int, overlap: float, mode: str, return_probs: bool = False) -> np.ndarray:
    """Returns binary mask (D,H,W) or class 1 probabilities if return_probs=True."""
    x = torch.from_numpy(image_3d).float().unsqueeze(0).unsqueeze(0).to(device)
    if x.max() > 0:
        x = x / 255.0

    with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")):
        out = sliding_window_inference(inputs=x, roi_size=roi_size, sw_batch_size=sw_batch_size, predictor=model, overlap=overlap, mode=mode, sigma_scale=SW_SIGMA_SCALE, padding_mode=PADDING_MODE, progress=False)

    if isinstance(out, (list, tuple)):
        out = out[0]
    if out.shape[2:] != x.shape[2:]:
        out = F.interpolate(out, size=x.shape[2:], mode='nearest', align_corners=False)

    if return_probs:
        return F.softmax(out, dim=1)[:, 1][0].detach().cpu().numpy()

    if PRED_METHOD == "threshold":
        probs = F.softmax(out, dim=1)
        thresh = PRED_THRESHOLD_1ST_STAGE[0] if isinstance(PRED_THRESHOLD_1ST_STAGE, tuple) else PRED_THRESHOLD_1ST_STAGE
        return (probs[:, 1] > thresh)[0].detach().cpu().numpy().astype(np.uint8, copy=False)
    return torch.argmax(out, dim=1)[0].detach().cpu().numpy().astype(np.uint8, copy=False)


def _find_file(base_dir: Path, sample_id: str, extensions: tuple = (".npy", ".tif")) -> Path:
    """Find file with given extensions."""
    for ext in extensions:
        path = base_dir / f"{sample_id}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No file found for {sample_id} in {base_dir}")



def _load_4th_stage_refinement_models(fold_key: str, device: torch.device):
    """Load 4th stage refinement model(s) (SegmentationModule2ndStage). Returns list of models or []."""
    if fold_key not in CHECKPOINT_PATHS_4TH_STAGE:
        return []

    ckpt_raw = CHECKPOINT_PATHS_4TH_STAGE[fold_key]
    ckpt_paths = [Path(p) for p in (ckpt_raw if isinstance(ckpt_raw, list) else [ckpt_raw])]

    models_4th = []
    for i, ckpt_path in enumerate(ckpt_paths):
        if not ckpt_path.exists():
            print(f"WARNING: 4th stage checkpoint not found: {ckpt_path}")
            return []
        print(f"  Loading 4th stage refinement model {i+1}/{len(ckpt_paths)}: {ckpt_path.name}")
        m = _load_model_with_ema(SegmentationModule2ndStage, str(ckpt_path), device)
        models_4th.append(m)
    return models_4th


def evaluate_fold(
    fold_idx: int,
    val_df: pd.DataFrame,
    device: torch.device,
    max_samples: int = MAX_SAMPLES,
) -> pd.DataFrame:
    """Evaluate a single fold with multi-stage inference and return results DataFrame.

    Supports three modes:
    1. CACHE_3RD_STAGE_DIR set: load pre-computed 3rd stage OOF probs, run 4th stage
    2. CACHE_2ND_STAGE_DIR set: load pre-computed 2nd stage OOF probs, run 3rd–4th stages
    3. CACHE_1ST_STAGE_ONLY: load cached 1st stage probs, run 2nd–4th stage inference
    4. Full: run all stages from scratch
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Fold {fold_idx} ({NUM_STAGES}-Stage)")
    print(f"{'='*60}")

    fold_key = f"fold{fold_idx}"

    # ==================
    # MODE 1c: Load cached 3rd stage OOF probs (run 4th stage only)
    # ==================
    if CACHE_3RD_STAGE_DIR is not None:
        cache_3rd_dir = Path(CACHE_3RD_STAGE_DIR)
        cache_3rd_fold_dir = cache_3rd_dir / f"fold_{fold_idx}"
        print(f"CACHE_3RD_STAGE mode – loading from {cache_3rd_dir}, running 4th stage")

        # Load 4th stage refinement model(s)
        models_4th = _load_4th_stage_refinement_models(fold_key, device)
        if not models_4th:
            print(f"  No 4th stage models for {fold_key}, skipping")
            return
        patch_size_4th = PATCH_SIZE_4TH_STAGE or 128
        roi_size_4th = (patch_size_4th, patch_size_4th, patch_size_4th)

        val_df = val_df.head(max_samples)
        print(f"Evaluating {len(val_df)} samples for fold {fold_idx}")

        for sample_idx, (_, row) in enumerate(val_df.iterrows()):
            sample_id = row["id"]
            print(f"    [{sample_idx+1}/{len(val_df)}] Processing: {sample_id}")

            # Load cached 3rd stage probs
            probs = None
            for try_dir in [cache_3rd_dir, cache_3rd_fold_dir]:
                cache_file = try_dir / f"{sample_id}_probs.npy"
                if cache_file.exists():
                    probs = np.load(cache_file).astype(np.float32)
                    break

            if probs is None:
                print(f"    SKIP: No cached 3rd stage probs for {sample_id}")
                continue

            pred = (probs > PRED_THRESHOLD_3RD_STAGE).astype(np.uint8)
            del probs

            image = np.asarray(load_volume(_find_file(IMAGE_DIR, sample_id)))

            # 4TH STAGE: Refine with [image, 3rd stage mask]
            image_normalized_4th = image.astype(np.float32) / 255.0
            current_mask_4th = pred.astype(np.float32)

            for iter_idx_4th in range(NUM_ITERATIONS_4TH_STAGE):
                image_multichannel_4th = np.stack([image_normalized_4th, current_mask_4th], axis=0)

                ensemble_probs_4th = None
                for model_idx, model_4th in enumerate(models_4th):
                    print(f"      4th stage model {model_idx+1}/{len(models_4th)} (iter {iter_idx_4th+1}/{NUM_ITERATIONS_4TH_STAGE})")

                    x = torch.from_numpy(image_multichannel_4th).float().unsqueeze(0)
                    torch.cuda.empty_cache()

                    with torch.inference_mode(), torch.autocast(
                        device_type="cuda" if device.type == "cuda" else "cpu",
                        enabled=(device.type == "cuda")
                    ):
                        out = sliding_window_inference(
                            inputs=x, roi_size=roi_size_4th, sw_batch_size=SW_BATCH_SIZE,
                            predictor=model_4th, overlap=OVERLAP_4TH_STAGE, mode=SW_MODE,
                            sigma_scale=SW_SIGMA_SCALE, padding_mode=PADDING_MODE, progress=False,
                            sw_device=device, device='cpu'
                        )

                    del x
                    torch.cuda.empty_cache()

                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    if out.shape[2:] != image.shape:
                        out = F.interpolate(out, size=image.shape, mode='trilinear', align_corners=False)

                    model_probs_4th = F.softmax(out, dim=1)[:, 1][0].detach().cpu().numpy()
                    del out
                    torch.cuda.empty_cache()

                    if ensemble_probs_4th is None:
                        ensemble_probs_4th = model_probs_4th / len(models_4th)
                    else:
                        ensemble_probs_4th += model_probs_4th / len(models_4th)
                    del model_probs_4th
                    torch.cuda.empty_cache()

                del image_multichannel_4th
                if iter_idx_4th == NUM_ITERATIONS_4TH_STAGE - 1:
                    _save_stage_probs(4, fold_idx, sample_id, ensemble_probs_4th)
                pred = (ensemble_probs_4th > PRED_THRESHOLD_4TH_STAGE).astype(np.uint8)
                del ensemble_probs_4th
                torch.cuda.empty_cache()

                if iter_idx_4th < NUM_ITERATIONS_4TH_STAGE - 1:
                    current_mask_4th = pred.astype(np.float32)

            del image_normalized_4th, current_mask_4th, image, pred
            torch.cuda.empty_cache()

        del models_4th
        torch.cuda.empty_cache()
        return

    # ==================
    # MODE 1b: Load cached 2nd stage OOF probs (run 3rd, 4th)
    # ==================
    if CACHE_2ND_STAGE_DIR is not None:
        cache_2nd_dir = Path(CACHE_2ND_STAGE_DIR)
        # Also try fold subdirectory and probs subdirectory
        cache_2nd_fold_dir = cache_2nd_dir / f"fold_{fold_idx}"
        cache_2nd_probs_dir = cache_2nd_fold_dir / "probs"
        print(f"CACHE_2ND_STAGE mode – loading from {cache_2nd_dir}")

        # Load 3rd stage model(s)
        models_3rd = []
        if NUM_STAGES >= 3 and fold_key in CHECKPOINT_PATHS_3RD_STAGE:
            ckpt_paths_3rd_raw = CHECKPOINT_PATHS_3RD_STAGE[fold_key]
            ckpt_paths_3rd = [Path(p) for p in (ckpt_paths_3rd_raw if isinstance(ckpt_paths_3rd_raw, list) else [ckpt_paths_3rd_raw])]
            for i, ckpt_path_3rd in enumerate(ckpt_paths_3rd):
                if not ckpt_path_3rd.exists():
                    print(f"WARNING: 3rd stage checkpoint not found: {ckpt_path_3rd}")
                    return pd.DataFrame()
                model_3rd = _load_model_with_ema(SegmentationModule2ndStage, str(ckpt_path_3rd), device)
                models_3rd.append(model_3rd)
                print(f"3rd stage model {i+1}/{len(ckpt_paths_3rd)} loaded")

        patch_size_3rd = PATCH_SIZE_3RD_STAGE or 128
        roi_size_3rd = (patch_size_3rd, patch_size_3rd, patch_size_3rd)

        # Load 4th stage refinement model(s)
        models_4th = []
        if NUM_STAGES >= 4:
            models_4th = _load_4th_stage_refinement_models(fold_key, device)
        patch_size_4th = PATCH_SIZE_4TH_STAGE or 128
        roi_size_4th = (patch_size_4th, patch_size_4th, patch_size_4th)

        val_df = val_df.head(max_samples)
        print(f"Evaluating {len(val_df)} samples for fold {fold_idx}")

        for sample_idx, (_, row) in enumerate(val_df.iterrows()):
            sample_id = row["id"]
            print(f"    [{sample_idx+1}/{len(val_df)}] Processing: {sample_id}")

            # Try to find cached probs: flat dir first, then fold subdir, then probs subdir
            probs = None
            for try_dir in [cache_2nd_dir, cache_2nd_fold_dir, cache_2nd_probs_dir]:
                cache_file = try_dir / f"{sample_id}_probs.npy"
                if cache_file.exists():
                    print(f"    Loading cached 2nd stage probs from {cache_file}")
                    probs = np.load(cache_file).astype(np.float32)
                    break

            if probs is None:
                print(f"    SKIP: No cached 2nd stage probs found for {sample_id}")
                continue

            pred = (probs > PRED_THRESHOLD_2ND_STAGE).astype(np.uint8)
            del probs

            # Load image for 3rd/4th stage
            image = np.asarray(load_volume(_find_file(IMAGE_DIR, sample_id)))

            # 3RD STAGE: Refine with [image, refineV2 mask]
            if models_3rd:
                image_normalized_3rd = image.astype(np.float32) / 255.0
                current_mask_3rd = pred.astype(np.float32)

                for iter_idx_3rd in range(NUM_ITERATIONS_3RD_STAGE):
                    image_multichannel_3rd = np.stack([image_normalized_3rd, current_mask_3rd], axis=0)

                    ensemble_probs_3rd = None
                    for model_idx, model_3rd in enumerate(models_3rd):
                        print(f"      3rd stage model {model_idx+1}/{len(models_3rd)} (iter {iter_idx_3rd+1}/{NUM_ITERATIONS_3RD_STAGE})")

                        x = torch.from_numpy(image_multichannel_3rd).float().unsqueeze(0)
                        torch.cuda.empty_cache()

                        with torch.inference_mode(), torch.autocast(
                            device_type="cuda" if device.type == "cuda" else "cpu",
                            enabled=(device.type == "cuda")
                        ):
                            out = sliding_window_inference(
                                inputs=x, roi_size=roi_size_3rd, sw_batch_size=SW_BATCH_SIZE,
                                predictor=model_3rd, overlap=OVERLAP_3RD_STAGE, mode=SW_MODE,
                                sigma_scale=SW_SIGMA_SCALE, padding_mode=PADDING_MODE, progress=False,
                                sw_device=device, device='cpu'
                            )

                        del x
                        torch.cuda.empty_cache()

                        if isinstance(out, (list, tuple)):
                            out = out[0]
                        if out.shape[2:] != image.shape:
                            out = F.interpolate(out, size=image.shape, mode='trilinear', align_corners=False)

                        model_probs_3rd = F.softmax(out, dim=1)[:, 1][0].detach().cpu().numpy()
                        del out
                        torch.cuda.empty_cache()

                        if ensemble_probs_3rd is None:
                            ensemble_probs_3rd = model_probs_3rd / len(models_3rd)
                        else:
                            ensemble_probs_3rd += model_probs_3rd / len(models_3rd)
                        del model_probs_3rd
                        torch.cuda.empty_cache()

                    del image_multichannel_3rd
                    if iter_idx_3rd == NUM_ITERATIONS_3RD_STAGE - 1:
                        _save_stage_probs(3, fold_idx, sample_id, ensemble_probs_3rd)
                    pred = (ensemble_probs_3rd > PRED_THRESHOLD_3RD_STAGE).astype(np.uint8)
                    del ensemble_probs_3rd
                    torch.cuda.empty_cache()

                    if iter_idx_3rd < NUM_ITERATIONS_3RD_STAGE - 1:
                        current_mask_3rd = pred.astype(np.float32)

                del image_normalized_3rd, current_mask_3rd
                torch.cuda.empty_cache()

            # 4TH STAGE: Refine with [image, 3rd stage mask] - 2 iterations
            if models_4th:
                image_normalized_4th = image.astype(np.float32) / 255.0
                current_mask_4th = pred.astype(np.float32)

                for iter_idx_4th in range(NUM_ITERATIONS_4TH_STAGE):
                    image_multichannel_4th = np.stack([image_normalized_4th, current_mask_4th], axis=0)

                    ensemble_probs_4th = None
                    for model_idx, model_4th in enumerate(models_4th):
                        print(f"      4th stage model {model_idx+1}/{len(models_4th)} (iter {iter_idx_4th+1}/{NUM_ITERATIONS_4TH_STAGE})")

                        x = torch.from_numpy(image_multichannel_4th).float().unsqueeze(0)
                        torch.cuda.empty_cache()

                        with torch.inference_mode(), torch.autocast(
                            device_type="cuda" if device.type == "cuda" else "cpu",
                            enabled=(device.type == "cuda")
                        ):
                            out = sliding_window_inference(
                                inputs=x, roi_size=roi_size_4th, sw_batch_size=SW_BATCH_SIZE,
                                predictor=model_4th, overlap=OVERLAP_4TH_STAGE, mode=SW_MODE,
                                sigma_scale=SW_SIGMA_SCALE, padding_mode=PADDING_MODE, progress=False,
                                sw_device=device, device='cpu'
                            )

                        del x
                        torch.cuda.empty_cache()

                        if isinstance(out, (list, tuple)):
                            out = out[0]
                        if out.shape[2:] != image.shape:
                            out = F.interpolate(out, size=image.shape, mode='trilinear', align_corners=False)

                        model_probs_4th = F.softmax(out, dim=1)[:, 1][0].detach().cpu().numpy()
                        del out
                        torch.cuda.empty_cache()

                        if ensemble_probs_4th is None:
                            ensemble_probs_4th = model_probs_4th / len(models_4th)
                        else:
                            ensemble_probs_4th += model_probs_4th / len(models_4th)
                        del model_probs_4th
                        torch.cuda.empty_cache()

                    del image_multichannel_4th
                    if iter_idx_4th == NUM_ITERATIONS_4TH_STAGE - 1:
                        _save_stage_probs(4, fold_idx, sample_id, ensemble_probs_4th)
                    pred = (ensemble_probs_4th > PRED_THRESHOLD_4TH_STAGE).astype(np.uint8)
                    del ensemble_probs_4th
                    torch.cuda.empty_cache()

                    if iter_idx_4th < NUM_ITERATIONS_4TH_STAGE - 1:
                        current_mask_4th = pred.astype(np.float32)

                del image_normalized_4th, current_mask_4th
                torch.cuda.empty_cache()

            del image, pred
            torch.cuda.empty_cache()

        del models_3rd, models_4th
        torch.cuda.empty_cache()
        return

    # ==================
    # MODE 2 & 3: Need model(s) for inference
    # ==================
    models_1st = []
    ckpt_paths_1st = []
    if not CACHE_1ST_STAGE_ONLY:
        # 1st stage checkpoint(s) - support both single checkpoint and ensemble
        if fold_key not in CHECKPOINT_PATHS_1ST_STAGE:
            print(f"WARNING: 1st stage checkpoint not configured for {fold_key}")
            return pd.DataFrame()

        ckpt_paths_1st_raw = CHECKPOINT_PATHS_1ST_STAGE[fold_key]
        if isinstance(ckpt_paths_1st_raw, str):
            ckpt_paths_1st = [Path(ckpt_paths_1st_raw)]
        else:
            ckpt_paths_1st = [Path(p) for p in ckpt_paths_1st_raw]

        for ckpt_path in ckpt_paths_1st:
            if not ckpt_path.exists():
                print(f"WARNING: 1st stage checkpoint not found: {ckpt_path}")
                return pd.DataFrame()
    else:
        print(f"CACHE_1ST_STAGE_ONLY mode – loading from {len(CACHE_1ST_STAGE_DIRS)} cache dir(s):")
        for cd in CACHE_1ST_STAGE_DIRS:
            print(f"  - {cd}")

    # 2nd stage checkpoint(s) - only needed if NUM_STAGES >= 2
    ckpt_paths_2nd = []
    if NUM_STAGES >= 2:
        if fold_key not in CHECKPOINT_PATHS_2ND_STAGE:
            print(f"WARNING: 2nd stage checkpoint not configured for {fold_key}")
            return pd.DataFrame()

        ckpt_paths_2nd_raw = CHECKPOINT_PATHS_2ND_STAGE[fold_key]
        # Normalize to list
        if isinstance(ckpt_paths_2nd_raw, str):
            ckpt_paths_2nd = [Path(ckpt_paths_2nd_raw)]
        else:
            ckpt_paths_2nd = [Path(p) for p in ckpt_paths_2nd_raw]

        # Validate all 2nd stage checkpoints exist
        for ckpt_path in ckpt_paths_2nd:
            if not ckpt_path.exists():
                print(f"WARNING: 2nd stage checkpoint not found: {ckpt_path}")
                return pd.DataFrame()

    if not CACHE_1ST_STAGE_ONLY:
        print(f"1st stage checkpoints ({len(ckpt_paths_1st)}):")
        for ckpt_path in ckpt_paths_1st:
            print(f"  - {ckpt_path}")
    if NUM_STAGES >= 2:
        print(f"2nd stage checkpoints ({len(ckpt_paths_2nd)}):")
        for ckpt_path in ckpt_paths_2nd:
            print(f"  - {ckpt_path}")

    # Load 1st stage model(s) (skip if cache-only)
    if not CACHE_1ST_STAGE_ONLY:
        for i, ckpt_path_1st in enumerate(ckpt_paths_1st):
            model_1st = _load_model_with_ema(SegmentationModule, str(ckpt_path_1st), device)
            models_1st.append(model_1st)
            print(f"1st stage model {i+1}/{len(ckpt_paths_1st)} loaded")

    # Load 2nd stage model(s)
    models_2nd = []
    if NUM_STAGES >= 2:
        for i, ckpt_path_2nd in enumerate(ckpt_paths_2nd):
            model_2nd = _load_model_with_ema(SegmentationModule2ndStage, str(ckpt_path_2nd), device)
            models_2nd.append(model_2nd)
            print(f"2nd stage model {i+1}/{len(ckpt_paths_2nd)} loaded")

    # Load 3rd stage model(s)
    models_3rd = []
    if NUM_STAGES >= 3:
        if fold_key not in CHECKPOINT_PATHS_3RD_STAGE:
            print(f"WARNING: 3rd stage checkpoint not configured for {fold_key}")
            return pd.DataFrame()
        ckpt_paths_3rd_raw = CHECKPOINT_PATHS_3RD_STAGE[fold_key]
        ckpt_paths_3rd = [Path(p) for p in (ckpt_paths_3rd_raw if isinstance(ckpt_paths_3rd_raw, list) else [ckpt_paths_3rd_raw])]
        for ckpt_path in ckpt_paths_3rd:
            if not ckpt_path.exists():
                print(f"WARNING: 3rd stage checkpoint not found: {ckpt_path}")
                return pd.DataFrame()
        for i, ckpt_path_3rd in enumerate(ckpt_paths_3rd):
            model_3rd = _load_model_with_ema(SegmentationModule2ndStage, str(ckpt_path_3rd), device)
            models_3rd.append(model_3rd)
            print(f"3rd stage model {i+1}/{len(ckpt_paths_3rd)} loaded")

    # Load 4th stage refinement model(s)
    models_4th = []
    if NUM_STAGES >= 4:
        models_4th = _load_4th_stage_refinement_models(fold_key, device)

    patch_size_1st = PATCH_SIZE_1ST_STAGE or 128
    patch_size_2nd = PATCH_SIZE_2ND_STAGE or 128
    patch_size_3rd = PATCH_SIZE_3RD_STAGE or 128
    patch_size_4th = PATCH_SIZE_4TH_STAGE or 128
    roi_size_1st = (patch_size_1st, patch_size_1st, patch_size_1st)
    roi_size_2nd = (patch_size_2nd, patch_size_2nd, patch_size_2nd)
    roi_size_3rd = (patch_size_3rd, patch_size_3rd, patch_size_3rd)
    roi_size_4th = (patch_size_4th, patch_size_4th, patch_size_4th)
    print(f"Sliding window roi_size_1st={roi_size_1st}, roi_size_2nd={roi_size_2nd}, roi_size_3rd={roi_size_3rd}, roi_size_4th={roi_size_4th}")

    # Limit samples
    val_df = val_df.head(max_samples)
    print(f"Evaluating {len(val_df)} samples for fold {fold_idx}")

    save_dir = Path(SAVE_PREDS_DIR) / f"fold_{fold_idx}" if SAVE_PREDS_DIR else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Setup directory for 2nd stage probability masks
    save_probs_dir = None
    if SAVE_2ND_STAGE_PROBS:
        if SAVE_2ND_STAGE_PROBS_DIR:
            save_probs_dir = Path(SAVE_2ND_STAGE_PROBS_DIR) / f"fold_{fold_idx}"
        elif save_dir:
            save_probs_dir = save_dir / "probs"
        if save_probs_dir:
            save_probs_dir.mkdir(parents=True, exist_ok=True)
            print(f"2nd stage probabilities will be saved to: {save_probs_dir}")

    for sample_idx, (_, row) in enumerate(val_df.iterrows()):
        sample_id = row["id"]
        print(f"    [{sample_idx+1}/{len(val_df)}] Processing: {sample_id}")

        if NUM_STAGES >= 2 or not CACHE_1ST_STAGE_ONLY:
            image = np.asarray(load_volume(_find_file(IMAGE_DIR, sample_id)))

        # ==================
        # 1ST STAGE: Get probability predictions per cache dir (each becomes an OOF channel)
        # ==================
        oof_channels = []  # List of binary OOF arrays, one per cache dir

        if CACHE_1ST_STAGE_DIRS:
            for cache_d in CACHE_1ST_STAGE_DIRS:
                # Try fold-specific subdirectory first, then flat
                cache_file = Path(cache_d) / f"fold_{fold_idx}" / f"{sample_id}_probs.npy"
                if not cache_file.exists():
                    cache_file = Path(cache_d) / f"{sample_id}_probs.npy"
                if cache_file.exists():
                    print(f"    Loading cached probs from {cache_file}")
                    probs_ch = np.load(cache_file).astype(np.float32)
                    ch_idx = len(oof_channels)
                    thresh = PRED_THRESHOLD_1ST_STAGE[min(ch_idx, len(PRED_THRESHOLD_1ST_STAGE) - 1)] if isinstance(PRED_THRESHOLD_1ST_STAGE, tuple) else PRED_THRESHOLD_1ST_STAGE
                    oof_channels.append((probs_ch > thresh).astype(np.float32))
                    del probs_ch
                else:
                    print(f"    WARNING: Cache file not found: {cache_d}/{sample_id}_probs.npy")

        if len(oof_channels) == 0 and CACHE_1ST_STAGE_ONLY:
            print(f"    SKIP: No cached probs found for {sample_id}")
            continue

        if len(oof_channels) == 0:
            # Fallback: run 1st stage inference (single OOF channel)
            ensemble_probs_1st = None
            for model_idx, model_1st in enumerate(models_1st):
                print(f"      1st stage model {model_idx+1}/{len(models_1st)}")
                if USE_TTA:
                    transforms = get_tta_transforms()
                    model_avg_probs = None
                    for flip_dims, rotation_k in transforms:
                        probs_transformed = predict_volume(
                            model_1st, apply_tta_transform(image, flip_dims, rotation_k),
                            device=device, roi_size=roi_size_1st, sw_batch_size=SW_BATCH_SIZE,
                            overlap=OVERLAP_1ST_STAGE, mode=SW_MODE, return_probs=True
                        )
                        probs_reversed = reverse_tta_transform(probs_transformed, flip_dims, rotation_k)
                        if model_avg_probs is None:
                            model_avg_probs = probs_reversed / len(transforms)
                        else:
                            model_avg_probs += probs_reversed / len(transforms)
                        del probs_transformed, probs_reversed
                    torch.cuda.empty_cache()
                else:
                    model_avg_probs = predict_volume(
                        model_1st, image, device=device, roi_size=roi_size_1st,
                        sw_batch_size=SW_BATCH_SIZE, overlap=OVERLAP_1ST_STAGE, mode=SW_MODE, return_probs=True
                    )

                if ensemble_probs_1st is None:
                    ensemble_probs_1st = model_avg_probs / len(models_1st)
                else:
                    ensemble_probs_1st += model_avg_probs / len(models_1st)
                del model_avg_probs
                torch.cuda.empty_cache()

            oof_channels.append(ensemble_probs_1st)
            del ensemble_probs_1st

        print(f"    {len(oof_channels)} OOF channel(s)")

        if NUM_STAGES == 1:
            # ==================
            # STAGE 1 ONLY: Save probabilities and/or binary predictions
            # ==================
            if len(oof_channels) == 1:
                probs_1st = oof_channels[0]
            else:
                probs_1st = np.mean(np.stack(oof_channels, axis=0), axis=0)

            # Save probability map for downstream stages
            if save_dir:
                np.save(save_dir / f"{sample_id}_probs.npy", probs_1st.astype(np.float16))
                print(f"    Saved probs to {save_dir / f'{sample_id}_probs.npy'}")

            thresh = PRED_THRESHOLD_1ST_STAGE[0] if isinstance(PRED_THRESHOLD_1ST_STAGE, tuple) else PRED_THRESHOLD_1ST_STAGE
            pred = (probs_1st > thresh).astype(np.uint8)
            del probs_1st, oof_channels
            torch.cuda.empty_cache()
        else:
            # ==================
            # 2ND STAGE: Refine predictions with ensemble
            # ==================
            image_normalized = image.astype(np.float32) / 255.0
            # current_oofs: list of binary OOF arrays (one per channel)
            current_oofs = oof_channels
            del oof_channels

            pred = None
            for iter_idx in range(NUM_ITERATIONS):
                # Stack image + all OOF channels: (1 + num_oof_channels, D, H, W)
                image_multichannel = np.stack([image_normalized] + current_oofs, axis=0)

                # Ensemble across all 2nd stage models
                ensemble_probs_2nd = None
                for model_idx, model_2nd in enumerate(models_2nd):
                    print(f"      2nd stage model {model_idx+1}/{len(models_2nd)} (iter {iter_idx+1}/{NUM_ITERATIONS})")

                    # Run 2nd stage inference
                    x = torch.from_numpy(image_multichannel).float().unsqueeze(0)
                    torch.cuda.empty_cache()

                    with torch.inference_mode(), torch.autocast(
                        device_type="cuda" if device.type == "cuda" else "cpu",
                        enabled=(device.type == "cuda")
                    ):
                        out = sliding_window_inference(
                            inputs=x, roi_size=roi_size_2nd, sw_batch_size=SW_BATCH_SIZE,
                            predictor=model_2nd, overlap=OVERLAP_2ND_STAGE, mode=SW_MODE,
                            sigma_scale=SW_SIGMA_SCALE, padding_mode=PADDING_MODE, progress=False,
                            sw_device=device, device='cpu'
                        )

                    del x
                    torch.cuda.empty_cache()

                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    if out.shape[2:] != image.shape:
                        out = F.interpolate(out, size=image.shape, mode='trilinear', align_corners=False)

                    model_probs_2nd = F.softmax(out, dim=1)[:, 1][0].detach().cpu().numpy()
                    del out
                    torch.cuda.empty_cache()

                    # Add to ensemble
                    if ensemble_probs_2nd is None:
                        ensemble_probs_2nd = model_probs_2nd / len(models_2nd)
                    else:
                        ensemble_probs_2nd += model_probs_2nd / len(models_2nd)
                    del model_probs_2nd
                    torch.cuda.empty_cache()

                del image_multichannel
                avg_probs_2nd = ensemble_probs_2nd
                del ensemble_probs_2nd
                torch.cuda.empty_cache()

                # Save 2nd stage probability masks if enabled
                if save_probs_dir:
                    iter_suffix = f"_iter{iter_idx}" if NUM_ITERATIONS > 1 else ""
                    probs_filename = f"{sample_id}{iter_suffix}_probs.npy"
                    np.save(save_probs_dir / probs_filename, avg_probs_2nd.astype(np.float16))
                    print(f"      Saved 2nd stage probabilities to {probs_filename}")
                if iter_idx == NUM_ITERATIONS - 1:
                    _save_stage_probs(2, fold_idx, sample_id, avg_probs_2nd)

                # Threshold to get prediction
                pred = (avg_probs_2nd > PRED_THRESHOLD_2ND_STAGE).astype(np.uint8)
                del avg_probs_2nd
                torch.cuda.empty_cache()

                # Prepare for next iteration if needed
                if iter_idx < NUM_ITERATIONS - 1:
                    # Replace all OOF channels with the refined prediction
                    current_oofs = [pred.astype(np.float32)] * len(current_oofs)

            del image_normalized, current_oofs
            torch.cuda.empty_cache()

            # ==================
            # 3RD STAGE: Refine with [image, refineV2 mask]
            # ==================
            if NUM_STAGES >= 3:
                image_normalized_3rd = image.astype(np.float32) / 255.0
                current_mask_3rd = pred.astype(np.float32)

                for iter_idx_3rd in range(NUM_ITERATIONS_3RD_STAGE):
                    image_multichannel_3rd = np.stack([image_normalized_3rd, current_mask_3rd], axis=0)

                    ensemble_probs_3rd = None
                    for model_idx, model_3rd in enumerate(models_3rd):
                        print(f"      3rd stage model {model_idx+1}/{len(models_3rd)} (iter {iter_idx_3rd+1}/{NUM_ITERATIONS_3RD_STAGE})")

                        x = torch.from_numpy(image_multichannel_3rd).float().unsqueeze(0)
                        torch.cuda.empty_cache()

                        with torch.inference_mode(), torch.autocast(
                            device_type="cuda" if device.type == "cuda" else "cpu",
                            enabled=(device.type == "cuda")
                        ):
                            out = sliding_window_inference(
                                inputs=x, roi_size=roi_size_3rd, sw_batch_size=SW_BATCH_SIZE,
                                predictor=model_3rd, overlap=OVERLAP_3RD_STAGE, mode=SW_MODE,
                                sigma_scale=SW_SIGMA_SCALE, padding_mode=PADDING_MODE, progress=False,
                                sw_device=device, device='cpu'
                            )

                        del x
                        torch.cuda.empty_cache()

                        if isinstance(out, (list, tuple)):
                            out = out[0]
                        if out.shape[2:] != image.shape:
                            out = F.interpolate(out, size=image.shape, mode='trilinear', align_corners=False)

                        model_probs_3rd = F.softmax(out, dim=1)[:, 1][0].detach().cpu().numpy()
                        del out
                        torch.cuda.empty_cache()

                        if ensemble_probs_3rd is None:
                            ensemble_probs_3rd = model_probs_3rd / len(models_3rd)
                        else:
                            ensemble_probs_3rd += model_probs_3rd / len(models_3rd)
                        del model_probs_3rd
                        torch.cuda.empty_cache()

                    del image_multichannel_3rd
                    if iter_idx_3rd == NUM_ITERATIONS_3RD_STAGE - 1:
                        _save_stage_probs(3, fold_idx, sample_id, ensemble_probs_3rd)
                    pred = (ensemble_probs_3rd > PRED_THRESHOLD_3RD_STAGE).astype(np.uint8)
                    del ensemble_probs_3rd
                    torch.cuda.empty_cache()

                    if iter_idx_3rd < NUM_ITERATIONS_3RD_STAGE - 1:
                        current_mask_3rd = pred.astype(np.float32)

                del image_normalized_3rd, current_mask_3rd
                torch.cuda.empty_cache()

            # ==================
            # 4TH STAGE: Refine with [image, 3rd stage mask] - 2 iterations
            # ==================
            if NUM_STAGES >= 4 and models_4th:
                image_normalized_4th = image.astype(np.float32) / 255.0
                current_mask_4th = pred.astype(np.float32)

                for iter_idx_4th in range(NUM_ITERATIONS_4TH_STAGE):
                    image_multichannel_4th = np.stack([image_normalized_4th, current_mask_4th], axis=0)

                    ensemble_probs_4th = None
                    for model_idx, model_4th in enumerate(models_4th):
                        print(f"      4th stage model {model_idx+1}/{len(models_4th)} (iter {iter_idx_4th+1}/{NUM_ITERATIONS_4TH_STAGE})")

                        x = torch.from_numpy(image_multichannel_4th).float().unsqueeze(0)
                        torch.cuda.empty_cache()

                        with torch.inference_mode(), torch.autocast(
                            device_type="cuda" if device.type == "cuda" else "cpu",
                            enabled=(device.type == "cuda")
                        ):
                            out = sliding_window_inference(
                                inputs=x, roi_size=roi_size_4th, sw_batch_size=SW_BATCH_SIZE,
                                predictor=model_4th, overlap=OVERLAP_4TH_STAGE, mode=SW_MODE,
                                sigma_scale=SW_SIGMA_SCALE, padding_mode=PADDING_MODE, progress=False,
                                sw_device=device, device='cpu'
                            )

                        del x
                        torch.cuda.empty_cache()

                        if isinstance(out, (list, tuple)):
                            out = out[0]
                        if out.shape[2:] != image.shape:
                            out = F.interpolate(out, size=image.shape, mode='trilinear', align_corners=False)

                        model_probs_4th = F.softmax(out, dim=1)[:, 1][0].detach().cpu().numpy()
                        del out
                        torch.cuda.empty_cache()

                        if ensemble_probs_4th is None:
                            ensemble_probs_4th = model_probs_4th / len(models_4th)
                        else:
                            ensemble_probs_4th += model_probs_4th / len(models_4th)
                        del model_probs_4th
                        torch.cuda.empty_cache()

                    del image_multichannel_4th
                    if iter_idx_4th == NUM_ITERATIONS_4TH_STAGE - 1:
                        _save_stage_probs(4, fold_idx, sample_id, ensemble_probs_4th)
                    pred = (ensemble_probs_4th > PRED_THRESHOLD_4TH_STAGE).astype(np.uint8)
                    del ensemble_probs_4th
                    torch.cuda.empty_cache()

                    if iter_idx_4th < NUM_ITERATIONS_4TH_STAGE - 1:
                        current_mask_4th = pred.astype(np.float32)

                del image_normalized_4th, current_mask_4th
                torch.cuda.empty_cache()

            torch.cuda.empty_cache()

        if save_dir:
            np.save(save_dir / f"{sample_id}.npy", pred.astype(np.uint8, copy=False))

        if NUM_STAGES >= 2 or not CACHE_1ST_STAGE_ONLY:
            del image
        del pred
        torch.cuda.empty_cache()

    # Clean up models
    del models_1st, models_2nd, models_3rd, models_4th
    torch.cuda.empty_cache()



def main() -> None:
    global NUM_STAGES, SAVE_PREDS_DIR, CHECKPOINT_PATHS_1ST_STAGE, CACHE_1ST_STAGE_ONLY
    global CHECKPOINT_PATHS_2ND_STAGE, CHECKPOINT_PATHS_3RD_STAGE, CHECKPOINT_PATHS_4TH_STAGE
    global CACHE_1ST_STAGE_DIRS, CACHE_2ND_STAGE_DIR, CACHE_3RD_STAGE_DIR
    global SAVE_FINAL_PROBS_DIR

    args = parse_args()

    # Override globals from CLI args
    if args.num_stages is not None:
        NUM_STAGES = args.num_stages
    if args.save_preds_dir is not None:
        SAVE_PREDS_DIR = Path(args.save_preds_dir)
    if args.save_probs_dir is not None:
        SAVE_FINAL_PROBS_DIR = Path(args.save_probs_dir)
    if args.oof_dirs is not None:
        CACHE_1ST_STAGE_DIRS = [Path(d) for d in args.oof_dirs]
        CACHE_1ST_STAGE_ONLY = True
    if args.cache_2nd_stage_dir is not None:
        CACHE_2ND_STAGE_DIR = Path(args.cache_2nd_stage_dir)
    if args.cache_3rd_stage_dir is not None:
        CACHE_3RD_STAGE_DIR = Path(args.cache_3rd_stage_dir)

    # Auto-discover checkpoints from fold directories
    if args.checkpoint_dir_2nd is not None:
        print(f"Discovering 2nd stage checkpoints from {args.checkpoint_dir_2nd}:")
        CHECKPOINT_PATHS_2ND_STAGE = _discover_fold_checkpoints(args.checkpoint_dir_2nd, args.n_folds)
    if args.checkpoint_dir_3rd is not None:
        print(f"Discovering 3rd stage checkpoints from {args.checkpoint_dir_3rd}:")
        CHECKPOINT_PATHS_3RD_STAGE = _discover_fold_checkpoints(args.checkpoint_dir_3rd, args.n_folds)
    if args.checkpoint_dir_4th is not None:
        print(f"Discovering 4th stage checkpoints from {args.checkpoint_dir_4th}:")
        CHECKPOINT_PATHS_4TH_STAGE = _discover_fold_checkpoints(args.checkpoint_dir_4th, args.n_folds)

    if args.ckpt_pattern is not None:
        # Build CHECKPOINT_PATHS_1ST_STAGE from glob pattern + checkpoint_dir
        import glob
        ckpt_dir = Path(args.checkpoint_dir)
        CHECKPOINT_PATHS_1ST_STAGE = {}
        CACHE_1ST_STAGE_ONLY = False
        for fold_idx in range(args.n_folds):
            pattern = str(ckpt_dir / args.ckpt_pattern.replace("{fold}", str(fold_idx)))
            matches = sorted(glob.glob(pattern))
            if matches:
                CHECKPOINT_PATHS_1ST_STAGE[f"fold{fold_idx}"] = matches
                print(f"  fold{fold_idx}: {[Path(m).name for m in matches]}")
            else:
                print(f"  fold{fold_idx}: no checkpoints matching {pattern}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load training data and create folds
    train_df = pd.read_csv(args.train_csv)
    if "id" not in train_df.columns:
        raise ValueError(f"Training CSV must have an 'id' column")

    print(f"Loaded {len(train_df)} samples from {args.train_csv}")
    print(f"Creating {args.n_folds}-fold CV split with seed={args.seed}")

    folds = get_folds(train_df, n_folds=args.n_folds, seed=args.seed)

    # Determine which folds to run
    if "all" in args.folds:
        folds_to_run = list(range(args.n_folds))
    else:
        folds_to_run = [int(f) for f in args.folds]

    print(f"Folds to evaluate: {folds_to_run}")
    print(f"NUM_STAGES={NUM_STAGES}")
    if CACHE_3RD_STAGE_DIR:
        print(f"Mode: CACHE_3RD_STAGE – loading pre-computed 3rd stage probs from {CACHE_3RD_STAGE_DIR}")
    elif CACHE_2ND_STAGE_DIR:
        print(f"Mode: CACHE_2ND_STAGE – loading pre-computed probs from {CACHE_2ND_STAGE_DIR}")
    else:
        print(f"Mode: {NUM_STAGES}-Stage inference")
    if SAVE_PREDS_DIR:
        print(f"Saving predictions to: {SAVE_PREDS_DIR}")
    if SAVE_FINAL_PROBS_DIR:
        print(f"Saving stage {NUM_STAGES} probs to: {SAVE_FINAL_PROBS_DIR}")

    # Evaluate each fold
    for fold_idx in folds_to_run:
        if fold_idx >= len(folds):
            print(f"WARNING: Fold {fold_idx} does not exist (only {len(folds)} folds)")
            continue

        _, val_df = folds[fold_idx]
        evaluate_fold(
            fold_idx=fold_idx,
            val_df=val_df,
            device=device,
            max_samples=args.max_samples,
        )

    print("\nOOF generation complete.")


if __name__ == "__main__":
    main()
