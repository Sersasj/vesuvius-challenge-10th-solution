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
from monai.inferers.inferer import SlidingWindowInfererAdapt
from skimage import morphology
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from sklearn.model_selection import KFold
from collections import OrderedDict
from monai import transforms as monai_transforms
from omegaconf import OmegaConf

# Ensure repo root is on sys.path so `import src...` works even when running via an absolute script path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.lightning_module import SegmentationModule
from src_2nd_4th_stages.models.lightning_module import SegmentationModule as SegmentationModule2ndStage
from src.utils.metric import load_volume

# Tom-style DeformNet (DeformDynUnetV2 from tom_submission_stuff)
import importlib.util as _ilu
_deformnet_path = str(_REPO_ROOT / "tom_submission_stuff" / "deformnetv2" / "src" / "models" / "deformNet3d.py")
_spec = _ilu.spec_from_file_location("deformNet3d", _deformnet_path)
_deformNet3d = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_deformNet3d)
DeformDynUnetV2 = _deformNet3d.DeformDynUnetV2

# Sergio-style DeformNet (src_deformnet_stage Lightning module)
from src_deformnet_stage.models.lightning_module import SegmentationModule as DeformSegModule


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

# Direct checkpoint paths mapping - 1st stage models (can be list for ensemble with multiple models including primus)
CHECKPOINT_PATHS_1ST_STAGE = {
    "fold0": [
        "1st_stage_models/fold-0-best-epoch369-val_loss0.3720-val_dice0.5789.ckpt", # resEnc
        "primus_pretrained/fold0_best-epoch=264-val_loss=0.3657-val_dice=0.5852.ckpt"  #primus
        "primus_v2/fold0_best-epoch=404-val_loss=0.3655-val_dice=0.5881.ckpt", # primus V2
    ],
    "fold1": [
        "1st_stage_models/fold-1-best-epoch364-val_loss0.3767-val_dice0.5813.ckpt",
        # "1st_stage_models_primus/fold-1-best.ckpt",
    ],
    "fold2": [
        "1st_stage_models/fold-2-best-epoch384-val_loss0.3647-val_dice0.5865.ckpt",
        # "1st_stage_models_primus/fold-2-best.ckpt",
    ],
    "fold3": [
        "1st_stage_models/fold-3-best-epoch349-val_loss0.3471-val_dice0.6062.ckpt",
        # "1st_stage_models_primus/fold-3-best.ckpt",
    ],
    "fold4": [
        "1st_stage_models/fold-4-best-epoch364-val_loss0.3534-val_dice0.6028.ckpt",
        # "1st_stage_models_primus/fold-4-best.ckpt",
    ],
}

# Direct checkpoint paths mapping - 2nd stage models (can be list for ensemble)
# 4-channel input: [image, resEnc OOF, primus OOF, primusV2 OOF]
CHECKPOINT_PATHS_2ND_STAGE = {
    "fold0": [
        "refineV2_4ch/fold0_best-epoch134-val_dice0.6015-val_loss0.3528.ckpt",
    ],
    "fold1": [
        "refineV2_4ch/fold1_best-epoch119-val_dice0.6102-val_loss0.3507.ckpt",
    ],
    "fold2": [
        "refineV2_4ch/fold_2best-epoch139-val_dice0.6082-val_loss0.3450.ckpt",
    ],
    "fold3": [
        "refineV2_4ch/fold3_best.ckpt",  # TODO: not yet trained
    ],
    "fold4": [
        "refineV2_4ch/fold4_best.ckpt",  # TODO: not yet trained
    ],
}

# Input is [image, refineV2 mask]
CHECKPOINT_PATHS_3RD_STAGE = {
    "fold0": [
        "refineV1_new/refine-v1-fold0-best-epoch69-val_dice0.6158-val_loss0.3396.ckpt",
    ],
    "fold1": [
        "refineV1_new/refine-v1-fold1-best-epoch74-val_dice0.6267-val_loss0.3363.ckpt",
    ]
}
CHECKPOINT_PATHS_4TH_STAGE = {
    "fold0": [
        "new_refineTom/fold-0-best-epoch=99-val_dice=0.5995-val_loss=0.3575.ckpt",
        #"RefineTom/fold-0-best-epoch=114-val_dice=0.6000-val_loss=0.3570.ckpt"
    ],
    "fold1": "RefineTom/fold-1-best-epoch=64-val_dice=0.6097-val_loss=0.3562.ckpt",
}
CHECKPOINT_PATHS_5TH_STAGE = {
    "fold0": "5th_stage_tom/fold0_best-epoch=99-val_dice=0.5994-val_loss=0.3583.ckpt",
    "fold1": "5th_stage_tom/best-epoch=114-val_dice=0.6088-val_loss=0.3558.ckpt",
}

# 6th stage: DeformNet (diffeomorphic registration)
# Each entry can be:
#   "path/to/ckpt"             – plain string, defaults to "tom" loader (DeformDynUnetV2)
#   ("path/to/ckpt", "tom")    – Tom's DeformDynUnetV2 (tom_submission_stuff)
#   ("path/to/ckpt", "sergio") – Sergio's DeformSegModule (src_deformnet_stage Lightning module)
CHECKPOINT_PATHS_6TH_STAGE = {
    "fold0": [
        ("deformSergio/fold-0-best-epoch=74-val_dice=0.5980-val_loss=0.4271.ckpt", "sergio"),
        ("bestDeformnet/0594/deform-dynunet-v2-k3-s5-customFalse-fold0-epoch=19-val_bias_comp_metric=0.7012.ckpt", "tom"),
        ("bestDeformnet/0594/deform-dynunet-v2-k3-s5-fold0-epoch=29-val_bias_comp_metric=0.6885.ckpt", "tom"),
    ],
    "fold1": ("diffeomophic-3stage-vesuvius-challenge-pytorch-default-v9/deform-dynunet-v2-k3-s5-customFalse-fold1-epoch59-val_bias_comp_metric0.7453.ckpt", "tom"),
    "fold2": ("diffeomophic-3stage-vesuvius-challenge-pytorch-default-v9/deform-dynunet-v2-k3-s5-customFalse-fold2-epoch109-val_bias_comp_metric0.7393.ckpt", "tom"),
    "fold3": ("diffeomophic-3stage-vesuvius-challenge-pytorch-default-v9/deform-dynunet-v2-k3-s5-customFalse-fold3-epoch74-val_bias_comp_metric0.7413.ckpt", "tom"),
    "fold4": ("diffeomophic-3stage-vesuvius-challenge-pytorch-default-v9/deform-dynunet-v2-k3-s5-customFalse-fold4-epoch89-val_bias_comp_metric0.7496.ckpt", "tom"),
}

# DeformNet hyperparameters – Tom loader (DeformDynUnetV2)
DEFORM_KERNEL_SIZE_TOM = 3
DEFORM_SIGMA_TOM = 5
DEFORM_BINARIZE_THRESHOLD = 0.3
DEFORM_FINAL_THRESHOLD = 0.5
DEFORM_WEIGHT_SERGIO = 0.3  # Weight for Sergio-style DeformNet in ensemble
DEFORM_WEIGHT_TOM = 0.7     # Weight for Tom-style DeformNet in ensemble
DEFORM_OVERLAP = 0.5
DEFORM_INPUT_SIZE = (160, 160, 160)

DEFORM_CFG = OmegaConf.create({
    "use_resenc": True,
    "max_v": 1.5,
    "n_steps": 6,
    "input_size": list(DEFORM_INPUT_SIZE),
    "out_channels": 4,
    "max_topo_offset": 1.0,
})

# DeformNet hyperparameters – Sergio loader (src_deformnet_stage DeformSegModule)
DEFORM_KERNEL_SIZE_SERGIO = 3
DEFORM_SIGMA_SERGIO = 5

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
PATCH_SIZE_5TH_STAGE: Optional[int] = 160
SW_BATCH_SIZE = 1
OVERLAP_1ST_STAGE = 0.5
OVERLAP_2ND_STAGE = 0.5
OVERLAP_3RD_STAGE = 0.5
OVERLAP_4TH_STAGE = 0.5
OVERLAP_5TH_STAGE = 0.5
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
PRED_THRESHOLD_5TH_STAGE = 0.3

USE_TTA = True
USE_POST_PROCESSING = True
POST_PROCESS_MIN_CC_VOLUME = 3000

# How many stages to run: 1 = 1st only, ..., 5 = +5th refinement, 6 = all including DeformNet
NUM_STAGES = 6

# 2nd stage specific settings
NUM_ITERATIONS = 1  # Number of refinement iterations for 2nd stage
NUM_ITERATIONS_3RD_STAGE = 2  # Number of refinement iterations for 3rd stage
NUM_ITERATIONS_4TH_STAGE = 1  # Number of refinement iterations for 4th stage
NUM_ITERATIONS_5TH_STAGE = 0  # Number of refinement iterations for 5th stage
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

# 4th stage caching – load pre-computed 4th stage OOF probabilities (skip 1st–4th inference, run only DeformNet)
# Set to a directory with {sample_id}_probs.npy files (e.g. output of generate_oof_4_stage.py)
CACHE_4TH_STAGE_DIR: Optional[Path] = None  # Set to None to run full pipeline

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
        help="Number of stages to run (1-6). Overrides NUM_STAGES global."
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

# ==========================
# DeformNet Utilities (6th Stage)
# ==========================

def gaussian_kernel_3d(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Returns a normalized 3D Gaussian kernel."""
    ax = torch.arange(kernel_size, device=device) - kernel_size // 2
    xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def gaussian_blur_3d(x: torch.Tensor, kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    _, C, _, _, _ = x.shape
    kernel = gaussian_kernel_3d(kernel_size, sigma, device)
    kernel = kernel.expand(C, 1, kernel_size, kernel_size, kernel_size)
    return F.conv3d(x, kernel, padding=kernel_size // 2, groups=C)


def load_deform_model_from_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    """Load DeformNet model weights from checkpoint (Tom-style, strips 'model.' prefix)."""
    ckpt = torch.load(ckpt_path, weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("model.", "") if k.startswith("model.") else k
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)


@torch.no_grad()
def one_step_inference_deform(
    prob_mask: torch.Tensor,
    vol: torch.Tensor,
    kernel_size: int,
    sigma: float,
    deformnet_list: list,
    is_prob: bool,
    apply_gaussian: bool,
    threshold: float,
    sliding_window_inferer: SlidingWindowInfererAdapt,
    device: torch.device,
) -> torch.Tensor:
    """Single-GPU DeformNet inference step (Tom-style)."""
    if not is_prob:
        prob_mask = (prob_mask > threshold).float()

    if apply_gaussian:
        prob_mask_device = prob_mask.to(device)
        prev_mask_pred = gaussian_blur_3d(prob_mask_device, kernel_size, sigma, device)
        del prob_mask_device
    else:
        prev_mask_pred = prob_mask.to(device)

    vol_device = vol.to(device)
    x = torch.cat([vol_device, prev_mask_pred], dim=1)
    del vol_device, prev_mask_pred
    x = x.cpu()

    predictions = []
    for model in deformnet_list:
        x_device = x.to(device)
        pred_warped = sliding_window_inferer(x_device, model)
        predictions.append(pred_warped.cpu())
        del x_device
        torch.cuda.empty_cache()

    return torch.cat(predictions, dim=0).mean(dim=0)


@torch.no_grad()
def _predict_deformnet_tom(
    tom_models: list,
    image_3d: np.ndarray,
    prob_mask: np.ndarray,
    *,
    device: torch.device,
    binarize_threshold: float,
    final_threshold: float,
    sliding_window_inferer: SlidingWindowInfererAdapt,
    deformnet_transforms_fn,
) -> np.ndarray:
    """Run Tom-style DeformDynUnetV2 inference. Returns probability array (D,H,W) in [0,1]."""
    raw = {"Image": image_3d, "Mask_OOF": prob_mask}
    _data = deformnet_transforms_fn(raw)
    vol = _data['Image'][None,]
    mask = _data['Mask_OOF'][None,]

    vol = vol.to(device)
    mask = mask.to(device).float()

    prediction = one_step_inference_deform(
        mask, vol, DEFORM_KERNEL_SIZE_TOM, DEFORM_SIGMA_TOM,
        tom_models, is_prob=False, apply_gaussian=True,
        threshold=binarize_threshold,
        sliding_window_inferer=sliding_window_inferer,
        device=device,
    )

    del vol, mask
    torch.cuda.empty_cache()

    return prediction[0].cpu().numpy().astype(np.float32)


@torch.no_grad()
def _predict_deformnet_sergio(
    sergio_models: list,
    image_3d: np.ndarray,
    prob_mask: np.ndarray,
    *,
    device: torch.device,
    binarize_threshold: float,
) -> np.ndarray:
    """Run Sergio-style DeformSegModule (src_deformnet_stage) inference. Returns probability array (D,H,W) in [0,1]."""
    vol = torch.from_numpy(image_3d.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    if vol.max() > 0:
        vol = vol / 255.0

    mask = torch.from_numpy(prob_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    mask = (mask > binarize_threshold).float()
    mask = gaussian_blur_3d(mask.to(device), DEFORM_KERNEL_SIZE_SERGIO, DEFORM_SIGMA_SERGIO, device).cpu()

    x = torch.cat([vol, mask], dim=1)  # (1, 2, D, H, W)
    del vol, mask

    roi_size = DEFORM_INPUT_SIZE
    predictions = []
    for model in sergio_models:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            enabled=(device.type == "cuda"),
        ):
            out = sliding_window_inference(
                inputs=x, roi_size=roi_size, sw_batch_size=1,
                predictor=model, overlap=DEFORM_OVERLAP, mode="gaussian",
                sigma_scale=SW_SIGMA_SCALE, padding_mode=PADDING_MODE,
                progress=True, sw_device=device, device='cpu',
            )
        predictions.append(out)
        torch.cuda.empty_cache()

    prediction_ensemble = torch.cat(predictions, dim=0).mean(dim=0)  # (1, D, H, W)
    del predictions, x
    torch.cuda.empty_cache()

    return prediction_ensemble[0].cpu().numpy().astype(np.float32)


@torch.no_grad()
def predict_volume_deformnet(
    deformnet_entries: list,
    image_3d: np.ndarray,
    prob_mask: np.ndarray,
    *,
    device: torch.device,
    binarize_threshold: float,
    final_threshold: float,
    sliding_window_inferer: SlidingWindowInfererAdapt,
    deformnet_transforms_fn,
) -> np.ndarray:
    """Run mixed-ensemble DeformNet inference (supports both 'tom' and 'sergio' entries).

    deformnet_entries: list of (model, kind) where kind is 'tom' or 'sergio'.
    Returns binary mask (D,H,W) uint8.
    """
    tom_models = [m for m, kind in deformnet_entries if kind == "tom"]
    sergio_models = [m for m, kind in deformnet_entries if kind == "sergio"]

    all_probs = []

    if tom_models and sliding_window_inferer is not None and deformnet_transforms_fn is not None:
        probs_tom = _predict_deformnet_tom(
            tom_models, image_3d, prob_mask,
            device=device,
            binarize_threshold=binarize_threshold,
            final_threshold=final_threshold,
            sliding_window_inferer=sliding_window_inferer,
            deformnet_transforms_fn=deformnet_transforms_fn,
        )
        all_probs.append(probs_tom)

    if sergio_models:
        probs_sergio = _predict_deformnet_sergio(
            sergio_models, image_3d, prob_mask,
            device=device,
            binarize_threshold=binarize_threshold,
        )
        all_probs.append(probs_sergio)

    if not all_probs:
        return prob_mask.astype(np.uint8)

    # Weighted ensemble: if both tom and sergio are present, use configured weights
    if tom_models and sergio_models and len(all_probs) == 2:
        # all_probs[0] = tom, all_probs[1] = sergio (appended in that order above)
        avg_probs = DEFORM_WEIGHT_TOM * all_probs[0] + DEFORM_WEIGHT_SERGIO * all_probs[1]
    else:
        avg_probs = np.mean(np.stack(all_probs, axis=0), axis=0)
    return (avg_probs > final_threshold).astype(np.uint8)


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


def _load_5th_stage_refinement_models(fold_key: str, device: torch.device):
    """Load 5th stage refinement model(s) (SegmentationModule2ndStage). Returns list of models or []."""
    if fold_key not in CHECKPOINT_PATHS_5TH_STAGE:
        return []

    ckpt_raw = CHECKPOINT_PATHS_5TH_STAGE[fold_key]
    ckpt_paths = [Path(p) for p in (ckpt_raw if isinstance(ckpt_raw, list) else [ckpt_raw])]

    models_5th = []
    for i, ckpt_path in enumerate(ckpt_paths):
        if not ckpt_path.exists():
            print(f"WARNING: 5th stage checkpoint not found: {ckpt_path}")
            return []
        print(f"  Loading 5th stage refinement model {i+1}/{len(ckpt_paths)}: {ckpt_path.name}")
        m = _load_model_with_ema(SegmentationModule2ndStage, str(ckpt_path), device)
        models_5th.append(m)
    return models_5th


def _parse_deform_ckpt_entry(entry) -> Tuple[str, str]:
    """Parse a checkpoint entry into (path_str, kind).

    Accepts:
      "path/to/ckpt"           → (path, "tom")  [default]
      ("path/to/ckpt", "tom")  → (path, "tom")
      ("path/to/ckpt", "sergio") → (path, "sergio")
    """
    if isinstance(entry, str):
        return entry, "tom"
    path, kind = entry
    assert kind in ("tom", "sergio"), f"Unknown deform kind '{kind}', expected 'tom' or 'sergio'"
    return str(path), kind


def _load_deformnet_models(fold_key: str, device: torch.device):
    """Load 6th stage DeformNet model(s).

    Checkpoint entries may mix 'tom' (DeformDynUnetV2) and 'sergio' (DeformSegModule) models.
    Returns (deformnet_entries, deform_swi, deform_tfn) where:
      deformnet_entries – list of (model, kind)
      deform_swi        – SlidingWindowInfererAdapt for tom models (None if no tom models)
      deform_tfn        – monai transforms for tom models (None if no tom models)
    """
    if fold_key not in CHECKPOINT_PATHS_6TH_STAGE:
        print(f"  No DeformNet checkpoint for {fold_key}, skipping")
        return None, None, None

    ckpt_raw = CHECKPOINT_PATHS_6TH_STAGE[fold_key]
    raw_list = ckpt_raw if isinstance(ckpt_raw, list) else [ckpt_raw]

    deformnet_entries = []
    has_tom = False

    for i, raw_entry in enumerate(raw_list):
        path_str, kind = _parse_deform_ckpt_entry(raw_entry)
        ckpt_path = Path(path_str)

        if not ckpt_path.exists():
            print(f"WARNING: DeformNet checkpoint not found: {ckpt_path}")
            return None, None, None

        print(f"  Loading DeformNet [{kind}] model {i+1}/{len(raw_list)}: {ckpt_path.name}")

        if kind == "tom":
            m = DeformDynUnetV2(DEFORM_CFG).to(device)
            m.eval()
            load_deform_model_from_checkpoint(m, ckpt_path)
            has_tom = True
        else:  # "sergio"
            m = _load_model_with_ema(DeformSegModule, str(ckpt_path), device)

        deformnet_entries.append((m, kind))

    deform_swi, deform_tfn = None, None
    if has_tom:
        deform_swi = SlidingWindowInfererAdapt(
            roi_size=DEFORM_INPUT_SIZE, sw_batch_size=1,
            overlap=DEFORM_OVERLAP, mode="gaussian", sigma_scale=SW_SIGMA_SCALE, progress=True,
        )

        deform_tfn = monai_transforms.Compose([
            monai_transforms.EnsureChannelFirstd(keys=["Image", "Mask_OOF"], channel_dim="no_channel"),
            monai_transforms.EnsureTyped(keys=["Image", "Mask_OOF"], dtype=np.float32),
            monai_transforms.ScaleIntensityRanged(
                keys=["Image"], a_min=0.0, a_max=213.0, b_min=-5.0, b_max=5.0, clip=True,
            ),
            monai_transforms.ToTensord(keys=["Image", "Mask_OOF"]),
        ])

    return deformnet_entries, deform_swi, deform_tfn


def evaluate_fold(
    fold_idx: int,
    val_df: pd.DataFrame,
    device: torch.device,
    max_samples: int = MAX_SAMPLES,
) -> pd.DataFrame:
    """Evaluate a single fold with multi-stage inference and return results DataFrame.

    Supports four modes:
    1. CACHE_4TH_STAGE_DIR set: load pre-computed 4th stage OOF probs, run 5th + 6th stages
    2. CACHE_2ND_STAGE_DIR set: load pre-computed 2nd stage OOF probs, run 3rd–6th stages
    3. CACHE_1ST_STAGE_ONLY: load cached 1st stage probs, run 2nd–6th stage inference
    4. Full: run all stages from scratch
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Fold {fold_idx} ({NUM_STAGES}-Stage)")
    print(f"{'='*60}")

    fold_key = f"fold{fold_idx}"

    # ==================
    # MODE 1a: Load cached 4th stage OOF probs (run only 5th refinement + DeformNet)
    # ==================
    if CACHE_4TH_STAGE_DIR is not None:
        cache_4th_dir = Path(CACHE_4TH_STAGE_DIR)
        cache_4th_fold_dir = cache_4th_dir / f"fold_{fold_idx}"
        cache_4th_probs_dir = cache_4th_fold_dir / "probs"
        print(f"CACHE_4TH_STAGE mode – loading from {cache_4th_dir}, running 5th refinement + DeformNet")

        # Load 5th stage refinement model(s)
        models_5th = []
        if NUM_STAGES >= 5:
            models_5th = _load_5th_stage_refinement_models(fold_key, device)
        patch_size_5th = PATCH_SIZE_5TH_STAGE or 128
        roi_size_5th = (patch_size_5th, patch_size_5th, patch_size_5th)

        # Load 6th stage (DeformNet) model(s)
        deformnet_entries, deform_swi, deform_tfn = None, None, None
        if NUM_STAGES >= 6:
            deformnet_entries, deform_swi, deform_tfn = _load_deformnet_models(fold_key, device)

        if not models_5th and deformnet_entries is None:
            print(f"  WARNING: No 5th or 6th stage checkpoint for {fold_key}, will evaluate 4th stage only")

        val_df = val_df.head(max_samples)
        print(f"Evaluating {len(val_df)} samples for fold {fold_idx}")

        for sample_idx, (_, row) in enumerate(val_df.iterrows()):
            sample_id = row["id"]
            print(f"    [{sample_idx+1}/{len(val_df)}] Processing: {sample_id}")

            # Try to find cached 4th stage probs
            probs = None
            for try_dir in [cache_4th_dir, cache_4th_fold_dir, cache_4th_probs_dir]:
                cache_file = try_dir / f"{sample_id}_probs.npy"
                if cache_file.exists():
                    print(f"    Loading cached 4th stage probs from {cache_file}")
                    probs = np.load(cache_file).astype(np.float32)
                    break

            if probs is None:
                print(f"    SKIP: No cached 4th stage probs found for {sample_id}")
                continue

            pred = (probs > PRED_THRESHOLD_4TH_STAGE).astype(np.uint8)
            del probs

            image = np.asarray(load_volume(_find_file(IMAGE_DIR, sample_id)))

            # 5TH STAGE: Refine with [image, 4th stage mask] - N iterations
            if models_5th:
                image_normalized_5th = image.astype(np.float32) / 255.0
                current_mask_5th = pred.astype(np.float32)

                for iter_idx_5th in range(NUM_ITERATIONS_5TH_STAGE):
                    image_multichannel_5th = np.stack([image_normalized_5th, current_mask_5th], axis=0)

                    ensemble_probs_5th = None
                    for model_idx, model_5th in enumerate(models_5th):
                        print(f"      5th stage model {model_idx+1}/{len(models_5th)} (iter {iter_idx_5th+1}/{NUM_ITERATIONS_5TH_STAGE})")

                        x = torch.from_numpy(image_multichannel_5th).float().unsqueeze(0)
                        torch.cuda.empty_cache()

                        with torch.inference_mode(), torch.autocast(
                            device_type="cuda" if device.type == "cuda" else "cpu",
                            enabled=(device.type == "cuda")
                        ):
                            out = sliding_window_inference(
                                inputs=x, roi_size=roi_size_5th, sw_batch_size=SW_BATCH_SIZE,
                                predictor=model_5th, overlap=OVERLAP_5TH_STAGE, mode=SW_MODE,
                                sigma_scale=SW_SIGMA_SCALE, padding_mode=PADDING_MODE, progress=False,
                                sw_device=device, device='cpu'
                            )

                        del x
                        torch.cuda.empty_cache()

                        if isinstance(out, (list, tuple)):
                            out = out[0]
                        if out.shape[2:] != image.shape:
                            out = F.interpolate(out, size=image.shape, mode='trilinear', align_corners=False)

                        model_probs_5th = F.softmax(out, dim=1)[:, 1][0].detach().cpu().numpy()
                        del out
                        torch.cuda.empty_cache()

                        if ensemble_probs_5th is None:
                            ensemble_probs_5th = model_probs_5th / len(models_5th)
                        else:
                            ensemble_probs_5th += model_probs_5th / len(models_5th)
                        del model_probs_5th
                        torch.cuda.empty_cache()

                    del image_multichannel_5th
                    pred = (ensemble_probs_5th > PRED_THRESHOLD_5TH_STAGE).astype(np.uint8)
                    del ensemble_probs_5th
                    torch.cuda.empty_cache()

                    if iter_idx_5th < NUM_ITERATIONS_5TH_STAGE - 1:
                        current_mask_5th = pred.astype(np.float32)

                del image_normalized_5th, current_mask_5th
                torch.cuda.empty_cache()

            # 6TH STAGE: DeformNet refinement
            if deformnet_entries:
                print(f"      Running 6th stage (DeformNet) with {len(deformnet_entries)} model(s)...")
                pred_deform = predict_volume_deformnet(
                    deformnet_entries,
                    image,
                    pred.astype(np.float32),
                    device=device,
                    binarize_threshold=DEFORM_BINARIZE_THRESHOLD,
                    final_threshold=DEFORM_FINAL_THRESHOLD,
                    sliding_window_inferer=deform_swi,
                    deformnet_transforms_fn=deform_tfn,
                )
                print(f"      DeformNet complete. Positive ratio: {pred_deform.mean():.4f}")
                pred = pred_deform
                del pred_deform
            torch.cuda.empty_cache()

            del image, pred
            torch.cuda.empty_cache()

        del models_5th
        if deformnet_entries:
            del deformnet_entries
        torch.cuda.empty_cache()
        return

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
    # MODE 1b: Load cached 2nd stage OOF probs (run 3rd, 4th, 5th, 6th)
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

        # Load 5th stage refinement model(s)
        models_5th = []
        if NUM_STAGES >= 5:
            models_5th = _load_5th_stage_refinement_models(fold_key, device)
        patch_size_5th = PATCH_SIZE_5TH_STAGE or 128
        roi_size_5th = (patch_size_5th, patch_size_5th, patch_size_5th)

        # Load 6th stage (DeformNet) model(s)
        deformnet_entries, deform_swi, deform_tfn = None, None, None
        if NUM_STAGES >= 6:
            deformnet_entries, deform_swi, deform_tfn = _load_deformnet_models(fold_key, device)

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

            # 5TH STAGE: Refine with [image, 4th stage mask] - N iterations
            if models_5th:
                image_normalized_5th = image.astype(np.float32) / 255.0
                current_mask_5th = pred.astype(np.float32)

                for iter_idx_5th in range(NUM_ITERATIONS_5TH_STAGE):
                    image_multichannel_5th = np.stack([image_normalized_5th, current_mask_5th], axis=0)

                    ensemble_probs_5th = None
                    for model_idx, model_5th in enumerate(models_5th):
                        print(f"      5th stage model {model_idx+1}/{len(models_5th)} (iter {iter_idx_5th+1}/{NUM_ITERATIONS_5TH_STAGE})")

                        x = torch.from_numpy(image_multichannel_5th).float().unsqueeze(0)
                        torch.cuda.empty_cache()

                        with torch.inference_mode(), torch.autocast(
                            device_type="cuda" if device.type == "cuda" else "cpu",
                            enabled=(device.type == "cuda")
                        ):
                            out = sliding_window_inference(
                                inputs=x, roi_size=roi_size_5th, sw_batch_size=SW_BATCH_SIZE,
                                predictor=model_5th, overlap=OVERLAP_5TH_STAGE, mode=SW_MODE,
                                sigma_scale=SW_SIGMA_SCALE, padding_mode=PADDING_MODE, progress=False,
                                sw_device=device, device='cpu'
                            )

                        del x
                        torch.cuda.empty_cache()

                        if isinstance(out, (list, tuple)):
                            out = out[0]
                        if out.shape[2:] != image.shape:
                            out = F.interpolate(out, size=image.shape, mode='trilinear', align_corners=False)

                        model_probs_5th = F.softmax(out, dim=1)[:, 1][0].detach().cpu().numpy()
                        del out
                        torch.cuda.empty_cache()

                        if ensemble_probs_5th is None:
                            ensemble_probs_5th = model_probs_5th / len(models_5th)
                        else:
                            ensemble_probs_5th += model_probs_5th / len(models_5th)
                        del model_probs_5th
                        torch.cuda.empty_cache()

                    del image_multichannel_5th
                    pred = (ensemble_probs_5th > PRED_THRESHOLD_5TH_STAGE).astype(np.uint8)
                    del ensemble_probs_5th
                    torch.cuda.empty_cache()

                    if iter_idx_5th < NUM_ITERATIONS_5TH_STAGE - 1:
                        current_mask_5th = pred.astype(np.float32)

                del image_normalized_5th, current_mask_5th
                torch.cuda.empty_cache()

            # 6TH STAGE: DeformNet refinement
            if deformnet_entries:
                print(f"      Running 6th stage (DeformNet) with {len(deformnet_entries)} model(s)...")
                pred_deform = predict_volume_deformnet(
                    deformnet_entries,
                    image,
                    pred.astype(np.float32),
                    device=device,
                    binarize_threshold=DEFORM_BINARIZE_THRESHOLD,
                    final_threshold=DEFORM_FINAL_THRESHOLD,
                    sliding_window_inferer=deform_swi,
                    deformnet_transforms_fn=deform_tfn,
                )
                print(f"      DeformNet complete. Positive ratio: {pred_deform.mean():.4f}")
                pred = pred_deform
                del pred_deform
                torch.cuda.empty_cache()

            del image, pred
            torch.cuda.empty_cache()

        del models_3rd, models_4th, models_5th
        if deformnet_entries:
            del deformnet_entries
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

    # Load 5th stage refinement model(s)
    models_5th = []
    if NUM_STAGES >= 5:
        models_5th = _load_5th_stage_refinement_models(fold_key, device)

    # Load 6th stage (DeformNet) model(s)
    deformnet_entries, deform_swi, deform_tfn = None, None, None
    if NUM_STAGES >= 6:
        deformnet_entries, deform_swi, deform_tfn = _load_deformnet_models(fold_key, device)

    patch_size_1st = PATCH_SIZE_1ST_STAGE or 128
    patch_size_2nd = PATCH_SIZE_2ND_STAGE or 128
    patch_size_3rd = PATCH_SIZE_3RD_STAGE or 128
    patch_size_4th = PATCH_SIZE_4TH_STAGE or 128
    patch_size_5th = PATCH_SIZE_5TH_STAGE or 128
    roi_size_1st = (patch_size_1st, patch_size_1st, patch_size_1st)
    roi_size_2nd = (patch_size_2nd, patch_size_2nd, patch_size_2nd)
    roi_size_3rd = (patch_size_3rd, patch_size_3rd, patch_size_3rd)
    roi_size_4th = (patch_size_4th, patch_size_4th, patch_size_4th)
    roi_size_5th = (patch_size_5th, patch_size_5th, patch_size_5th)
    print(f"Sliding window roi_size_1st={roi_size_1st}, roi_size_2nd={roi_size_2nd}, roi_size_3rd={roi_size_3rd}, roi_size_4th={roi_size_4th}, roi_size_5th={roi_size_5th}")

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

            # ==================
            # 5TH STAGE: Refine with [image, 4th stage mask] - N iterations
            # ==================
            if NUM_STAGES >= 5 and models_5th:
                image_normalized_5th = image.astype(np.float32) / 255.0
                current_mask_5th = pred.astype(np.float32)

                for iter_idx_5th in range(NUM_ITERATIONS_5TH_STAGE):
                    image_multichannel_5th = np.stack([image_normalized_5th, current_mask_5th], axis=0)

                    ensemble_probs_5th = None
                    for model_idx, model_5th in enumerate(models_5th):
                        print(f"      5th stage model {model_idx+1}/{len(models_5th)} (iter {iter_idx_5th+1}/{NUM_ITERATIONS_5TH_STAGE})")

                        x = torch.from_numpy(image_multichannel_5th).float().unsqueeze(0)
                        torch.cuda.empty_cache()

                        with torch.inference_mode(), torch.autocast(
                            device_type="cuda" if device.type == "cuda" else "cpu",
                            enabled=(device.type == "cuda")
                        ):
                            out = sliding_window_inference(
                                inputs=x, roi_size=roi_size_5th, sw_batch_size=SW_BATCH_SIZE,
                                predictor=model_5th, overlap=OVERLAP_5TH_STAGE, mode=SW_MODE,
                                sigma_scale=SW_SIGMA_SCALE, padding_mode=PADDING_MODE, progress=False,
                                sw_device=device, device='cpu'
                            )

                        del x
                        torch.cuda.empty_cache()

                        if isinstance(out, (list, tuple)):
                            out = out[0]
                        if out.shape[2:] != image.shape:
                            out = F.interpolate(out, size=image.shape, mode='trilinear', align_corners=False)

                        model_probs_5th = F.softmax(out, dim=1)[:, 1][0].detach().cpu().numpy()
                        del out
                        torch.cuda.empty_cache()

                        if ensemble_probs_5th is None:
                            ensemble_probs_5th = model_probs_5th / len(models_5th)
                        else:
                            ensemble_probs_5th += model_probs_5th / len(models_5th)
                        del model_probs_5th
                        torch.cuda.empty_cache()

                    del image_multichannel_5th
                    pred = (ensemble_probs_5th > PRED_THRESHOLD_5TH_STAGE).astype(np.uint8)
                    del ensemble_probs_5th
                    torch.cuda.empty_cache()

                    if iter_idx_5th < NUM_ITERATIONS_5TH_STAGE - 1:
                        current_mask_5th = pred.astype(np.float32)

                del image_normalized_5th, current_mask_5th
                torch.cuda.empty_cache()

            # ==================
            # 6TH STAGE: DeformNet refinement
            # ==================
            if NUM_STAGES >= 6 and deformnet_entries:
                print(f"      Running 6th stage (DeformNet) with {len(deformnet_entries)} model(s)...")
                pred_deform = predict_volume_deformnet(
                    deformnet_entries,
                    image,
                    pred.astype(np.float32),
                    device=device,
                    binarize_threshold=DEFORM_BINARIZE_THRESHOLD,
                    final_threshold=DEFORM_FINAL_THRESHOLD,
                    sliding_window_inferer=deform_swi,
                    deformnet_transforms_fn=deform_tfn,
                )
                print(f"      DeformNet complete. Positive ratio: {pred_deform.mean():.4f}")
                pred = pred_deform
                del pred_deform
            torch.cuda.empty_cache()

        if save_dir:
            np.save(save_dir / f"{sample_id}.npy", pred.astype(np.uint8, copy=False))

        if NUM_STAGES >= 2 or not CACHE_1ST_STAGE_ONLY:
            del image
        del pred
        torch.cuda.empty_cache()

    # Clean up models
    del models_1st, models_2nd, models_3rd, models_4th, models_5th
    if deformnet_entries:
        del deformnet_entries
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
    elif CACHE_4TH_STAGE_DIR:
        print(f"Mode: CACHE_4TH_STAGE – loading pre-computed 4th stage probs from {CACHE_4TH_STAGE_DIR}, running 5th refinement + DeformNet")
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
