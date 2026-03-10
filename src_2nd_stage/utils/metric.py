"""
Vesuvius competition metric helpers.

Supports two modes:
- **Kaggle**: auto-install `topometrics` from `/kaggle/input/vesuvius-metric-resources`
- **Local**: uses `topometrics` if installed, otherwise raises error.
"""
import glob
import importlib
import os
import subprocess
import sys
import numpy as np
import pandas as pd

from .io import load_volume


class ParticipantVisibleError(Exception):
    pass


class HostVisibleError(Exception):
    pass


def install_dependencies():
    """On Kaggle, the topometrics library must be installed during the run. This function handles the entire process."""
    try:
        import topometrics.leaderboard

        return None
    # The broad exception is necessary as the initial import can fail for multiple reasons.
    except:
        pass

    # Only attempt the Kaggle auto-install if we appear to be in a Kaggle filesystem.
    if not (os.path.exists("/kaggle/input") and os.path.exists("/kaggle/working")):
        raise HostVisibleError(
            "topometrics is not installed. For local scoring either:\n"
            "- install `topometrics` in your environment, or\n"
            "- run on Kaggle where the metric resources are available.\n"
            "For training-time monitoring without topometrics, use the proxy metrics in "
            "`src/utils/leaderboard_metrics.py`."
        )

    resources_dir = os.environ.get("VESUVIUS_METRIC_RESOURCES_DIR", "/kaggle/input/vesuvius-metric-resources")
    install_dir = os.environ.get("VESUVIUS_TOPO_INSTALL_DIR", "/kaggle/working/topological-metrics-kaggle")

    try:
        subprocess.run(
            f'cd {resources_dir} && uv pip install --no-index --find-links=wheels -r topological-metrics-kaggle/requirements.txt',
            shell=True,
            check=True,
        )
        subprocess.run(f'cd /kaggle/working && cp -r {resources_dir}/topological-metrics-kaggle .', shell=True, check=True)
        subprocess.run(
            f'cd {install_dir} && chmod +x scripts/setup_submodules.sh scripts/build_betti.sh && make build-betti',
            shell=True,
            check=True,
        )
        subprocess.run(
            f'cd {install_dir} && uv pip install -e . --no-deps --no-index --no-build-isolation -v',
            shell=True,
            check=True,
        )
        # Add the new library to Python's path and invalidate caches to ensure it's found.
        sys.path.append('/kaggle/working/topological-metrics-kaggle/src')
        importlib.invalidate_caches()

    except Exception as err:
        raise HostVisibleError(f'Failed to install topometrics library: {err}')


def generate_standard_submission(submission_dir: str) -> None:
    # Dependencies installed here as generate_standard_submission is the first metric function that gets called by the orchestrator.
    submission_tifs = glob.glob(f'{submission_dir}/**/*.tif', recursive=True)
    if len(submission_tifs) == 0:
        submission_tifs = glob.glob('/kaggle/tmp/**/*.tif', recursive=True)
    if len(submission_tifs) == 0:
        raise ParticipantVisibleError('No submission files found')
    df = pd.DataFrame({'tif_paths': submission_tifs})
    df['id'] = df['tif_paths'].apply(lambda x: x.split('/')[-1].split('.')[0])
    os.chdir('/kaggle/working')
    df[['id', 'tif_paths']].to_csv('submission.csv', index=False)


def score_single_tif(
    gt_path,
    pred_path,
    surface_tolerance,
    voi_connectivity=26,
    voi_transform='one_over_one_plus',
    voi_alpha=0.3,
    topo_weight=0.3,
    surface_dice_weight=0.35,
    voi_weight=0.35,
):
    gt: np.ndarray = load_volume(gt_path)
    pr: np.ndarray = load_volume(pred_path)

    install_dependencies()
    # The import is here to ensure dependencies are loaded first.
    try:
        # Use a standard import now that the path is reliably set.
        import topometrics.leaderboard
    except Exception as err:
        raise HostVisibleError(f'Failed to import topometrics after installation: {err}')

    score_report = topometrics.leaderboard.compute_leaderboard_score(
        predictions=pr,
        labels=gt,
        dims=(0, 1, 2),
        spacing=(1.0, 1.0, 1.0),  # (z, y, x)
        surface_tolerance=surface_tolerance,  # in spacing units
        voi_connectivity=voi_connectivity,
        voi_transform=voi_transform,
        voi_alpha=voi_alpha,
        combine_weights=(topo_weight, surface_dice_weight, voi_weight),  # (Topo, SurfaceDice, VOI)
        fg_threshold=None,  # None => legacy "!= 0"; else uses "x > threshold"
        ignore_label=2,  # voxels with this GT label are ignored
        ignore_mask=None,  # or pass an explicit boolean mask
    )
    return np.clip(score_report.score, a_min=0.0, a_max=1.0)


def try_score_arrays(
    gt: np.ndarray,
    pr: np.ndarray,
    surface_tolerance: float = 2.0,
    spacing=(1.0, 1.0, 1.0),
    voi_connectivity: int = 26,
    voi_transform: str = "one_over_one_plus",
    voi_alpha: float = 0.3,
    topo_weight: float = 0.3,
    surface_dice_weight: float = 0.35,
    voi_weight: float = 0.35,
):
    """
    Compute the official leaderboard score for in-memory arrays.

    Returns:
        score_report (topometrics.leaderboard.LeaderboardScoreReport) if available, else None.
    """
    try:
        import topometrics.leaderboard  # type: ignore
    except Exception:
        # Try Kaggle install path if applicable
        try:
            install_dependencies()
            import topometrics.leaderboard  # type: ignore
        except Exception:
            return None

    score_report = topometrics.leaderboard.compute_leaderboard_score(
        predictions=pr,
        labels=gt,
        dims=(0, 1, 2),
        spacing=spacing,
        surface_tolerance=surface_tolerance,
        voi_connectivity=voi_connectivity,
        voi_transform=voi_transform,
        voi_alpha=voi_alpha,
        combine_weights=(topo_weight, surface_dice_weight, voi_weight),
        fg_threshold=None,
        ignore_label=2,
        ignore_mask=None,
    )
    return score_report


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    surface_tolerance: float = 2.0,
    voi_connectivity: int = 26,
    voi_transform: str = 'one_over_one_plus',
    voi_alpha: float = 0.3,
    topo_weight: float = 0.3,
    surface_dice_weight: float = 0.35,
    voi_weight: float = 0.35,
) -> float:
    """Returns the mean per-volume Topological Score, Surface Dice, and VOI Scores."""
    if not solution['tif_paths'].apply(os.path.exists).all():
        raise HostVisibleError('Invalid solution file paths')

    solution['pred_paths'] = submission['tif_paths']
    solution['image_score'] = solution.apply(
        lambda row: score_single_tif(
            row['tif_paths'],
            row['pred_paths'],
            surface_tolerance,
            voi_connectivity=voi_connectivity,
            voi_transform=voi_transform,
            voi_alpha=voi_alpha,
            topo_weight=topo_weight,
            surface_dice_weight=surface_dice_weight,
            voi_weight=voi_weight,
        ),
        axis=1,
    )
    return float(np.mean(solution['image_score']))