"""
Surface Dice implementation for topology-preserving segmentation.

Adapted from: clDice - A Novel Topology-Preserving Loss Function for Tubular
Structure Segmentation (membrain-seg implementation).

Original Authors: Johannes C. Paetzold and Suprosanna Shit
License: MIT License.
source code taken from 12_08_2025:  https://github.com/teamtomo/membrain-seg#

"""
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    """Apply soft erosion using min-pooling."""
    assert len(img.shape) == 5
    return -F.max_pool3d(-img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """Apply soft dilation using max-pooling."""
    assert len(img.shape) == 5
    return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(img: torch.Tensor) -> torch.Tensor:
    """Apply soft opening (erosion followed by dilation)."""
    return soft_dilate(soft_erode(img))


def soft_skel(img: torch.Tensor, iter_: int) -> torch.Tensor:
    """
    Compute soft skeleton by iterative erosion and opening.

    Parameters
    ----------
    img : torch.Tensor
        Input tensor (B, C, D, H, W).
    iter_ : int
        Number of skeletonization iterations.

    Returns
    -------
    torch.Tensor
        Soft skeleton tensor.
    """
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def gaussian_kernel_3d(size: int, sigma: float) -> torch.Tensor:
    """Create a 3D Gaussian kernel."""
    grid = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    x, y, z = torch.meshgrid(grid, grid, grid, indexing='ij')
    kernel = torch.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


# Cache for gaussian kernels
_gaussian_cache = {}


def apply_gaussian_filter(seg: torch.Tensor, kernel_size: int = 15, sigma: float = 2.0) -> torch.Tensor:
    """Apply 3D Gaussian filter to segmentation."""
    key = (kernel_size, sigma, seg.device)
    if key not in _gaussian_cache:
        kernel = gaussian_kernel_3d(kernel_size, sigma).to(seg.device)
        _gaussian_cache[key] = kernel.view(1, 1, kernel_size, kernel_size, kernel_size)

    g_kernel = _gaussian_cache[key]
    padding = kernel_size // 2
    return F.conv3d(seg, g_kernel, padding=padding, groups=seg.shape[1])


def get_gt_skeleton(gt_seg: torch.Tensor, ignore_label: int = 2, iterations: int = 5) -> torch.Tensor:
    """Generate smoothed skeleton from ground truth with ignore label support."""
    mask = (gt_seg != ignore_label).float()
    gt_binary = (gt_seg == 1).float()

    # Zero out ignore regions before filtering to prevent bleed-through
    gt_masked = gt_binary * mask
    gt_smooth = apply_gaussian_filter(gt_masked, kernel_size=15, sigma=2.0) * 1.5

    # Also mask the smoothed result to ensure no leakage
    gt_smooth = gt_smooth * mask
    return soft_skel(gt_smooth, iter_=iterations)


def masked_surface_dice(
    data: torch.Tensor,
    target: torch.Tensor,
    ignore_label: int = 2,
    soft_skel_iterations: int = 5,
    smooth: float = 1.0,
    reduction: str = "none",
) -> torch.Tensor:
    data = torch.sigmoid(data)
    mask = target != ignore_label
    target_binary = (target == 1).float()


    skel_pred = soft_skel(data.clone(), soft_skel_iterations)
    skel_true = get_gt_skeleton(target.clone(), ignore_label, soft_skel_iterations)

    skel_pred = skel_pred * mask.float()
    skel_true = skel_true * mask.float()

    tprec = (torch.sum(skel_pred * target_binary, dim=(1, 2, 3, 4)) + smooth) / \
            (torch.sum(skel_pred, dim=(1, 2, 3, 4)) + smooth)
    tsens = (torch.sum(skel_true * data, dim=(1, 2, 3, 4)) + smooth) / \
            (torch.sum(skel_true, dim=(1, 2, 3, 4)) + smooth)

    surf_dice = 2.0 * (tprec * tsens) / (tprec + tsens + 1e-8)

    if reduction == "mean":
        return surf_dice.mean()
    return surf_dice


class SurfaceDiceLoss(_Loss):
    """
    Surface Dice loss for topology-preserving segmentation.

    Encourages predictions to match the skeleton/surface structure
    of the ground truth, which is important for thin structures.

    Parameters
    ----------
    ignore_label : int
        Label to ignore (default 2).
    soft_skel_iterations : int
        Skeletonization iterations (default 5).
    smooth : float
        Smoothing factor (default 1.0).
    """

    def __init__(
        self,
        ignore_label: int = 2,
        soft_skel_iterations: int = 5,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.ignore_label = ignore_label
        self.soft_skel_iterations = soft_skel_iterations
        self.smooth = smooth

    def forward(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute 1 - surface_dice as loss."""
        surf_dice = masked_surface_dice(
            data=data,
            target=target,
            ignore_label=self.ignore_label,
            soft_skel_iterations=self.soft_skel_iterations,
            smooth=self.smooth,
            reduction="none",
        )
        return 1.0 - surf_dice