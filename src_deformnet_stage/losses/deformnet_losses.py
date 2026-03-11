import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_blur_3d(x, kernel_size=3, sigma=5.0):
    """
    x: (B, C, D, H, W)
    """
    B, C, D, H, W = x.shape
    kernel = gaussian_kernel_3d(kernel_size, sigma, device=x.device)

    # shape: (C, 1, K, K, K)
    kernel = kernel.expand(C, 1, kernel_size, kernel_size, kernel_size)

    # depthwise convolution
    return F.conv3d(x, kernel, padding=kernel_size // 2, groups=C)


def gaussian_kernel_3d(kernel_size=5, sigma=1.0, device="cuda"):
    """Returns a normalized 3D Gaussian kernel (1,1,K,K,K)."""
    ax = torch.arange(kernel_size, device=device) - kernel_size // 2
    xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


# ----------------------------------------
# Smoothness: ||∇v||² on the SVF
# ----------------------------------------
def svf_smoothness(v):
    return (
            (v[:, :, 1:] - v[:, :, :-1]).pow(2).mean() +
            (v[:, :, :, 1:] - v[:, :, :, :-1]).pow(2).mean() +
            (v[:, :, :, :, 1:] - v[:, :, :, :, :-1]).pow(2).mean()
    ) / 3.0


def jacobian_log_barrier(flow, eps=1e-6):
    """
    flow: (B, 3, D, H, W) displacement field u(x)
    returns: log-barrier jacobian loss
    """
    det = jacobian_determinant(flow)
    # clamp to avoid log(0) or negative numbers
    det_clamped = torch.clamp(det, min=eps)
    loss = -torch.log(det_clamped).mean()
    return loss


# ---------------------
# Jacobian Determinant
# ---------------------
def jacobian_determinant(flow):
    """
    flow: (B, 3, D, H, W) displacement field u(x)
    returns: (B, D, H, W) jacobian determinant of φ(x)=x+u(x)
    """
    B, C, D, H, W = flow.shape
    assert C == 3

    # gradients wrt spatial axes (z = depth, y = height, x = width)
    du_dx = torch.gradient(flow, dim=4)[0]  # width axis
    du_dy = torch.gradient(flow, dim=3)[0]  # height axis
    du_dz = torch.gradient(flow, dim=2)[0]  # depth axis

    # components
    ux_x = du_dx[:,0]; ux_y = du_dy[:,0]; ux_z = du_dz[:,0]
    uy_x = du_dx[:,1]; uy_y = du_dy[:,1]; uy_z = du_dz[:,1]
    uz_x = du_dx[:,2]; uz_y = du_dy[:,2]; uz_z = du_dz[:,2]

    # deformation gradient J = I + ∇u
    j11 = 1 + ux_x; j12 =     ux_y; j13 =     ux_z
    j21 =     uy_x; j22 = 1 + uy_y; j23 =     uy_z
    j31 =     uz_x; j32 =     uz_y; j33 = 1 + uz_z

    det = (
        j11 * (j22 * j33 - j23 * j32)
        - j12 * (j21 * j33 - j23 * j31)
        + j13 * (j21 * j32 - j22 * j31)
    )
    return det


class SkeletonRecallLoss(nn.Module):
    """
    Ensures the model 'recalls' the thin centerline/skeleton of the sheet.
    """

    def __init__(self, ignore_index=2, smooth=1e-5):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, probs, target_skeleton, original_labels):
        # Create mask to exclude ignore_index pixels from loss
        mask = (original_labels != self.ignore_index).float()

        # We only care about the recall on the skeleton voxels
        # skeleton is 1-voxel thin, so we want the model to be high there
        # We multiply by mask to ensure we don't penalize in ignore regions
        active_skeleton = target_skeleton * mask

        # Weighted recall: focus only on skeleton points
        numerator = torch.sum(probs * active_skeleton, dim=(1, 2, 3, 4))
        denominator = torch.sum(active_skeleton, dim=(1, 2, 3, 4))

        # Avoid division by zero if a patch has no skeleton
        recall = (numerator + self.smooth) / (denominator + self.smooth)

        return 1.0 - recall.mean()


def topo_sparsity(t):
    return t.mean()


def topo_tv(t):
    return (
        (t[:,:,1:] - t[:,:,:-1]).abs().mean() +
        (t[:,:,:,1:] - t[:,:,:,:-1]).abs().mean() +
        (t[:,:,:,:,1:] - t[:,:,:,:,:-1]).abs().mean()
    ) / 3.0


def topo_boundary(t, warped):
    # encourage topo edits near surface only
    boundary = warped * (1.0 - warped)
    return (t * (1.0 - boundary)).mean()
