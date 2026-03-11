import torch
import torch.nn as nn
import torch.nn.functional as F
from .residual_unet import create_residual_unet


def make_base_grid(B, D, H, W, device):
    zz = torch.linspace(0, D-1, D, device=device)
    yy = torch.linspace(0, H-1, H, device=device)
    xx = torch.linspace(0, W-1, W, device=device)
    zz, yy, xx = torch.meshgrid(zz, yy, xx, indexing='ij')  # D,H,W
    grid = torch.stack((xx, yy, zz), dim=3)  # D,H,W,3 (x,y,z)
    grid = grid.unsqueeze(0).repeat(B,1,1,1,1)  # B,D,H,W,3
    return grid

def disp_to_grid_for_sampling(disp_voxel: torch.Tensor):
    B, C, D, H, W = disp_voxel.shape
    device = disp_voxel.device
    grid = make_base_grid(B, D, H, W, device)  # B,D,H,W,3 (x,y,z)
    disp = disp_voxel.permute(0,2,3,4,1)  # B,D,H,W,3 (dx,dy,dz)
    pos = grid + disp
    pos_norm = torch.empty_like(pos)
    pos_norm[...,0] = 2.0 * pos[...,0] / max(W-1,1) - 1.0  # x
    pos_norm[...,1] = 2.0 * pos[...,1] / max(H-1,1) - 1.0  # y
    pos_norm[...,2] = 2.0 * pos[...,2] / max(D-1,1) - 1.0  # z
    return pos_norm

def warp_vol_using_disp(vol: torch.Tensor, disp_voxel: torch.Tensor, mode='bilinear'):
    pos_norm = disp_to_grid_for_sampling(disp_voxel)
    warped = F.grid_sample(vol, pos_norm, mode=mode, padding_mode='border', align_corners=True)
    return warped

def warp_displacement(disp_voxel: torch.Tensor, by_disp_voxel: torch.Tensor):
    warped = warp_vol_using_disp(by_disp_voxel, disp_voxel, mode='bilinear')
    return warped

def scaling_and_squaring(v, n_steps=6) -> torch.Tensor:
    flow = v / (2.0 ** n_steps)
    for _ in range(n_steps):
        flowed = warp_displacement(flow, flow)
        flow = flow + flowed
    return flow

def soft_sdf(x, eps=1e-4):
    # x in [0,1]
    return torch.log(x + eps) - torch.log(1 - x + eps)

class TopoFix(nn.Module):
    def __init__(self, max_offset=2.0):
        super().__init__()
        self.max_offset = max_offset

    def forward(self, warped_mask, raw_t):
        """
        warped_mask: (B,1,D,H,W) in [0,1]
        raw_t: network raw 4th channel (B,1,D,H,W), can be positive or negative
        """
        sdf = soft_sdf(warped_mask)          # convert to SDF
        t = torch.sigmoid(raw_t)             # gate: where to apply
        delta = self.max_offset * torch.tanh(raw_t)  # signed magnitude
        sdf_corr = sdf + delta * t            # apply offset only where t>0
        corrected = torch.sigmoid(sdf_corr)  # back to probability
        return corrected, t, delta

class DiffeomorphicNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, n_steps, max_v, max_topo_offset):
        super().__init__()
        '''
        max_v: 1.5
        n_steps: 6
        lambda_jac: 0.3
        lambda_smooth: 0.05
        lambda_ce: 0.5
        lambda_dice: 1.5
        lambda_sparse: 0.1
        lambda_tv: 0.02
        lambda_boundary: 0.1
        max_topo_offset: 1.0
        '''
        self.predictor = create_residual_unet(
            in_channels=in_channels,
            out_channels=out_channels
        )
        self.max_v = max_v
        self.n_steps = n_steps
        self.topofix = TopoFix(max_offset=max_topo_offset)

    def forward(self, x, return_params=False):
        raw = self.predictor(x)
        raw_v = raw[:, :3]
        raw_t = raw[:, 3:4]

        # SVF
        v = torch.tanh(raw_v) * self.max_v
        phi = scaling_and_squaring(v, n_steps=self.n_steps)

        # warp
        warped = warp_vol_using_disp(x[:, 1:2], phi)

        # topo fix
        corrected, t, delta = self.topofix(warped, raw_t)

        if return_params:
            return corrected, v, phi, t
        return corrected