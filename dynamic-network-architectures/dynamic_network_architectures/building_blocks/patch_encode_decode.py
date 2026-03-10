from typing import Literal, Tuple
import torch
from torch import nn
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op
import numpy as np

from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD, StackedResidualBlocks
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks


block_type = Literal["basic", "bottleneck"]
block_style = Literal["residual", "conv"]


class LayerNormNd(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        idx = (None, slice(None), *([None] * (x.ndim - 2)))
        x = (
            self.weight[idx] * x
            + self.bias[idx]
        )
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    Loosely inspired by https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L364

    """

    def __init__(
        self,
        patch_size: Tuple[int, ...] = (16, 16, 16),
        input_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            patch_size (Tuple): patch size.
            padding (Tuple): padding size of the projection layer.
            input_channels (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = convert_dim_to_conv_op(len(patch_size))(
            input_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        returns shape (B, embed_dim, px, py, pz) where (px, py, pz) is patch_size.
        This output will need to be rearranged to whatever your transformer expects!
        """
        x = self.proj(x)
        return x


class PatchDecode(nn.Module):
    """
    Loosely inspired by SAM decoder
    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py#L53
    """

    def __init__(
        self,
        patch_size,
        embed_dim: int,
        out_channels: int,
        norm=LayerNormNd,
        activation=nn.GELU,
    ):
        """
        patch size must be 2^x, so 2, 4, 8, 16, 32, etc. Otherwise we die
        """
        super().__init__()

        def _round_to_8(inp):
            return int(max(8, np.round((inp + 1e-6) / 8) * 8))

        num_stages = int(np.log(max(patch_size)) / np.log(2))
        strides = [[2 if (p / 2**n) % 2 == 0 else 1 for p in patch_size] for n in range(num_stages)][::-1]
        dim_red = (embed_dim / (2 * out_channels)) ** (1 / num_stages)

        # don't question me
        channels = [embed_dim] + [_round_to_8(embed_dim / dim_red ** (x + 1)) for x in range(num_stages)]
        channels[-1] = out_channels

        stages = []
        for s in range(num_stages - 1):
            stages.append(
                nn.Sequential(
                    nn.ConvTranspose3d(channels[s], channels[s + 1], kernel_size=strides[s], stride=strides[s]),
                    norm(channels[s + 1]),
                    activation(),
                )
            )
        stages.append(nn.ConvTranspose3d(channels[-2], channels[-1], kernel_size=strides[-1], stride=strides[-1]))
        self.decode = nn.Sequential(*stages)

    def forward(self, x):
        """
        Expects input of shape (B, embed_dim, px, py, pz)! This will require you to reshape the output of your transformer!
        """
        return self.decode(x)


class PatchEmbed_deeper(nn.Module):
    """ResNet-style patch embedding with progressive downsampling"""

    def __init__(
        self,
        input_channels: int = 3,
        embed_dim: int = 864,
        base_features: int = 32,
        depth_per_level: tuple = (1, 1, 1),
        embed_proj_3x3x3: bool = False,
        embed_block_type: block_type = "basic",
        embed_block_style: block_style = "residual",
    ) -> None:
        """
        Iterative, convolutional patch embedding layer.

        Args:
            input_channels (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            base_features (int): Number of base features for the first stage.
            depth_per_level (tuple): Number of blocks per stage/level.
            embed_proj_3x3x3 (bool): Whether to use a 3x3x3 convolution for the final projection to embed_dim.
            embed_block_type (block_type): Type of residual block to use if embed_block_style is 'residual'.
            embed_block_style (block_style): Style of blocks to use in the embedding stages.
        """
        super().__init__()

        norm_op = nn.InstanceNorm3d
        block = BottleneckD if embed_block_type == "bottleneck" else BasicBlockD
        nonlin = nn.LeakyReLU if embed_block_type == "bottleneck" else nn.ReLU
        norm_op_kwargs = {"eps": 1e-5, "affine": True}
        nonlin_kwargs = {"inplace": True}

        if embed_block_type == "bottleneck":
            bottleneck_channels = base_features // 4
        else:
            bottleneck_channels = None

        if embed_block_style == "residual":
            self.stem = StackedResidualBlocks(
                1,
                nn.Conv3d,
                input_channels,
                base_features,
                [3, 3, 3],
                1,
                True,
                norm_op,
                norm_op_kwargs,
                None,
                None,
                nonlin,
                nonlin_kwargs,
                block=block,
            )
        elif embed_block_style == "conv":
            self.stem = StackedConvBlocks(
                1,
                nn.Conv3d,
                input_channels,
                base_features,
                [3, 3, 3],
                1,
                True,
                norm_op,
                norm_op_kwargs,
                None,
                None,
                nonlin,
                nonlin_kwargs,
            )
        else:
            raise ValueError(f"Unknown embed_block_style: {embed_block_style}. Must be 'residual' or 'conv'.")

        levels_needed = len(depth_per_level)

        self.stages = nn.ModuleList()
        input_ch = base_features

        for i in range(levels_needed):
            stride = 2
            output_channels = base_features * (2 ** i)
            if embed_block_style == "residual":
                if embed_block_type == "bottleneck":
                    bottleneck_channels = output_channels // 4
                else:
                    bottleneck_channels = None
                stage = StackedResidualBlocks(
                    n_blocks=depth_per_level[i],
                    conv_op=nn.Conv3d,
                    input_channels=input_ch,
                    output_channels=output_channels,
                    kernel_size=3,
                    initial_stride=stride,
                    conv_bias=False,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=None,
                    dropout_op_kwargs=None,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    block=block,
                    bottleneck_channels=bottleneck_channels,
                )
            elif embed_block_style == "conv":
                stage = StackedConvBlocks(
                    num_convs=depth_per_level[i],
                    conv_op=nn.Conv3d,
                    input_channels=input_ch,
                    output_channels=output_channels,
                    kernel_size=3,
                    initial_stride=stride,
                    conv_bias=False,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=None,
                    dropout_op_kwargs=None,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                )
            self.stages.append(stage)
            input_ch = output_channels

        final_proj_kernel = [3, 3, 3] if embed_proj_3x3x3 else [1, 1, 1]
        final_pad = [1, 1, 1] if embed_proj_3x3x3 else [0, 0, 0]
        self.final_proj = nn.Conv3d(
            input_ch, embed_dim, kernel_size=final_proj_kernel, stride=[1, 1, 1], padding=final_pad
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.final_proj(x)
        return x
