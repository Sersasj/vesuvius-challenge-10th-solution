"""Residual Encoder U-Net wrapper using dynamic-network-architectures."""
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet


def create_residual_unet(
    in_channels=1,
    out_channels=2,
    channels=(32, 64, 128, 256, 320, 320),
    strides=(1, 2, 2, 2, 2, 2),
    n_blocks_per_stage=(1, 3, 4, 6, 6, 6),
    deep_supervision=False,
):
    """Create ResidualEncoderUNet matching nnUNet 3d_fullres configuration.

    Args:
        in_channels: Input channels (default: 1)
        out_channels: Output channels (default: 2)
        channels: Feature channels at each level (tuple of ints)
        strides: Strides for downsampling at each level (tuple of ints)
        n_blocks_per_stage: Number of residual blocks per stage (tuple of ints)
        deep_supervision: Whether to use deep supervision

    Returns:
        ResidualEncoderUNet model
    """
    # Number of stages in decoder is len(channels) - 1
    n_conv_per_stage_decoder = [1] * (len(channels) - 1)

    model = ResidualEncoderUNet(
        input_channels=in_channels,
        n_stages=len(channels),
        features_per_stage=channels,
        conv_op=nn.Conv3d,
        kernel_sizes=3,
        strides=strides,
        n_blocks_per_stage=n_blocks_per_stage,
        num_classes=out_channels,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={},
        dropout_op=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=deep_supervision,
    )
    return model

