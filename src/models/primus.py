import torch.nn as nn
from dynamic_network_architectures.architectures.primus import PrimusB, PrimusV2B


def create_primus(
        in_channels=1,
        out_channels=2,
        input_shape = 160,
        patch_embed_size = 8,
        drop_path_rate = 0.0,
):
    model = PrimusB(in_channels,
                    out_channels,
                    (patch_embed_size, patch_embed_size, patch_embed_size),
                    (input_shape, input_shape, input_shape),
                    drop_path_rate=drop_path_rate)
    return model


def create_primus_v2(
        in_channels=1,
        out_channels=2,
        input_shape = 160,
        patch_embed_size = 8,
        drop_path_rate = 0.0,
):
    model = PrimusV2B(in_channels,
                      out_channels,
                      (patch_embed_size, patch_embed_size, patch_embed_size),
                      (input_shape, input_shape, input_shape),
                      drop_path_rate=drop_path_rate)
    return model