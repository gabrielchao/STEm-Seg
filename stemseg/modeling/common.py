from stemseg.config import cfg

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_pooling_layer_creator(PoolType):
    def pooling_module_creator(*args, **kwargs):
        return PoolType(*args, **kwargs)

    def identity_module_creator(*args, **kwargs):
        return nn.Identity(*args, **kwargs)

    if cfg.INPUT.NUM_FRAMES == 2:
        return [identity_module_creator for _ in range(3)]
    elif cfg.INPUT.NUM_FRAMES == 4:
        return [pooling_module_creator] + [identity_module_creator for _ in range(2)]
    elif cfg.INPUT.NUM_FRAMES == 8:
        return [pooling_module_creator for _ in range(2)] + [identity_module_creator]
    elif cfg.INPUT.NUM_FRAMES in (16, 24, 32):
        return [pooling_module_creator for _ in range(3)]
    else:
        raise NotImplementedError()


def get_temporal_scales():
    if cfg.INPUT.NUM_FRAMES == 2:
        return [1, 1, 1]
    elif cfg.INPUT.NUM_FRAMES == 4:
        return [1, 1, 2]
    elif cfg.INPUT.NUM_FRAMES == 8:
        return [1, 2, 2]
    elif cfg.INPUT.NUM_FRAMES in (16, 24, 32):
        return [2, 2, 2]


class AtrousPyramid3D(nn.Module):
    def __init__(self, in_channels, pyramid_channels,  dilation_rates, out_channels=None, include_1x1_conv=True):
        super().__init__()

        pyramid_channels = [pyramid_channels] * len(dilation_rates)

        atrous_convs = [
            nn.Conv3d(in_channels, channels, 3, padding=rate, dilation=rate, bias=False)
            for (channels, rate) in zip(pyramid_channels, dilation_rates)
        ]
        if include_1x1_conv:
            atrous_convs.append(nn.Conv3d(in_channels, pyramid_channels[0], 1, bias=False))
            total_channels = sum(pyramid_channels) + pyramid_channels[0]
        else:
            total_channels = sum(pyramid_channels)

        self.atrous_convs = nn.ModuleList(atrous_convs)

        if out_channels:
            self.conv_out = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv3d(total_channels, out_channels, 1, bias=False)
            )
        else:
            self.conv_out = nn.Identity()

    def forward(self, x):
        x = torch.cat([conv(x) for conv in self.atrous_convs], dim=1)
        return self.conv_out(x)


class UpsampleTrilinear3D(nn.Module):
    def __init__(self, size=None, scale_factor=None, align_corners=None):
        super().__init__()

        self.size = size
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, self.size, self.scale_factor, mode='trilinear', align_corners=self.align_corners)


def add_conv_channels(restore_dict: dict, name: str, original_channels: int, new_channels: int):
    """
    Add additional input channels to the state dict for the layer with the specified name.
    :param restore_dict: Model state dict
    :param name: Name of the layer to be augmented
    :param original_channels: Original number of input channels for the target layer
    :param new_channels: New number of input channels for the target layer after augmentation
    """
    assert restore_dict[name].shape[1] == original_channels
    assert new_channels > original_channels
    out_channels, original_channels, height, width = restore_dict[name].shape
    extra_channels = new_channels - original_channels
    pads = pads = torch.zeros((out_channels, extra_channels, height, width), dtype=restore_dict[name].dtype, device=restore_dict[name].device)
    nn.init.kaiming_uniform_(pads, a=1)
    restore_dict[name] = torch.cat([restore_dict[name], pads], 1)


def add_conv_channels_3d(restore_dict: dict, name: str, original_channels: int, new_channels: int):
    """
    Add additional input channels to the state dict for the 3D conv layer with the specified name.
    :param restore_dict: Model state dict
    :param name: Name of the layer to be augmented
    :param original_channels: Original number of input channels for the target layer
    :param new_channels: New number of input channels for the target layer after augmentation
    """
    assert restore_dict[name].shape[1] == original_channels
    assert new_channels > original_channels
    out_channels, original_channels, height, width, depth = restore_dict[name].shape
    extra_channels = new_channels - original_channels
    pads = pads = torch.zeros((out_channels, extra_channels, height, width, depth), dtype=restore_dict[name].dtype, device=restore_dict[name].device)
    nn.init.kaiming_uniform_(pads, a=1)
    restore_dict[name] = torch.cat([restore_dict[name], pads], 1)
