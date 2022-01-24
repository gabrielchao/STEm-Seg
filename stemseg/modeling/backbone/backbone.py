# This script has been reproduced with slight modification from Facebook AI's maskrcnn-benchmark repository at:
# https://github.com/facebookresearch/maskrcnn-benchmark

from collections import OrderedDict

import torch
from torch import nn

from stemseg.modeling.backbone.make_layers import conv_with_kaiming_uniform
from stemseg.modeling.backbone import fpn as fpn_module
from stemseg.modeling.backbone import resnet
from stemseg.modeling.backbone.feature_fusion import LateFusion


def build_resnet_fpn_backbone(cfg):
    return ResnetFPNBackbone(cfg)

def build_multi_fusion_backbone(cfg):
    return MultiStageFusionBackbone(cfg)


def add_conv_channels(restore_dict: dict, name: str, original_channels: int, new_channels: int):
    """
    Add additional input channels to the state dict for the layer with the specified name.
    :param restore_dict: Model state dict
    :param name: Name of the layer to be augmented
    :param original_shape: Original shape of the target layer
    :param new_shape: New shape of the target layer after augmentation
    """
    assert restore_dict[name].shape[1] == original_channels
    assert new_channels > original_channels
    out_channels, original_channels, height, width = restore_dict[name].shape
    extra_channels = new_channels - original_channels
    pads = pads = torch.zeros((out_channels, extra_channels, height, width), dtype=restore_dict[name].dtype, device=restore_dict[name].device)
    nn.init.kaiming_uniform_(pads, a=1)
    restore_dict[name] = torch.cat([restore_dict[name], pads], 1)

def add_stem_in_channels(restore_dict: dict, in_channels: int):
    """
    Add additional input channels to the end of the first layer of the given state dict.
    :param restore_dict: Model state dict
    :param in_channels: Total number of input channels in the updated dict
    """
    # Original shape: (64, 3, 7, 7)
    add_conv_channels(restore_dict, 'body.stem.conv1.weight', 3, in_channels)


class ResnetFPNBackbone(nn.Module):
    """
    Backbone that consists of a Resnet followed by a FPN.
    """

    def __init__(self, cfg):
        super(ResnetFPNBackbone, self).__init__()
        self.in_channels = cfg.MODEL.RESNETS.STEM_IN_CHANNELS
        self.body = resnet.ResNet(cfg)
        in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        self.fpn = fpn_module.FPN(
            in_channels_list=[
                in_channels_stage2,
                in_channels_stage2 * 2,
                in_channels_stage2 * 4,
                in_channels_stage2 * 8,
            ],
            out_channels=out_channels,
            conv_block=conv_with_kaiming_uniform(
                cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
            ),
        )
        self.out_channels = out_channels
        self.is_3d = False
    
    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x
    
    def adapt_state_dict(self, restore_dict: dict, print_fn=None):
        if self.in_channels > 3:
            if print_fn:
                print_fn(f"Adapting backbone to {self.in_channels} input channels")
            add_stem_in_channels(restore_dict, self.in_channels)
        return restore_dict


class MultiStageFusionBackbone(nn.Module):
    """
    Backbone that fuses guidance maps both before and after the feature extractor.
    """

    def __init__(self, cfg):
        super(MultiStageFusionBackbone, self).__init__()

        # Initialize ResNet
        self.body = resnet.ResNet(cfg)
        self.in_channels = cfg.MODEL.RESNETS.STEM_IN_CHANNELS
        in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        self.feature_channels_list = [
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]

        # Initialize guidance fusion module
        guidance_channels = cfg.MODEL.RESNETS.STEM_IN_CHANNELS - 3
        guidance_inter_channels = cfg.MODEL.FUSION.GUIDANCE_INTER_CHANNELS
        self.fusion = LateFusion(guidance_channels, self.feature_channels_list, guidance_inter_channels)
        self.fused_feature_channels_list = [channels + guidance_inter_channels for channels in self.feature_channels_list]

        # Initialize FPN
        self.fpn = fpn_module.FPN(
            in_channels_list=self.fused_feature_channels_list,
            out_channels=out_channels,
            conv_block=conv_with_kaiming_uniform(
                cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
            ),
        )

        self.out_channels = out_channels
        self.is_3d = False
    
    def forward(self, full_tensor):
        """
        Arguments:
            full_tensor (Tensor): B x C x H x W tensor containing RGB and guidance maps concatenated
                in the channels dimension.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        guidance_tensor = full_tensor[:, 3:, :, :]

        feature_maps = self.body(full_tensor)
        x = self.fusion(guidance_tensor, feature_maps) # fused maps
        x = self.fpn(x)
        return x
    
    def adapt_state_dict(self, restore_dict: dict, print_fn=None):
        if self.in_channels > 3:
            if print_fn:
                print_fn(f"Adapting backbone to {self.in_channels} input channels")
            add_stem_in_channels(restore_dict, self.in_channels)

        # Add FPN channels
        if print_fn:
            print_fn(f"Adapting FPN to {self.fused_feature_channels_list} channels")
        names = ['fpn.fpn_inner1.weight', 'fpn.fpn_inner2.weight', 'fpn.fpn_inner3.weight', 'fpn.fpn_inner4.weight']
        for name, original_channels, new_channels in zip(names, self.feature_channels_list, self.fused_feature_channels_list):
            add_conv_channels(restore_dict, name, original_channels, new_channels)

        return restore_dict
