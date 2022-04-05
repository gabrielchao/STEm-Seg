# This script has been reproduced with slight modification from Facebook AI's maskrcnn-benchmark repository at:
# https://github.com/facebookresearch/maskrcnn-benchmark

from collections import OrderedDict

import torch
from torch import nn

from stemseg.modeling.backbone.make_layers import conv_with_kaiming_uniform
from stemseg.modeling.backbone import fpn as fpn_module
from stemseg.modeling.backbone import resnet
from stemseg.modeling.backbone.feature_fusion import GuidanceEncoder
from stemseg.modeling.common import AdaptationError, add_conv_channels


def build_resnet_fpn_backbone(cfg):
    return ResnetFPNBackbone(cfg)

def build_multi_fusion_backbone(cfg):
    return MultiStageFusionBackbone(cfg)

def add_stem_in_channels(restore_dict: dict, in_channels: int, prefix=''):
    """
    Add additional input channels to the end of the first layer of the given state dict.
    :param restore_dict: Model state dict
    :param in_channels: Total number of input channels in the updated dict
    :param prefix: Prefix to account for before layer names
    """
    # Original shape: (64, 3, 7, 7)
    try:
        add_conv_channels(restore_dict, prefix + 'body.stem.conv1.weight', 3, in_channels)
    except AdaptationError as error:
        # it's ok if the channels are already there
        print(error)


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
        """
        Arguments:
            x (Tensor): B x C x H x W tensor containing RGB images.
        Returns:
            results (dict): {
                'features': Feature maps after FPN layers. They are ordered from highest resolution first.
            }
        """
        x = self.body(x)
        x = self.fpn(x)
        return {'features': x}
    
    def adapt_state_dict(self, restore_dict: dict, print_fn=None, prefix=''):
        if self.in_channels > 3:
            if print_fn:
                print_fn(f"Adapting backbone to {self.in_channels} input channels")
            add_stem_in_channels(restore_dict, self.in_channels, prefix)
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
        self.fusion = GuidanceEncoder(guidance_channels, self.feature_channels_list, guidance_inter_channels)
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
            results (dict): {
                'features': tuple[Tensor]. Feature maps after FPN layers. They are ordered from highest resolution first.
                'guidance': list[Tensor]. Multi-scale guidance maps after the guidance encoder.
            }
        """
        guidance_tensor = full_tensor[:, 3:, :, :]

        feature_maps = self.body(full_tensor)
        guidance_maps = self.fusion(guidance_tensor) # fused maps
        assert len(feature_maps) == len(guidance_maps)
        
        # Perform late fusion on feature and guidance maps
        for i in range(len(feature_maps)):
            feature_maps[i] = torch.cat([feature_maps[i], guidance_maps[i]], 1)

        feature_maps = self.fpn(feature_maps)
        return {'features': feature_maps, 'guidance': guidance_maps}
    
    def adapt_state_dict(self, restore_dict: dict, print_fn=None, prefix=''):
        if self.in_channels > 3:
            if print_fn:
                print_fn(f"Adapting backbone to {self.in_channels} input channels")
            add_stem_in_channels(restore_dict, self.in_channels, prefix)

        # Add FPN channels
        if print_fn:
            print_fn(f"Adapting FPN to {self.fused_feature_channels_list} channels")
        names = ['fpn.fpn_inner1.weight', 'fpn.fpn_inner2.weight', 'fpn.fpn_inner3.weight', 'fpn.fpn_inner4.weight']
        for name, original_channels, new_channels in zip(names, self.feature_channels_list, self.fused_feature_channels_list):
            try:
                add_conv_channels(restore_dict, prefix + name, original_channels, new_channels)
            except AdaptationError as error:
                # it's ok if the channels are already there
                print(error)

        return restore_dict
