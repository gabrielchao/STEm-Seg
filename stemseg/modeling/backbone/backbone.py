# This script has been reproduced with slight modification from Facebook AI's maskrcnn-benchmark repository at:
# https://github.com/facebookresearch/maskrcnn-benchmark

from collections import OrderedDict

from torch import nn

from stemseg.modeling.backbone.make_layers import conv_with_kaiming_uniform
from stemseg.modeling.backbone import fpn as fpn_module
from stemseg.modeling.backbone import resnet
from stemseg.modeling.backbone.feature_fusion import LateFusion


def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
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
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    model.is_3d = False
    return model

def build_late_fusion_backbone(cfg):
    return Backbone(cfg)

class Backbone(nn.Module):
    def __init__(self, cfg):
        # Initialize ResNet
        self.body = resnet.ResNet(cfg)
        in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        feature_channels_list = [
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]

        # Initialize guidance fusion module
        guidance_channels = cfg.MODEL.RESNETS.STEM_IN_CHANNELS - 3
        guidance_inter_channels = cfg.MODEL.FUSION.GUIDANCE_INTER_CHANNELS
        self.fusion = LateFusion(guidance_channels, feature_channels_list, guidance_inter_channels)
        fused_feature_channels_list = [channels + guidance_inter_channels for channels in feature_channels_list]

        # Initialize FPN
        self.fpn = fpn_module.FPN(
            in_channels_list=fused_feature_channels_list,
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
