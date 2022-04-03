from stemseg.modeling.common import UpsampleTrilinear3D, AtrousPyramid3D, get_pooling_layer_creator, \
    get_temporal_scales, add_conv_channels_3d
from stemseg.utils.global_registry import GlobalRegistry

import torch
import torch.nn as nn

SEEDINESS_HEAD_REGISTRY = GlobalRegistry.get("SeedinessHead")


@SEEDINESS_HEAD_REGISTRY.add("squeeze_expand_decoder")
class SqueezingExpandDecoder(nn.Module):
    def __init__(self, in_channels, inter_channels, ConvType=nn.Conv3d, PoolType=nn.AvgPool3d, NormType=nn.Identity, **kwargs):
        super().__init__()

        PoolingLayerCallbacks = get_pooling_layer_creator(PoolType)

        self.block_32x = nn.Sequential(
            ConvType(in_channels, inter_channels[0], 3, stride=1, padding=1),
            NormType(inter_channels[0]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[0](3, stride=(2, 1, 1), padding=1),

            ConvType(inter_channels[0], inter_channels[0], 3, stride=1, padding=1),
            NormType(inter_channels[0]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[1](3, stride=(2, 1, 1), padding=1),
            ConvType(inter_channels[0], inter_channels[0], 3, stride=1, padding=1),

            NormType(inter_channels[0]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[2](3, stride=(2, 1, 1), padding=1),
        )

        self.block_16x = nn.Sequential(
            ConvType(in_channels, inter_channels[1], 3, stride=1, padding=1),
            NormType(inter_channels[1]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[0](3, stride=(2, 1, 1), padding=1),

            ConvType(inter_channels[1], inter_channels[1], 3, stride=1, padding=1),
            NormType(inter_channels[1]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[1](3, stride=(2, 1, 1), padding=1),
        )

        self.block_8x = nn.Sequential(
            ConvType(in_channels, inter_channels[2], 3, stride=1, padding=1),
            NormType(inter_channels[2]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[0](3, stride=(2, 1, 1), padding=1),
        )

        self.block_4x = nn.Sequential(
            ConvType(in_channels, inter_channels[3], 3, stride=1, padding=1),
            NormType(inter_channels[3]),
            nn.ReLU(inplace=True)
        )

        t_scales = get_temporal_scales()

        # 32x -> 16x
        self.upsample_32_to_16 = nn.Sequential(
            UpsampleTrilinear3D(scale_factor=(t_scales[0], 2, 2), align_corners=False),
        )
        self.conv_16 = nn.Conv3d(inter_channels[0] + inter_channels[1], inter_channels[1], 1, bias=False)

        # 16x to 8x
        self.upsample_16_to_8 = nn.Sequential(
            UpsampleTrilinear3D(scale_factor=(t_scales[1], 2, 2), align_corners=False)
        )
        self.conv_8 = nn.Conv3d(inter_channels[1] + inter_channels[2], inter_channels[2], 1, bias=False)

        # 8x to 4x
        self.upsample_8_to_4 = nn.Sequential(
            UpsampleTrilinear3D(scale_factor=(t_scales[2], 2, 2), align_corners=False)
        )
        self.conv_4 = nn.Conv3d(inter_channels[2] + inter_channels[3], inter_channels[3], 1, bias=False)

        self.conv_out = nn.Conv3d(inter_channels[3], 1, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        """
        :param x: list of multiscale feature map tensors of shape [N, C, T, H, W]. For this implementation, there
        should be 4 features maps in increasing order of spatial dimensions
        :return: embedding map of shape [N, E, T, H, W]
        """
        assert len(x) == 4

        feat_map_32x, feat_map_16x, feat_map_8x, feat_map_4x = x

        feat_map_32x = self.block_32x(feat_map_32x)

        # 32x to 16x
        x = self.upsample_32_to_16(feat_map_32x)
        feat_map_16x = self.block_16x(feat_map_16x)
        x = torch.cat((x, feat_map_16x), 1)
        x = self.conv_16(x)

        # 16x to 8x
        x = self.upsample_16_to_8(x)
        feat_map_8x = self.block_8x(feat_map_8x)
        x = torch.cat((x, feat_map_8x), 1)
        x = self.conv_8(x)

        # 8x to 4x
        x = self.upsample_8_to_4(x)
        feat_map_4x = self.block_4x(feat_map_4x)
        x = torch.cat((x, feat_map_4x), 1)
        x = self.conv_4(x)

        return self.conv_out(x).sigmoid()
    
    def adapt_state_dict(self, restore_dict: dict, print_fn=None):
        # This is the original seediness head, nothing needs to be done.
        pass


@SEEDINESS_HEAD_REGISTRY.add("fusion_decoder")
class FusionDecoder(nn.Module):
    """
    Squeeze-expand decoder that also fuses guidance maps at each scale.
    """
    def __init__(self, in_channels, inter_channels, guidance_channels, ConvType=nn.Conv3d, PoolType=nn.AvgPool3d, NormType=nn.Identity):
        super().__init__()

        PoolingLayerCallbacks = get_pooling_layer_creator(PoolType)

        fused_channels = in_channels + guidance_channels

        self.block_32x = nn.Sequential(
            ConvType(fused_channels, inter_channels[0], 3, stride=1, padding=1),
            NormType(inter_channels[0]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[0](3, stride=(2, 1, 1), padding=1),

            ConvType(inter_channels[0], inter_channels[0], 3, stride=1, padding=1),
            NormType(inter_channels[0]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[1](3, stride=(2, 1, 1), padding=1),
            ConvType(inter_channels[0], inter_channels[0], 3, stride=1, padding=1),

            NormType(inter_channels[0]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[2](3, stride=(2, 1, 1), padding=1),
        )

        self.block_16x = nn.Sequential(
            ConvType(fused_channels, inter_channels[1], 3, stride=1, padding=1),
            NormType(inter_channels[1]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[0](3, stride=(2, 1, 1), padding=1),

            ConvType(inter_channels[1], inter_channels[1], 3, stride=1, padding=1),
            NormType(inter_channels[1]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[1](3, stride=(2, 1, 1), padding=1),
        )

        self.block_8x = nn.Sequential(
            ConvType(fused_channels, inter_channels[2], 3, stride=1, padding=1),
            NormType(inter_channels[2]),
            nn.ReLU(inplace=True),
            PoolingLayerCallbacks[0](3, stride=(2, 1, 1), padding=1),
        )

        self.block_4x = nn.Sequential(
            ConvType(fused_channels, inter_channels[3], 3, stride=1, padding=1),
            NormType(inter_channels[3]),
            nn.ReLU(inplace=True)
        )

        t_scales = get_temporal_scales()

        # 32x -> 16x
        self.upsample_32_to_16 = nn.Sequential(
            UpsampleTrilinear3D(scale_factor=(t_scales[0], 2, 2), align_corners=False),
        )
        self.conv_16 = nn.Conv3d(inter_channels[0] + inter_channels[1], inter_channels[1], 1, bias=False)

        # 16x to 8x
        self.upsample_16_to_8 = nn.Sequential(
            UpsampleTrilinear3D(scale_factor=(t_scales[1], 2, 2), align_corners=False)
        )
        self.conv_8 = nn.Conv3d(inter_channels[1] + inter_channels[2], inter_channels[2], 1, bias=False)

        # 8x to 4x
        self.upsample_8_to_4 = nn.Sequential(
            UpsampleTrilinear3D(scale_factor=(t_scales[2], 2, 2), align_corners=False)
        )
        self.conv_4 = nn.Conv3d(inter_channels[2] + inter_channels[3], inter_channels[3], 1, bias=False)

        self.conv_out = nn.Conv3d(inter_channels[3], 1, kernel_size=1, padding=0, bias=False)

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.fused_channels = fused_channels

    def forward(self, features, guidance):
        """
        :param features: list of multiscale feature map tensors of shape [N, C, T, H, W]. For this implementation, there
        should be 4 features maps in increasing order of spatial dimensions
        :param guidance: list of multiscale guidance map tensors
        :return: embedding map of shape [N, E, T, H, W]
        """
        assert len(features) == 4

        feat_map_32x, feat_map_16x, feat_map_8x, feat_map_4x = features
        g_map_32x, g_map_16x, g_map_8x, g_map_4x = guidance
        
        fused_map_32x = torch.cat([feat_map_32x, g_map_32x], 1)
        fused_map_16x = torch.cat([feat_map_16x, g_map_16x], 1)
        fused_map_8x = torch.cat([feat_map_8x, g_map_8x], 1)
        fused_map_4x = torch.cat([feat_map_4x, g_map_4x], 1)

        fused_map_32x = self.block_32x(fused_map_32x)

        # 32x to 16x
        x = self.upsample_32_to_16(fused_map_32x)
        fused_map_16x = self.block_16x(fused_map_16x)
        x = torch.cat((x, fused_map_16x), 1)
        x = self.conv_16(x)

        # 16x to 8x
        x = self.upsample_16_to_8(x)
        fused_map_8x = self.block_8x(fused_map_8x)
        x = torch.cat((x, fused_map_8x), 1)
        x = self.conv_8(x)

        # 8x to 4x
        x = self.upsample_8_to_4(x)
        fused_map_4x = self.block_4x(fused_map_4x)
        x = torch.cat((x, fused_map_4x), 1)
        x = self.conv_4(x)

        return self.conv_out(x).sigmoid()

    def adapt_state_dict(self, restore_dict: dict, print_fn=None):
        # Add extra input channels for first conv layer of each block
        fused_feature_channels_list = [self.fused_channels] * 4
        if print_fn:
            print_fn(f"Adapting seediness decoder to {fused_feature_channels_list} input channels")
        names = ['seediness_head.block_32x.0.weight', 'seediness_head.block_16x.0.weight', 'seediness_head.block_8x.0.weight', 'seediness_head.block_4x.0.weight']
        for name, inter_channels, new_channels in zip(names, self.inter_channels, fused_feature_channels_list):
            assert inter_channels == restore_dict[name].shape[0], \
                f"Mismatch in fusion decoder inter channels: expected {inter_channels} but \
                    got {restore_dict[name].shape[0]} in layer {name}"
            add_conv_channels_3d(restore_dict, name, self.in_channels, new_channels)
