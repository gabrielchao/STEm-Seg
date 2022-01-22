import torch
import torch.nn.functional as F
from torch import nn

from stemseg.modeling.backbone.make_layers import conv_with_kaiming_uniform

class LateFusion(nn.Module):
    """
    Module that fuses guidance maps with the multi-scale feature maps produced by the backbone.
    """

    def __init__(self, guidance_channels, feature_channels_list, inter_channels=256):
        """
        Arguments:
            guidance_channels (int): number of guidance map channels
            feature_channels_list (list[int]): number of channels for each feature map that
                will be fed
            inter_channels (int): number of intermediate channels for the guidance maps
        """
        self.feature_channels_list = feature_channels_list

        self.block_4x = SEBasicBlock(guidance_channels, inter_channels, stride=4)
        self.block_8x = SEBasicBlock(inter_channels, inter_channels, stride=2)
        self.block_16x = SEBasicBlock(inter_channels, inter_channels, stride=2)
        self.block_32x = SEBasicBlock(inter_channels, inter_channels, stride=2)

    def forward(self, guidance_maps, feature_maps):
        """
        Arguments:
            guidance_maps (Tensor): B x C x H x W
            feature_maps (list(Tensor)): feature maps from the ResNet for each feature level.
        Returns:
            feature_maps (list): fused feature maps for each feature level.
        """
        
        g_map_4x = self.block_4x(guidance_maps)
        g_map_8x = self.block_8x(g_map_4x)
        g_map_16x = self.block_16x(g_map_8x)
        g_map_32x = self.block_32x(g_map_16x)
        guidance_maps = [g_map_4x, g_map_8x, g_map_16x, g_map_32x]
        assert len(guidance_maps) == len(feature_maps)
        
        for i in range(len(feature_maps)):
            feature_maps[i] = torch.cat([feature_maps[i], guidance_maps[i]])
        
        return feature_maps


class SEBasicBlock(nn.Module):
    """
    Full squeeze and excitation block with convolutions and residual.
    Adapted from https://amaarora.github.io/2020/07/24/SeNet.html
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, r=16):
        super(SEBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(inplanes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        # add SE block
        self.se = SE_Block(planes, r)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # add SE operation
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SE_Block(nn.Module):
    """
    Squeeze and excitation block.
    Adapted from https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    """
    def __init__(self, channel, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return conv_with_kaiming_uniform(False, False)(in_planes, out_planes,
                    kernel_size=3, stride=stride, dilation=dilation)
    
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=dilation, groups=groups, bias=False, dilation=dilation)
