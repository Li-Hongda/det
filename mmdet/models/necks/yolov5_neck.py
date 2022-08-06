import math
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule
from ..builder import NECKS

class YOLOV5BottleNeck(BaseModule):
    """The basic residual unit used in YOLOV5-v6.1 CSPBlock.
    Args:
        in_channels (int): The input channels of this Module.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Default: False
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
    """

    def __init__(self,
                in_channels,
                use_depthwise=False,
                conv_cfg=None,
                norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                act_cfg=dict(type='Swish'),
                init_cfg=None):
        super().__init__(init_cfg)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1 = ConvModule(
            in_channels,
            in_channels,
            kernel_size=1,
            **cfg)
        self.conv2 = conv(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            **cfg)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out

class YOLOV5NeckCSPBlock(BaseModule):
    """Cross Stage Partial Block used in YOLOV5-v6.1,differs from
    backbone.

    Args:
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        num_blocks (int): Number of blocks. Default: 3
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish')
    """    

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 use_depthwise=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=None):
        super().__init__(init_cfg)
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        mid_channels = in_channels // 2
        self.long_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            **cfg)
        self.short_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            **cfg)
        self.final_conv = ConvModule(
            mid_channels * 2,
            out_channels,
            1,
            **cfg)
        self.blocks = nn.Sequential(*[
            YOLOV5BottleNeck(
                mid_channels,
                use_depthwise,
                **cfg) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x_short = self.short_conv(x)
        x_long = self.long_conv(x)
        x_long = self.blocks(x_long)

        x_final = torch.cat((x_long, x_short), dim=1)
        return self.final_conv(x_final)


@NECKS.register_module()
class YOLOV5Neck(BaseModule):
    """Path Aggregation Network used in YOLOV5.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 3
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish')
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 use_depthwise=False,
                 upsample_cfg=dict(scale_factor=2, mode='nearest'),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super(YOLOV5Neck, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    **cfg))
            self.top_down_blocks.append(
                YOLOV5NeckCSPBlock(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_blocks,
                    use_depthwise=use_depthwise,
                    **cfg))

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv(
                    in_channels[idx],
                    in_channels[idx],
                    3,
                    stride=2,
                    padding=1,
                    **cfg))
            self.bottom_up_blocks.append(
                YOLOV5NeckCSPBlock(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    num_blocks=num_blocks,
                    use_depthwise=use_depthwise,
                    **cfg))

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1))
            outs.append(out)

        return tuple(outs)
