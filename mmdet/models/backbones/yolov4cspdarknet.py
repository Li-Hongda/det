
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
from ..builder import BACKBONES

# 此处YOLOV4所使用的SPP结构中没有使用卷积层,相应卷积层的CBL结构(Conv-Bn-LeakyReLU)放在了backbone中
class SPPBottleneck(BaseModule):
    """Spatial pyramid pooling layer used in YOLOv4.

    Args:
        kernel_sizes (tuple[int]): Sequential of kernel sizes of pooling
            layers. Default: (5, 9, 13).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 kernel_sizes=(5, 9, 13),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
    def forward(self, x):
        x = torch.cat([x] + [pooling(x) for pooling in self.poolings], dim=1)
        return x


class YOLOV4ResUnit(BaseModule):
    """The basic residual unit used in each YOLOV4 CSPBlock.
    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        change_channel (bool): Whether to change the number of channels in 
        middle of ResUnit.
            Default: False        
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Default: False
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Mish').
    """

    def __init__(self,
                in_channels,
                out_channels,
                change_channel=False,
                use_depthwise=False,
                conv_cfg=None,
                norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                act_cfg=dict(type='Mish'),
                init_cfg=None):
        super().__init__(init_cfg)
        assert in_channels == out_channels
        if change_channel:
            mid_channels = out_channels / 2
        else:
            mid_channels = out_channels
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = ConvModule(
            in_channels,
            mid_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = conv(
            mid_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x

class YOLOV4CSPBlock(BaseModule):
    """Cross Stage Partial Block used in YOLOV4

    Args:
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        num_blocks (int): Number of blocks. Default: 1
        change_channel (bool): Whether to change the number of channels in block.
            Default: False 
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Mish')
    """    

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=1,
                 change_channel=False,
                 use_depthwise=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Mish'),
                 init_cfg=None):
        super().__init__(init_cfg)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        if change_channel:
            mid_channels = out_channels
        else:
            mid_channels = out_channels / 2 
        self.main_conv = conv(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            **cfg)
        self.short_conv = ConvModule(
            out_channels,
            mid_channels,
            1,
            **cfg)
        self.long_conv1 = ConvModule(
            out_channels,
            mid_channels,
            1,
            **cfg)
        self.long_conv2 = ConvModule(
            mid_channels,
            mid_channels,
            1,
            **cfg)
        self.final_conv = ConvModule(
            mid_channels * 2,
            out_channels,
            1,
            **cfg)
        self.blocks = nn.Sequential(*[
            YOLOV4ResUnit(
                mid_channels,
                mid_channels,
                change_channel,
                use_depthwise,
                **cfg) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = self.main_conv(x)
        x_short = self.short_conv(x)
        x_long = self.long_conv1(x)
        x_long = self.blocks(x_long)
        x_long = self.long_conv2(x_long)
        x_final = torch.cat((x_long, x_short), dim=1)
        return self.final_conv(x_final)


class CBLBlock(BaseModule):
    """CBL(Conv-BN-LeakyReLU) Block used in YOLOV4 backbone,differs from neck

    Args:
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU')
    """    

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='LeakyReLU'),
                 init_cfg=None):
        super().__init__(init_cfg)
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1 = ConvModule(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size=1,
            **cfg)
        self.conv2 = ConvModule(
            out_channels,
            1024,
            3,
            stride=2,
            padding=1,
            **cfg)
        self.conv3 = ConvModule(
            1024,
            out_channels,
            1,
            **cfg)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.conv3(x)
        return out

@BACKBONES.register_module()
class YOLOV4CSPDarknet(BaseModule):
    """CSP-Darknet backbone for YOLOV4.

    Args:
        arch (str): Architecture of CSP-Darknet, from {P5, P6}.
            Default: P5.
        out_indices (Sequence[int]): Output from which stages.
            Default: (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Default: -1.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Default: False.
        arch_ovewrite(list): Overwrite default arch settings. Default: None.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Mish').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    Example:
        >>> from mmdet.models import CSPDarknet
        >>> import torch
        >>> self = CSPDarknet(depth=53)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """
    # 从左到右依次为:
    # in_channels, out_channels, num_blocks, change_channel, use_spp
    # arch_settings = {
    #     'Standard': [[32, 64, 1, True, True], [64, 128, 2, False, False],
    #            [128, 256, 8, False, False], [256, 512, 8, False, False],
    #            [512, 1024, 4, False, False]]
    # }
    arch_setting =[[32, 64, 1, True, False], [64, 128, 2, False, False],
               [128, 256, 8, False, False], [256, 512, 8, False, False],
               [512, 1024, 4, False, True]]

    def __init__(self,
                #  arch='Standard',
                 out_indices=(2, 3, 4),
                 frozen_stages=-1,
                 use_depthwise=False,
                 arch_ovewrite=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Mish'),
                 norm_eval=False,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super().__init__(init_cfg)
        arch_setting = self.arch_setting
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        assert set(out_indices).issubset(
            i for i in range(len(arch_setting) + 1))
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError('frozen_stages must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.stem = ConvModule(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            **cfg)
        self.layers = ['stem']

        for i, (in_channels, out_channels, num_blocks, change_channel,
                use_spp) in enumerate(arch_setting):
            stage = []
            csp_block = YOLOV4CSPBlock(
                in_channels,
                out_channels,
                num_blocks=num_blocks,
                change_channel=change_channel,
                use_depthwise=use_depthwise,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(csp_block)
            if use_spp:
                conv1 = CBLBlock(
                    in_channels=out_channels,
                    out_channels=out_channels // 2)
                stage.append(conv1)
                spp = SPPBottleneck(
                    out_channels,
                    out_channels,
                    kernel_sizes=5,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                stage.append(spp)
                conv2 = CBLBlock(
                    in_channels=out_channels * 2,
                    out_channels=out_channels // 2)
                stage.append(conv2)
            self.add_module(f'stage{i + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{i + 1}')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(YOLOV4CSPDarknet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
