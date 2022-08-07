import math
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
from ..builder import BACKBONES


# YOLOV5改进版SPP，将原来并行的kernel_size为[5,9,13]的池化层改为串行通过kernel_size为5的池化层
class SPPFBottleneck(BaseModule):
    """Imprioved Spatial Pyramid Pooling - Fast layer used in YOLOv5.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_sizes (tuple[int]): Sequential of kernel sizes of pooling
            layers. Default: (5, 9, 13).
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=None):
        super().__init__(init_cfg)
        mid_channels = in_channels // 2
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1 = ConvModule(
            in_channels,
            mid_channels,
            1,
            stride=1,
            **cfg)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.conv2 = ConvModule(
            mid_channels * 4,
            out_channels,
            1,
            **cfg)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.maxpool(x)
        x2 = self.maxpool(x1)
        out = torch.cat((x, x1, x2, self.maxpool(x2)), dim=1)
        out = self.conv2(out)
        return out

# U版YOLOV3所使用的SPP
class SPPBottleneck(BaseModule):
    """Spatial pyramid pooling layer used in YOLOv3-SPP.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_sizes (tuple[int]): Sequential of kernel sizes of pooling
            layers. Default: (5, 9, 13).
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(5, 9, 13),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=None):
        super().__init__(init_cfg)
        mid_channels = in_channels // 2
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1 = ConvModule(
            in_channels,
            mid_channels,
            1,
            stride=1,
            **cfg)
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = mid_channels * (len(kernel_sizes) + 1)
        self.conv2 = ConvModule(
            conv2_channels,
            out_channels,
            1,
            **cfg)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [pooling(x) for pooling in self.poolings], dim=1)
        x = self.conv2(x)
        return x


# 原始Focus模块, 在YOLOV5官方v6.0以后的版本被替换为6x6的卷积层以提高运行速度。
class Focus(BaseModule):
    """Focus width and height information into channel space.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_size (int): The kernel size of the convolution. Default: 1
        stride (int): The stride of the convolution. Default: 1
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish')):
        super().__init__()
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv = ConvModule(
            in_channels * 4,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            **cfg)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(
            torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), dim = 1))

class YOLOV5ResUnit(BaseModule):
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
        return out + x

class YOLOV5CSPBlock(BaseModule):
    """Cross Stage Partial Block used in YOLOV5-v6.1.

    Args:
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        expansion (float): Ratio to adjust the number of channels of the
            hidden layer. Default: 0.5
        num_blocks (int): Number of blocks. Default: 1
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
                 expansion=0.5,
                 num_blocks=1,
                 use_depthwise=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=None):
        super().__init__(init_cfg)
        mid_channels = int(out_channels * expansion)
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.main_conv = conv(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            **cfg)
        self.long_conv = ConvModule(
            out_channels,
            mid_channels,
            1,
            **cfg)
        self.short_conv = ConvModule(
            out_channels,
            mid_channels,
            1,
            **cfg)
        self.final_conv = ConvModule(
            mid_channels * 2,
            out_channels,
            1,
            **cfg)

        self.blocks = nn.Sequential(*[
            YOLOV5ResUnit(
                mid_channels,
                use_depthwise,
                **cfg) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = self.main_conv(x)
        x_short = self.short_conv(x)
        x_long = self.long_conv(x)
        x_long = self.blocks(x_long)

        x_final = torch.cat((x_long, x_short), dim=1)
        return self.final_conv(x_final)


@BACKBONES.register_module()
class YOLOV5CSPDarknet(BaseModule):
    """CSP-Darknet backbone for YOLOV5-v6.1.

    Args:
        arch (str): Architecture of CSP-Darknet, from {P5, P6}.
            Default: P5.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Default: 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
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
            Default: dict(type='Swish').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    Example:
        Input:  (1, 3, 640, 640)
        Output:
                (1, 256, 80, 80)
                (1, 512, 40, 40)
                (1, 1024, 20, 20)
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, use_spp/sppf
    arch_settings = {
        'P5': [[64, 128, 3, False], [128, 256, 6, False],
               [256, 512, 9, False], [512, 1024, 3, True]],
        'P6': [[64, 128, 3, False], [128, 256, 9, False],
               [256, 512, 9, False], [512, 768, 3, False],
               [768, 1024, 3, True]]
    }

    def __init__(self,
                 arch='P5',
                 deepen_factor=1.0,
                 widen_factor=1.0,
                 out_indices=(2, 3, 4),
                 frozen_stages=-1,
                 use_depthwise=False,
                 use_sppf=True,
                 arch_ovewrite=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 norm_eval=False,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super().__init__(init_cfg)
        arch_setting = self.arch_settings[arch]
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
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # YOLOV5 tag-v6.0以后的版本所使用的FOCUS为6x6卷积，详情见：
        #     https://github.com/ultralytics/yolov5/issues/4825
        self.stem = ConvModule(
            3,
            int(arch_setting[0][0] * widen_factor),
            6,
            stride=2,
            padding=2,
            **cfg)

        # 原始FOCUS模块
        # self.focus = Focus(
        #     3,
        #     int(arch_setting[0][0] * widen_factor),
        #     kernel_size=3,
        #     conv_cfg=conv_cfg,
        #     norm_cfg=norm_cfg,
        #     act_cfg=act_cfg)
        self.layers = ['stem']

        for i, (in_channels, out_channels, num_blocks, 
                use_spp) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage = []
            csp_block = YOLOV5CSPBlock(
                in_channels,
                out_channels,
                num_blocks=num_blocks,
                use_depthwise=use_depthwise,
                **cfg)
            stage.append(csp_block)
            if use_spp:
                if use_sppf:
                    spp = SPPFBottleneck(
                        out_channels,
                        out_channels,
                        kernel_size=5,
                        **cfg)
                else:
                    spp = SPPBottleneck(
                        out_channels,
                        out_channels,
                        kernel_sizes=(5, 9, 13),
                        **cfg)
                stage.append(spp)
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
        super(YOLOV5CSPDarknet, self).train(mode)
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
