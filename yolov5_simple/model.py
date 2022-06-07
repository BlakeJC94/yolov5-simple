from math import ceil
from typing import Union, Optional, List, Tuple

import torch
from torch import nn

from seeralgo.models import SeerModule


def autopad(kernel, padding=None):
    if padding is None:
        padding = kernel // 2 if isinstance(kernel, int) else (x // 2 for x in kernel)
    return padding


def w_exp(chs, mult):
    return ceil(chs * mult / 8) * 8 if mult != 1 else chs


def d_exp(depth, mult):
    return max(round(depth * mult), 1) if depth > 1 else depth


class ConvBlock(nn.Module):
    """Standard convolutional block with batch normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        w_mult: Union[float, Tuple[float]] = 1.0,
        kernel: Union[int, Tuple[int]] = 1,
        stride: Union[int, Tuple[int]] = 1,
        padding: Optional[Union[str, int, Tuple[int]]] = None,
        groups: int = 1,
    ):
        """Constructor for ConvBlock.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            w_mult: Width multiplier. If set as a float, both `in_channels` and `out_channels` will
                be scaled. Scaling can be individually controlled by setting as a 2-element Tuple.
            kernel: kernel size for convolution layer.
            stride: stride for convolution layer.
            padding: padding for convolution layer. Determined from kernel size if None.
            groups: groups for convolution layer.
        """
        super().__init__()

        if isinstance(w_mult, int):
            w_mult = float(w_mult)
        if isinstance(w_mult, float):
            w_mult = (w_mult, w_mult)

        if isinstance(kernel, float):
            kernel = int(kernel)
        if isinstance(kernel, int):
            kernel = (kernel, kernel)

        in_channels = w_exp(in_channels, w_mult[0])
        out_channels = w_exp(out_channels, w_mult[1])

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel,
            stride=stride,
            padding=autopad(kernel[0], padding),
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """A forward pass through the block."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Bottleneck(nn.Module):
    """Standard bottleneck"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        groups: int = 1,
        expansion: float = 1.0,
    ):
        """Constructor for Bottleneck.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            shortcut: Whether to add output of the block to the input (if the channels are equal).
            groups: Groups for second convolution layer.
            expansion: Factor for channel expansion between convolution layers.
        """
        super().__init__()
        _channels = int(out_channels * expansion)
        self.conv_0 = ConvBlock(in_channels, _channels, kernel=1, stride=1)
        self.conv_1 = ConvBlock(_channels, out_channels, kernel=3, stride=1, groups=groups)
        self.add = (shortcut and in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """A forward pass through the block."""
        x0 = self.conv_0(x)
        x0 = self.conv_1(x0)
        return x + x0 if self.add else x0


class C3Block(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(
        self,
        in_channels: Union[int, Tuple[int]],
        out_channels: int,
        depth: int,
        d_mult: float = 1.0,
        w_mult: Union[float, Tuple[float]] = 1.0,
        shortcut: bool = True,
        expansion: float = 0.5,
    ):
        """Constructor for C3Block.

        Args:
            in_channels: Number of input channels. If set as a 2-element tuple, Number of input
                channels is set as first element (expanded by w_mult) added to the second element.
            out_channels: Number of output channels.
            depth: Number of bottlenecks to use in block.
            d_mult: Multiplier for number of bottlenecks in block.
            w_mult: Width multiplier. See docsting for `ConvBlock`.
                be scaled. Scaling can be individually controlled by setting as a 2-element Tuple.
            shortcut: See docstring for `Bottleneck`.
            expansion: Factor for channel expansion between convolution layers.
        """
        super().__init__()

        if isinstance(w_mult, int):
            w_mult = float(w_mult)
        if isinstance(w_mult, float):
            w_mult = (w_mult, w_mult)

        if isinstance(in_channels, int):
            in_channels = (in_channels, 0)

        in_channels = w_exp(in_channels[0], w_mult[0]) + in_channels[1]
        out_channels = w_exp(out_channels, w_mult[1])

        self.in_channels = in_channels
        self.out_channels = out_channels

        _channels = int(out_channels * expansion)
        self.conv_0 = ConvBlock(in_channels, _channels, kernel=1, stride=1)
        self.conv_1 = ConvBlock(in_channels, _channels, kernel=1, stride=1)
        self.conv_2 = ConvBlock(2 * _channels, out_channels, kernel=1)

        bottlenecks = []
        for _ in range(d_exp(depth, d_mult)):
            layer = Bottleneck(_channels, _channels, shortcut, groups=1, expansion=1.0)
            bottlenecks.append(layer)

        self.bottlenecks = nn.Sequential(*bottlenecks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """A forward pass through the block."""
        x0 = self.conv_0(x)
        x0 = self.bottlenecks(x0)
        x1 = self.conv_1(x)
        x = torch.cat((x0, x1), dim=1)
        x = self.conv_2(x)
        return x


class SPPFBlock(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        w_mult: Union[float, Tuple[float]] = 1.0,
        kernel: Union[int, List[int]] = 5,
    ):
        """Constructor for SPPFBlock.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__()

        if isinstance(w_mult, int):
            w_mult = float(w_mult)
        if isinstance(w_mult, float):
            w_mult = (w_mult, w_mult)

        in_channels = w_exp(in_channels, w_mult[0])
        out_channels = w_exp(out_channels, w_mult[1])

        _channels = in_channels // 2
        self.conv_0 = ConvBlock(in_channels, _channels, 1.0, 1, 1)
        self.conv_1 = ConvBlock(_channels * 4, out_channels, 1.0, 1, 1)
        self.mp = nn.MaxPool2d(kernel_size=kernel, stride=1, padding=kernel // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """A forward pass through the block."""
        x0 = self.conv_0(x)
        x1 = self.mp(x0)
        x2 = self.mp(x1)
        x3 = self.mp(x2)
        x = torch.cat((x0, x1, x2, x3), 1)
        x = self.conv_1(x)
        return x


class DetectBlock(nn.Module):
    """Detection layer for YOLOv5."""

    def __init__(
        self,
        anchors: Tuple[Tuple[int]],
        in_channels: Tuple[int],
        n_classes: int = 80,
    ):
        """Constructor for DetectBlock.

        Args:
            anchors: Anchors for detection layer.
            in_channels: List of number of input channels for convolutions (for each input).
            n_classes: Number of output classes.
        """
        super().__init__()

        self.n_classes = n_classes
        self.n_outputs_per_anchor = n_classes + 5
        self.n_layers = len(anchors)
        self.n_anchors = len(anchors[0]) // 2

        self.grid = [torch.zeros(1)] * self.n_layers
        self.anchor_grid = [torch.zeros(1)] * self.n_layers  # shape(n_layers, n_anchors, 2)
        anchors = torch.tensor(anchors).float().view(self.n_layers, -1, 2)
        self.register_buffer('anchors', anchors)

        self.layers = nn.ModuleList()
        out_chs = self.n_outputs_per_anchor * self.n_anchors
        for in_chs in in_channels:
            self.layers.append(nn.Conv2d(in_chs, out_chs, 1))

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """A forward pass through the block.

        Args:
            x: List of tensors to be passed through the block. Length must be equal to length of
                `in_channels` in constructor.
        """
        out = []
        for i in range(self.n_layers):
            x[i] = self.layers[i](x[i])

            bs, _, ny, nx = x[i].shape  # x(bs, 255, 20, 20) -> x(bs, 3, 20, 20, 85)
            x[i] = x[i].view(bs, self.n_anchors, self.n_outputs_per_anchor, ny, nx)
            x[i] = x[i].permute(0, 1, 3, 4, 2)
            x[i] = x[i].contiguous()

            # TODO Add self.strides from source code
            # if not self.training:
            #     if self.grid[i].shape[2:4] != x[i].shape[2:4]:
            #         self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

            #     y = x[i].sigmoid()
            #     xy, wh, conf = y.split((2, 2, self.n_classes + 1), dim=4)
            #     xy = (xy * 2 + self.grid[i]) * self.stride[i]
            #     wh = (wh * 2)**2 * self.anchor_grid[i]

            #     y = torch.cat((xy, wh, conf), dim=4)
            #     y = y.view(bs, -1, self.n_outputs_per_anchor)

            #     out.append(y)
        # return x if self.training else (torch.cat(out, dim=1), x)

        return x

    def _make_grid(self, nx: int = 20, ny: int = 20, i: int = 0) -> Tuple[torch.Tensor]:
        device, dtype = self.anchors[i].device, self.anchors[i].dtype
        grid_shape = (1, self.n_anchors, ny, nx, 2)

        y = torch.arange(ny, device=device, dtype=dtype)
        x = torch.arange(nx, device=device, dtype=dtype)

        # Use ij indexing for torch>=1.10.0
        yv, xv = torch.meshgrid(y, x, indexing='ij')

        # add grid offset, i.e. y = 2.0 * x - 0.5
        grid = torch.stack((xv, yv), dim=2).expand(grid_shape) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i])
        anchor_grid = anchor_grid.view((1, self.n_anchors, 1, 1, 2))
        anchor_grid = anchor_grid.expand(grid_shape)

        return grid, anchor_grid


class _YOLOv5(nn.Module):
    """Base YOLOv5 module."""

    def __init__(
        self,
        _d_mult: float,
        _w_mult: float,
        n_channels: int,
        n_classes: int,
        **kwargs,
    ):
        """Constructor for _YOLOv5.

        Args:
            _d_mult: Depth factor.
            _w_mult: Width factor.
            n_channels: Number of channels in input data.
            n_classes: Number of output classes.
            **kwargs: Additional keyword arguments to be passed to nn.Module.
        """
        super().__init__(**kwargs)
        anchors: List[List[int]] = [
            [10, 13, 16, 30, 33, 23],  # P3/8
            [30, 61, 62, 45, 59, 119],  # P4/16
            [116, 90, 156, 198, 373, 326],  # P5/32
        ]
        d_mult = _d_mult
        w_mult = _w_mult

        self.conv_0 = ConvBlock(n_channels, 64, (1.0, w_mult), kernel=6, stride=2, padding=2)
        self.conv_1 = ConvBlock(64, 128, w_mult, kernel=3, stride=2)
        self.c3_0 = C3Block(128, 128, 3, d_mult, w_mult, shortcut=True)
        self.conv_2 = ConvBlock(128, 256, w_mult, kernel=3, stride=2)
        self.c3_1 = C3Block(256, 256, 6, d_mult, w_mult, shortcut=True)
        self.conv_3 = ConvBlock(256, 512, w_mult, kernel=3, stride=2)
        self.c3_2 = C3Block(512, 512, 9, d_mult, w_mult, shortcut=True)
        self.conv_4 = ConvBlock(512, 1024, w_mult, kernel=3, stride=2)
        self.c3_3 = C3Block(1024, 1024, 3, d_mult, w_mult, shortcut=True)
        self.sppf_0 = SPPFBlock(1024, 1024, w_mult, kernel=5)

        self.conv_5 = ConvBlock(1024, 512, w_mult, kernel=1, stride=1)
        self.up_0 = nn.Upsample(None, scale_factor=2, mode='nearest')
        c3_4_in_chs = (512, self.c3_2.out_channels)
        self.c3_4 = C3Block(c3_4_in_chs, 512, 3, d_mult, w_mult, shortcut=False)

        self.conv_6 = ConvBlock(512, 256, w_mult, kernel=1, stride=1)
        self.up_1 = nn.Upsample(None, scale_factor=2, mode='nearest')
        c3_5_in_chs = (256, self.c3_1.out_channels)
        self.c3_5 = C3Block(c3_5_in_chs, 256, 3, d_mult, w_mult, shortcut=False)

        self.conv_7 = ConvBlock(256, 256, w_mult, kernel=3, stride=2)
        c3_6_in_chs = (256, self.conv_6.out_channels)
        self.c3_6 = C3Block(c3_6_in_chs, 512, 3, d_mult, w_mult, shortcut=False)

        self.conv_8 = ConvBlock(512, 512, w_mult, kernel=3, stride=2)
        c3_7_in_chs = (512, self.conv_5.out_channels)
        self.c3_7 = C3Block(c3_7_in_chs, 1024, 3, d_mult, w_mult, shortcut=False)

        detect_in_chs = (self.c3_5.out_channels, self.c3_6.out_channels, self.c3_7.out_channels)
        self.detect = DetectBlock(anchors, detect_in_chs, n_classes)

    def forward(self, x):
        """A forward pass through the network."""
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.c3_0(x)
        x = self.conv_2(x)
        x_c3_1 = self.c3_1(x)
        x = self.conv_3(x_c3_1)
        x_c3_2 = self.c3_2(x)
        x = self.conv_4(x_c3_2)
        x = self.c3_3(x)
        x = self.sppf_0(x)

        x_conv_5 = self.conv_5(x)
        x = self.up_0(x_conv_5)
        x = torch.cat((x, x_c3_2), dim=1)
        x = self.c3_4(x)

        x_conv_6 = self.conv_6(x)
        x = self.up_1(x_conv_6)
        x = torch.cat((x, x_c3_1), dim=1)
        x_c3_5 = self.c3_5(x)

        x = self.conv_7(x_c3_5)
        x = torch.cat((x, x_conv_6), dim=1)
        x_c3_6 = self.c3_6(x)

        x = self.conv_8(x_c3_6)
        x = torch.cat((x, x_conv_5), dim=1)
        x = self.c3_7(x)

        x = self.detect([x_c3_5, x_c3_6, x])
        return x


class YOLOv5n(_YOLOv5):

    def __init__(self, n_channels=3, n_classes=80, **kwargs):
        """Constructor for the YOLOv5n class."""
        super().__init__(0.33, 0.25, n_channels, n_classes, **kwargs)

    __init__.__doc__ += _YOLOv5.__init__.__doc__


class YOLOv5s(_YOLOv5):

    def __init__(self, n_channels=3, n_classes=80, **kwargs):
        """Constructor for the YOLOv5s class."""
        super().__init__(0.33, 0.50, n_channels, n_classes, **kwargs)

    __init__.__doc__ += _YOLOv5.__init__.__doc__


class YOLOv5m(_YOLOv5):

    def __init__(self, n_channels=3, n_classes=80, **kwargs):
        """Constructor for the YOLOv5m class."""
        super().__init__(0.67, 0.75, n_channels, n_classes, **kwargs)

    __init__.__doc__ += _YOLOv5.__init__.__doc__


class YOLOv5l(_YOLOv5):

    def __init__(self, n_channels=3, n_classes=80, **kwargs):
        """Constructor for the YOLOv5l class."""
        super().__init__(1.0, 1.0, n_channels, n_classes, **kwargs)

    __init__.__doc__ += _YOLOv5.__init__.__doc__


class YOLOv5x(_YOLOv5):

    def __init__(self, n_channels=3, n_classes=80, **kwargs):
        """Constructor for the YOLOv5x class."""
        super().__init__(1.33, 1.25, n_channels, n_classes, **kwargs)

    __init__.__doc__ += _YOLOv5.__init__.__doc__

