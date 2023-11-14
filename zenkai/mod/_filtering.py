# 1st party
import math
import typing

# 3rd party
import torch
from torch import nn

# local
from ._reversible import Reversible
from functools import singledispatch


@singledispatch
def to_2dtuple(value: int) -> typing.Tuple[int, int]:
    return (value, value)


@to_2dtuple.register
def _(value: tuple) -> typing.Tuple[int, int]:
    return value


def calc_stride2d(
    x: torch.Tensor, stride
) -> typing.Tuple[int, int, int, int, int, int]:
    """Calculate the stride when collapsing an image to a twod tensor

    Args:
        x (torch.Tensor): the image
        stride (Tuple[int, int]): the stride to collapse with

    Returns:
        typing.Tuple: The resulting shape to collapse with
    """
    return (x.stride(0), x.stride(1), x.stride(2), stride[0], x.stride(2), stride[1])


def calc_size2d(
    x: torch.Tensor, stride, kernel_size
) -> typing.Tuple[int, int, int, int, int, int]:
    """Calculate the size 2d

    Args:
        x (torch.Tensor): image
        stride (int): The amount to stride by
        kernel_size (Tuple[int, int]): the kernel used

    Returns:
        Tuple: The
    """
    return (
        x.size(0),
        x.size(1),
        (x.size(2) - (kernel_size[0] - 1)) // stride[0],
        (x.size(3) - (kernel_size[1] - 1)) // stride[1],
        kernel_size[0],
        kernel_size[1],
    )


class Stride2D(nn.Module):
    def __init__(
        self,
        in_features: int,
        kernel_size: typing.Tuple[int],
        stride: typing.Tuple[int] = None,
        padding: typing.Tuple[int] = None,
        reshape_output: bool = True,
    ):

        super().__init__()
        stride = stride or 1
        padding = padding or 0
        self._in_features = in_features
        self._kernel_size = to_2dtuple(kernel_size)
        self._stride = to_2dtuple(stride)
        if padding is not None:
            padding = to_2dtuple(padding)
            self._padding = tuple(
                map(int, (padding[0], padding[0], padding[1], padding[1]))
            )
            # self._padding = tuple(map(int, (
            #     math.floor(padding[0] / 2),
            #     math.ceil(padding[0] / 2),
            #     math.floor(padding[1] / 2),
            #     math.ceil(padding[1] / 2)
            # )))
        else:
            self._padding = None
        self._reshape_output = reshape_output

    @property
    def out_features(self) -> int:
        return self._kernel_size[0] * self._kernel_size[1] * self._in_features

    @property
    def out_shape(self) -> torch.Size:
        return torch.Size([self._in_features, *self._kernel_size])

    def calc_size(self, in_width: int, in_height: int) -> typing.Tuple[int, int]:
        """Calc the width and height after processeing

        Args:
            in_width (int): The width of the image input
            in_height (int): The height of the image output

        Returns:
            typing.Tuple[int, int]: The width and height after processing. These will
            become a part of the batch dimension
        """
        return (
            (in_width - self._kernel_size[0] + 1 + self._padding[0] + self._padding[1])
            // self._stride[0],
            (in_height - self._kernel_size[1] + 1 + self._padding[2] + self._padding[3])
            // self._stride[1],
        )

    def forward(self, x: torch.Tensor):
        if self._padding is not None:
            x = torch.nn.functional.pad(x, self._padding, "constant", 0)

        strided = torch.as_strided(
            x,
            calc_size2d(x, self._stride, self._kernel_size),
            calc_stride2d(x, self._stride),
        )
        strided = strided.permute(0, 2, 3, 1, 4, 5)
        if not self._reshape_output:
            return strided
        return strided.reshape(-1, math.prod(strided.shape[3:]))


class UndoStride2D(Reversible):
    """ """

    def __init__(self, n_channels: int, size: typing.Tuple[int]):
        super().__init__()
        self._n_channels = n_channels
        self._size = to_2dtuple(size)

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def out_shape(self) -> typing.Tuple[int, int, int]:
        return self._n_channels, self._size[0], self._size[1]

    def forward(self, x: torch.Tensor):

        return x.reshape(-1, self._size[0], self._size[1], self._n_channels).permute(
            0, 3, 1, 2
        )

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        return y.permute(0, 2, 3, 1).reshape(-1, y.size(1))


class TargetStride(nn.Module):
    """Convert the target to match a strided output"""

    def __init__(self, out_channels: int, width: int, height: int):
        super().__init__()

        self.out_channels = out_channels
        self.width = width
        self.height = height

    def forward(self, x: torch.Tensor):

        return (
            x.view(x.size(0), self.out_channels, self.width, self.height)
            .permute(0, 2, 3, 1)
            .reshape(-1, self.out_channels)
        )
