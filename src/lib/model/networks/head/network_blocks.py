import torch
import torch.nn as nn


class SiLU(nn.Module):
    """
    Export-friendly version of nn.SiLU() for platforms that do not support nn.SiLU natively.
    """
    @staticmethod
    def forward(x):
        """
        Apply the SiLU activation function to the input tensor.

        Parameters:
        - x: Input tensor.

        Returns:
        - Tensor: Output tensor after applying SiLU activation.
        """
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    """
    Retrieves an activation function by name.

    Parameters:
    - name: The name of the activation function ('silu', 'relu', 'lrelu', 'gelu', 'sigmoid').
    - inplace: Whether the operation should be performed inplace.

    Returns:
    - nn.Module: The corresponding activation function.
    """
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.2, inplace=inplace)
    elif name == "gelu":
        module = nn.GELU()
    elif name == "sigmoid":
        module = nn.Sigmoid()
    elif name == "":
        module = None
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


def get_normlization(name, num_features):
    """
    Create a normalization layer based on the specified type.

    Parameters:
    - name: Type of normalization layer ('bn' for BatchNorm, 'gn' for GroupNorm, or '' for None).
    - num_features: Number of features for which normalization will be applied.

    Returns:
    - nn.Module: The normalization layer or None.
    """
    if name == "bn":
        return nn.BatchNorm2d(num_features)
    elif name == "gn":
        # Avoid division by zero; at least one group.
        group = max(1, num_features // 32)
        return nn.GroupNorm(group, num_features)
    elif name == "":
        return None
    else:
        raise AttributeError(f"Unsupported normalization type: {name}")


class BaseConv(nn.Module):
    """A Conv2d -> norm -> leaky/relu/gelu/silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.ln = nn.LayerNorm([16, 500, 500])
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.ln(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class finalConv(nn.Module):
    """
    A module that encapsulates a Conv2d layer followed by an optional normalization
    and a non-linear activation layer.
    """

    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="leaky", norm="gn"):
        """
        Initializes the finalConv block with convolution, optional normalization, and activation.

        Parameters:
        - in_channels: Number of channels in the input image
        - out_channels: Number of channels produced by the convolution
        - ksize: Size of the convolving kernel
        - stride: Stride of the convolution
        - groups: Number of blocked connections from input channels to output channels
        - bias: If True, adds a learnable bias to the output
        - act: Type of activation to use ('leaky' for LeakyReLU, others can be added)
        - norm: Type of normalization to use ('bn', 'gn', or '')
        """
        super(finalConv, self).__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels,
                              ksize, stride, pad)
        self.norm = get_normlization(norm, out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        """
        Forward pass of the finalConv block.

        Parameters:
        - x: Input tensor to process through conv -> norm -> activation.

        Returns:
        - Tensor: Processed tensor.
        """
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x
