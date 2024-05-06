from torch import nn
from collections import OrderedDict
import torch


def get_normlization(name, num_features):
    """
    Creates a normalization layer based on the specified type and number of features.

    Parameters:
    - name: Type of normalization ('bn' for BatchNorm, 'gn' for GroupNorm, or an empty string for None).
    - num_features: Number of features in the layer for which normalization is to be applied.

    Returns:
    - nn.Module: The normalization layer, or None if no normalization is specified.
    """
    if name == "bn":
        module = nn.BatchNorm2d(num_features)
    elif name == "gn":
        group = 1 if num_features//32 == 0 else num_features//32
        module = nn.GroupNorm(group, num_features)
    elif name == "":
        module = None
    else:
        raise AttributeError("Unsupported normlization type: {}".format(name))
    return module


class SiLU(nn.Module):
    """
    Export-friendly version of nn.SiLU() for platforms that do not support nn.SiLU natively.
    """
    @staticmethod
    def forward(x):
        """
        Apply the SiLU activation function.

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
        module = nn.Sigmoid(inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


def make_layers(block, norm_name="", act="lrelu"):
    """
    Constructs a sequence of layers from a dictionary specification.

    Parameters:
    - block: A dictionary where keys are layer types and values are configurations for these layers.
    - norm_name: The type of normalization to apply.
    - act: The type of activation function to use.

    Returns:
    - nn.Sequential: An ordered dictionary of layers constructed according to the specifications.
    """
    layers = []

    for layer_name, v in block.items():
        v = [int(x) for x in v]
        if 'maxpool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'avgpool' in layer_name:
            layer = nn.AvgPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0],
                                                 out_channels=v[1],
                                                 kernel_size=v[2],
                                                 stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            # norm
            norm = get_normlization(norm_name, v[1])
            if norm is not None:
                layers.append(('norm_' + norm_name + "_" + layer_name, norm))
            # act
            layers.append((act + '_' + layer_name, get_activation(act)))

        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0],
                               out_channels=v[1],
                               kernel_size=v[2],
                               stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            # norm
            norm = get_normlization(norm_name, v[1])
            if norm is not None:
                layers.append(('norm_' + norm_name + "_" + layer_name, norm))
            # act
            layers.append((act + '_' + layer_name, get_activation(act)))

        else:
            raise NotImplementedError
    return nn.Sequential(OrderedDict(layers))
