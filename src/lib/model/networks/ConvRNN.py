import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class ModuleWrapperIgnores2ndArg(nn.Module):
    """
    A wrapper for any nn.Module that ignores a second argument in the forward pass.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, dummy_arg=None):
        """
        Forward pass which ignores the dummy argument.

        Parameters:
        - x: Input tensor.
        - dummy_arg: A dummy argument which is expected but not used in computation.
        """
        assert dummy_arg is not None
        x = self.module(x)
        return x


class CGRU_cell(nn.Module):
    """
    A ConvGRU cell implementation that can be used in an encoder or decoder configuration.
    """

    def __init__(self, use_checkpoint, shape, input_channels, filter_size,
                 num_features, module):
        """
        Initializes the CGRU cell with specific configurations for the convolutional operations and gating mechanisms.

        Parameters:
        - use_checkpoint: Boolean indicating whether to use gradient checkpointing to save memory.
        - shape: Dimensions (height, width) of the input feature maps.
        - input_channels: Number of channels in the input feature map.
        - filter_size: Size of the convolution kernel.
        - num_features: Number of output features for each convolution operation.
        - module: Specifies the configuration as 'encoder' or 'decoder' to adjust internal connections and channel dimensions.
        """
        super(CGRU_cell, self).__init__()
        self.shape = shape
        self.input_channels = int(input_channels)
        self.filter_size = filter_size
        self.num_features = int(num_features)
        self.padding = (filter_size - 1) // 2

        self.module = module

        if self.module == "encoder":
            conv_input_channels = self.input_channels + self.num_features
            conv_output_channels = 2 * self.num_features
        elif self.module == "decoder":
            conv_input_channels = self.input_channels + 2*self.num_features
            conv_output_channels = 2 * self.num_features

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                conv_input_channels,
                conv_output_channels,
                self.filter_size,
                1,
                self.padding,
            ),
            nn.GroupNorm(conv_output_channels // 32, conv_output_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                conv_input_channels,
                self.num_features,
                self.filter_size,
                1,
                self.padding,
            ),
            nn.GroupNorm(self.num_features // 32, self.num_features),
        )

        self.use_checkpoint = use_checkpoint
        self.dummy_tensor = torch.ones(1,
                                       dtype=torch.float32,
                                       requires_grad=True)
        self.conv1_module_wrapper = ModuleWrapperIgnores2ndArg(self.conv1)
        self.conv2_module_wrapper = ModuleWrapperIgnores2ndArg(self.conv2)

    def forward(self, inputs=None, hidden_state=None, seq_len=1):
        """
        Forward pass for sequential data processing through ConvGRU cell.

        Parameters:
        - inputs: Input tensor sequence.
        - hidden_state: Initial hidden state.
        - seq_len: Sequence length for processing.
        """
        if hidden_state is None:
            htprev = torch.zeros(inputs.size(1), self.num_features,
                                 self.shape[0], self.shape[1]).cuda()
        else:
            htprev = hidden_state
        output_inner = []

        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(htprev.size(0), self.input_channels,
                                self.shape[0], self.shape[1]).cuda()
            else:
                x = inputs[index, ...]

            combined_1 = torch.cat((x, htprev), 1)  # X_t + H_t-1
            if self.use_checkpoint:
                gates = checkpoint(self.conv1_module_wrapper, combined_1,
                                   self.dummy_tensor)
            else:
                gates = self.conv1(combined_1)  # W * (X_t + H_t-1)

            zgate, rgate = torch.split(gates, self.num_features, dim=1)
            z = torch.sigmoid(zgate)
            r = torch.sigmoid(rgate)

            # h' = tanh(W*(x+r*H_t-1))
            if self.module == "encoder":
                combined_2 = torch.cat((x, r * htprev), 1)
            elif self.module == "decoder":
                etprev, dtprev = torch.split(htprev, self.num_features, dim=1)
                combined_2 = torch.cat((x, etprev, r * dtprev), 1)

            if self.use_checkpoint:
                ht = checkpoint(self.conv2_module_wrapper, combined_2,
                                self.dummy_tensor)
            else:
                ht = self.conv2(combined_2)  # num_features
            ht = torch.tanh(ht)

            if self.module == "encoder":
                htnext = (1 - z) * htprev + z * ht
            elif self.module == "decoder":
                # decoder_htprev = htprev[:, htprev.shape[1]//2:]
                htnext = (1 - z) * dtprev + z * ht

            output_inner.append(htnext)
            htprev = htnext

        return torch.stack(output_inner)
