
from torch import nn
from src.lib.model.networks.utils import make_layers
import torch
from torch.utils.checkpoint import checkpoint


class ModuleWrapperIgnores2ndArg_cnn(nn.Module):
    """
    A module wrapper for a convolutional neural network (CNN) that ignores an additional dummy argument during the forward pass.
    This is typically used to enable gradient checkpointing while bypassing PyTorch limitations regarding non-Tensor inputs.
    """

    def __init__(self, module):
        """
        Initializes the ModuleWrapperIgnores2ndArg_cnn with the specified module.

        Parameters:
        - module: The module to wrap, which should be a CNN.
        """
        super().__init__()
        self.module = module

    def forward(self, x, dummy_arg=None):
        """
        Forwards the input through the module while ignoring the dummy argument.

        Parameters:
        - x: The input tensor to the CNN.
        - dummy_arg: A dummy argument that is not used but required for API compatibility. Must not be None.

        Returns:
        - Tensor: The output from the CNN module.
        """
        assert dummy_arg is not None, "dummy_arg is required but was None"
        x = self.module(x)
        return x


class ModuleWrapperIgnores2ndArg_lstm(nn.Module):
    """
    A module wrapper for an LSTM that ignores an additional dummy argument during the forward pass.
    This setup facilitates the use of gradient checkpointing with models that expect multiple inputs.
    """

    def __init__(self, module):
        """
        Initializes the ModuleWrapperIgnores2ndArg_lstm with the specified module.

        Parameters:
        - module: The module to wrap, which should be an LSTM.
        """
        super().__init__()
        self.module = module

    def forward(self, x, hx, cx, dummy_arg=None):
        """
        Forwards the inputs through the module while ignoring the dummy argument.

        Parameters:
        - x: The input tensor to the LSTM.
        - hx: The hidden state tensor for the LSTM.
        - cx: The cell state tensor for the LSTM.
        - dummy_arg: A dummy argument that is not used but required for API compatibility. Must not be None.

        Returns:
        - Tensor: The output from the LSTM module.
        """
        assert dummy_arg is not None, "dummy_arg is required but was None"
        x = self.module(x, (hx, cx))
        return x


class ModuleWrapperIgnores2ndArg_gru(nn.Module):
    """
    A module wrapper for a GRU that ignores an additional dummy argument during the forward pass.
    This allows for the use of gradient checkpointing with modules that expect multiple inputs.
    """

    def __init__(self, module):
        """
        Initializes the ModuleWrapperIgnores2ndArg_gru with the specified module.

        Parameters:
        - module: The module to wrap, which should be a GRU or similar recurrent unit.
        """
        super().__init__()
        self.module = module

    def forward(self, x, hx, dummy_arg=None):
        """
        Forwards the inputs through the module while ignoring the dummy argument.

        Parameters:
        - x: The input tensor to the GRU.
        - hx: The hidden state tensor for the GRU.
        - dummy_arg: A dummy argument that is not used but required for API compatibility. Must not be None.

        Returns:
        - Tensor: The output from the GRU module.
        """
        assert dummy_arg is not None, "dummy_arg is required but was None"
        x = self.module(x, hx)
        return x


class Encoder(nn.Module):
    """
    Encoder class for a network that integrates convolutional layers with RNN layers, optionally using LSTM or GRU cells.
    This encoder processes inputs through multiple stages each consisting of a convolution followed by an RNN layer.
    """

    def __init__(self, clstm, subnets, rnns, use_checkpoint):
        """
        Initialize the Encoder with specified layers and configuration.

        Parameters:
        - clstm (bool): Flag indicating whether to use LSTM (True) or GRU (False).
        - subnets (list): List of subnet configurations for convolutional layers.
        - rnns (list): List of RNN layers.
        - use_checkpoint (bool): Flag to enable gradient checkpointing for saving memory.
        """
        super().__init__()

        assert len(subnets) == len(
            rnns), "Each subnet must correspond to an RNN layer."

        self.blocks = len(subnets)
        self.use_checkpoint = use_checkpoint
        self.clstm = clstm

        # Convolutional layers
        self.stage1 = make_layers(subnets[0])
        self.stage2 = make_layers(subnets[1])
        self.stage3 = make_layers(subnets[2])

        # RNN layers
        self.rnn1 = rnns[0]
        self.rnn2 = rnns[1]
        self.rnn3 = rnns[2]

        # Dummy tensor for checkpointing
        self.dummy_tensor = torch.ones(
            1, dtype=torch.float32, requires_grad=True)

        # Wrapping layers to ignore the second argument when checkpointing
        self.stage1_wrapper = ModuleWrapperIgnores2ndArg_cnn(self.stage1)
        self.stage2_wrapper = ModuleWrapperIgnores2ndArg_cnn(self.stage2)
        self.stage3_wrapper = ModuleWrapperIgnores2ndArg_cnn(self.stage3)

        if clstm:
            self.rnn1_wrapper = ModuleWrapperIgnores2ndArg_lstm(self.rnn1)
            self.rnn2_wrapper = ModuleWrapperIgnores2ndArg_lstm(self.rnn2)
            self.rnn3_wrapper = ModuleWrapperIgnores2ndArg_lstm(self.rnn3)
        else:
            self.rnn1_wrapper = ModuleWrapperIgnores2ndArg_gru(self.rnn1)
            self.rnn2_wrapper = ModuleWrapperIgnores2ndArg_gru(self.rnn2)
            self.rnn3_wrapper = ModuleWrapperIgnores2ndArg_gru(self.rnn3)

    def forward_by_stage(self, i, inputs, hidden_state, subnet, rnn):
        """
        Process inputs through one stage of the encoder.

        Parameters:
        - i (int): Index of the current stage.
        - inputs (Tensor): Input tensor for the current stage.
        - hidden_state (Tensor): Hidden state tensor for the RNN.
        - subnet (nn.Module): Convolutional subnet for the current stage.
        - rnn (nn.Module): RNN module for the current stage.

        Returns:
        - outputs_stage (Tensor): Output tensor of the current stage.
        - state_stage (tuple): State tensor(s) output by the RNN.
        """
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))

        if self.use_checkpoint:
            if i == 1:
                inputs = checkpoint(self.stage1_wrapper, inputs,
                                    self.dummy_tensor)
            elif i == 2:
                inputs = checkpoint(self.stage2_wrapper, inputs,
                                    self.dummy_tensor)
            elif i == 3:
                inputs = checkpoint(self.stage3_wrapper, inputs,
                                    self.dummy_tensor)

        else:
            inputs = subnet(inputs)

        inputs = torch.reshape(
            inputs,
            (seq_number, batch_size, inputs.size(1), inputs.size(2),
             inputs.size(3)),
        )

        if self.clstm:
            if self.use_checkpoint:
                hidden_state = (None,
                                None) if hidden_state == None else hidden_state
                if i == 1:
                    outputs_state_stage = checkpoint(self.rnn1_wrapper, inputs,
                                                     hidden_state[0],
                                                     hidden_state[1],
                                                     self.dummy_tensor)
                elif i == 2:
                    outputs_state_stage = checkpoint(self.rnn2_wrapper, inputs,
                                                     hidden_state[0],
                                                     hidden_state[1],
                                                     self.dummy_tensor)
                elif i == 3:
                    outputs_state_stage = checkpoint(self.rnn3_wrapper, inputs,
                                                     hidden_state[0],
                                                     hidden_state[1],
                                                     self.dummy_tensor)
            else:
                outputs_state_stage = rnn(inputs, hidden_state)
        else:
            if self.use_checkpoint:
                if i == 1:
                    outputs_state_stage = checkpoint(self.rnn1_wrapper, inputs,
                                                     hidden_state,
                                                     self.dummy_tensor)
                elif i == 2:
                    outputs_state_stage = checkpoint(self.rnn2_wrapper, inputs,
                                                     hidden_state,
                                                     self.dummy_tensor)
                elif i == 3:
                    outputs_state_stage = checkpoint(self.rnn3_wrapper, inputs,
                                                     hidden_state,
                                                     self.dummy_tensor)
            else:
                outputs_state_stage = rnn(inputs, hidden_state)

        if self.clstm:  # LSTM
            hy, cy = outputs_state_stage[0], outputs_state_stage[1]
            outputs_stage = hy.unsqueeze(0)
            state_stage = (hy, cy)
        else:  # GRU
            hy = outputs_state_stage[0]
            outputs_stage = hy.unsqueeze(0)
            state_stage = (hy)

        return outputs_stage, state_stage

    def forward(self, inputs, state_stages):
        """
        Forward pass through all stages of the encoder.

        Parameters:
        - inputs (Tensor): Initial input to the encoder.
        - state_stages (list): Initial states for each stage of the encoder.

        Returns:
        - tuple: Tuple containing hidden states from all stages.
        """

        hidden_states = []

        for i in range(1, self.blocks + 1):
            inputs, state_stage = self.forward_by_stage(
                i, inputs, state_stages[i - 1],
                getattr(self, "stage" + str(i)), getattr(self, "rnn" + str(i)))

            hidden_states.append(state_stage)

        return tuple(hidden_states)
