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


class ModuleWrapperIgnores2ndArg_gru(nn.Module):
    """
    A module wrapper for a gated recurrent unit (GRU) that ignores an additional dummy argument during the forward pass.
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

    def forward(self, hx, x, dummy_arg=None):
        """
        Forwards the inputs through the module while ignoring the dummy argument.

        Parameters:
        - hx: The hidden state tensor for the GRU.
        - x: The input tensor to the GRU.
        - dummy_arg: A dummy argument that is not used but required for API compatibility. Must not be None.

        Returns:
        - Tensor: The output from the GRU module.
        """
        assert dummy_arg is not None, "dummy_arg is required but was None"
        x = self.module(x, hx)
        return x


class Decoder(nn.Module):
    """
    Decoder module that combines convolutional layers and GRU layers
    to process and decode the encoder states into outputs.
    """

    def __init__(self, clstm, subnets, rnns, use_checkpoint):
        """
        Initialize the Decoder with convolutional layers and RNNs.

        Parameters:
        - clstm : Flag to denote if clstm is used.
        - subnets: List of specifications for convolutional layers.
        - rnns: List of RNNs for processing.
        - use_checkpoint: Flag to use gradient checkpointing to save memory.
        """
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        self.stage3 = make_layers(subnets[0])
        self.stage2 = make_layers(subnets[1])
        self.stage1 = make_layers(subnets[2])
        self.rnn3 = rnns[0]
        self.rnn2 = rnns[1]
        self.rnn1 = rnns[2]

        self.clstm = clstm  # flag
        self.use_checkpoint = use_checkpoint

        # Dummy tensor for checkpointing
        self.dummy_tensor = torch.ones(
            1, dtype=torch.float32, requires_grad=True)

        # Wrappers to ignore the second argument
        self.stage1_wrapper = ModuleWrapperIgnores2ndArg_cnn(self.stage1)
        self.stage2_wrapper = ModuleWrapperIgnores2ndArg_cnn(self.stage2)
        self.stage3_wrapper = ModuleWrapperIgnores2ndArg_cnn(self.stage3)
        self.rnn1_wrapper = ModuleWrapperIgnores2ndArg_gru(self.rnn1)
        self.rnn2_wrapper = ModuleWrapperIgnores2ndArg_gru(self.rnn2)
        self.rnn3_wrapper = ModuleWrapperIgnores2ndArg_gru(self.rnn3)

    def forward_by_stage(self, i, inputs, encoder_states, decoder_states=None):
        """
        Process input data through one stage of the decoder.

        Parameters:
        - i: Stage index (1, 2, 3).
        - inputs: Input tensor.
        - encoder_states: Encoder state tensor.
        - decoder_states: Decoder state tensor, default is None.

        Returns:
        - Tensor: Output tensor after processing.
        - Tensor: Updated state tensor.
        """
        if decoder_states is None:
            decoder_states = torch.zeros_like(encoder_states).cuda()
        state = torch.cat((encoder_states, decoder_states),
                          dim=1)

        if self.use_checkpoint:
            if i == 1:
                outputs_state_stage = checkpoint(self.rnn1_wrapper,
                                                 state, inputs,  self.dummy_tensor)
            elif i == 2:
                outputs_state_stage = checkpoint(self.rnn2_wrapper,
                                                 state, inputs,  self.dummy_tensor)
            elif i == 3:
                outputs_state_stage = checkpoint(self.rnn3_wrapper,
                                                 state, inputs,  self.dummy_tensor)
        else:
            if i == 1:
                outputs_state_stage = self.rnn1(inputs, state)
            if i == 2:
                outputs_state_stage = self.rnn2(inputs, state)
            if i == 3:
                outputs_state_stage = self.rnn3(inputs, state)

        hy = outputs_state_stage[0]
        inputs = hy.unsqueeze(0)
        state_stage = (hy)

        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))

        if self.use_checkpoint:
            if i == 1:
                inputs = checkpoint(self.stage1_wrapper,
                                    inputs, self.dummy_tensor)
            elif i == 2:
                inputs = checkpoint(self.stage2_wrapper,
                                    inputs, self.dummy_tensor)
            elif i == 3:
                inputs = checkpoint(self.stage3_wrapper,
                                    inputs, self.dummy_tensor)
        else:
            if i == 1:
                inputs = self.stage1(inputs)
            if i == 2:
                inputs = self.stage2(inputs)
            if i == 3:
                inputs = self.stage3(inputs)

        inputs = torch.reshape(
            inputs,
            (seq_number, batch_size, inputs.size(
                1), inputs.size(2), inputs.size(3)),
        )

        return inputs, state_stage

    def forward(self, encoder_states, decoder_states):
        """
        Forward pass through the decoder by sequentially processing through all stages.

        Parameters:
        - encoder_states: List of encoder states.
        - decoder_states: List of decoder states.

        Returns:
        - Tensor: Output tensor of the final decoder stage.
        - tuple: Tuple of hidden states from all stages.
        """
        hidden_states = []

        inputs, state = self.forward_by_stage(
            3, None,
            encoder_states[-1],
            decoder_states[0],
        )

        hidden_states.append(state)
        for i in list(range(1, self.blocks))[::-1]:
            inputs, state = self.forward_by_stage(
                i, inputs,
                encoder_states[i - 1],
                decoder_states[self.blocks-i],
            )

            hidden_states.append(state)

        inputs = inputs.transpose(0, 1)

        return inputs, tuple(hidden_states)
