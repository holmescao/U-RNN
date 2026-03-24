"""
encoder.py — Multi-scale encoder for U-RNN.

The encoder processes the spatiotemporal input (rainfall history + static terrain
features) through 3 hierarchical stages, each comprising:

  Stage 1 (full resolution,  ×1):  Conv → ConvGRU → (no pooling)
  Stage 2 (half resolution,  ×½):  Conv → ConvGRU → AvgPool(2)
  Stage 3 (quarter resolution, ×¼):  Conv → ConvGRU → AvgPool(2)

The convolutional sublayer projects the input channels to the target feature
dimension; the ConvGRU cell then updates the hidden state by integrating the
projected features with the previous hidden state.

All three stage hidden states are passed to the decoder via skip connections:
  encoder_state[0] → decoder stage 1 (full resolution)
  encoder_state[1] → decoder stage 2 (half resolution)
  encoder_state[2] → decoder stage 3 (quarter resolution)
"""

from torch import nn
from src.lib.model.networks.utils import make_layers
import torch
from torch.utils.checkpoint import checkpoint


class ModuleWrapperIgnores2ndArg_cnn(nn.Module):
    """Checkpoint-compatible wrapper that ignores a dummy second argument."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, dummy_arg=None):
        assert dummy_arg is not None, "dummy_arg is required but was None"
        return self.module(x)


class ModuleWrapperIgnores2ndArg_lstm(nn.Module):
    """Checkpoint-compatible wrapper for ConvLSTM (3-argument: x, hx, cx)."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, hx, cx, dummy_arg=None):
        assert dummy_arg is not None, "dummy_arg is required but was None"
        return self.module(x, (hx, cx))


class ModuleWrapperIgnores2ndArg_gru(nn.Module):
    """Checkpoint-compatible wrapper for ConvGRU (2-argument: x, hx)."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, hx, dummy_arg=None):
        assert dummy_arg is not None, "dummy_arg is required but was None"
        return self.module(x, hx)


class Encoder(nn.Module):
    """Multi-scale ConvGRU encoder.

    Processes the full input sequence through three spatial scales.
    At each scale, a convolutional projection layer first reduces the input
    to a compact feature map, and then a ConvGRU cell integrates the temporal
    dynamics across timesteps.

    Parameters
    ----------
    clstm : bool
        Use ConvLSTM cells instead of ConvGRU (experimental).
    subnets : list[dict]
        Layer specifications for the three convolutional projection layers.
        Built by ``make_layers`` from ``configs/network.yaml``.
    rnns : list[CGRU_cell]
        Three ConvGRU (or ConvLSTM) cells, one per spatial scale.
    use_checkpoint : bool
        Enable gradient checkpointing to reduce peak GPU memory.
    """

    def __init__(self, clstm, subnets, rnns, use_checkpoint):
        super().__init__()
        assert len(subnets) == len(rnns), \
            "Number of conv stages must match number of RNN stages."

        self.blocks = len(subnets)
        self.use_checkpoint = use_checkpoint
        self.clstm = clstm

        # Convolutional projection layers (include pooling for stages 2 & 3)
        self.stage1 = make_layers(subnets[0])  # full resolution
        self.stage2 = make_layers(subnets[1])  # includes AvgPool(2) → ½ res
        self.stage3 = make_layers(subnets[2])  # includes AvgPool(2) → ¼ res

        # Recurrent cells
        self.rnn1 = rnns[0]
        self.rnn2 = rnns[1]
        self.rnn3 = rnns[2]

        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        # Checkpoint wrappers
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
        """Process one encoder stage: Conv projection → ConvGRU update.

        Parameters
        ----------
        i : int
            Stage index (1 = full, 2 = half, 3 = quarter resolution).
        inputs : Tensor, shape (S, B, C, H, W)
            Input from the previous stage (or the model input for stage 1).
        hidden_state : Tensor or None
            GRU hidden state from the previous timestep at this scale.
        subnet : nn.Module
            Convolutional projection layer for this stage.
        rnn : CGRU_cell
            ConvGRU cell for this stage.

        Returns
        -------
        outputs_stage : Tensor, shape (1, B, F, H', W')
            New hidden state wrapped in a length-1 sequence dimension.
        state_stage : Tensor or tuple
            Updated hidden state (tensor for GRU, (h, c) tuple for LSTM).
        """
        seq_number, batch_size, input_channel, height, width = inputs.size()
        # Flatten sequence into batch for spatial convolution
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))

        if self.use_checkpoint:
            stage_wrappers = [None, self.stage1_wrapper,
                              self.stage2_wrapper, self.stage3_wrapper]
            inputs = checkpoint(stage_wrappers[i], inputs, self.dummy_tensor)
        else:
            inputs = subnet(inputs)

        # Restore sequence dimension
        inputs = torch.reshape(
            inputs,
            (seq_number, batch_size, inputs.size(1), inputs.size(2), inputs.size(3)),
        )

        # Run ConvGRU (or ConvLSTM) across the sequence
        if self.clstm:
            if self.use_checkpoint:
                hidden_state = (None, None) if hidden_state is None else hidden_state
                rnn_wrappers = [None, self.rnn1_wrapper,
                                self.rnn2_wrapper, self.rnn3_wrapper]
                outputs_state_stage = checkpoint(
                    rnn_wrappers[i], inputs,
                    hidden_state[0], hidden_state[1], self.dummy_tensor)
            else:
                outputs_state_stage = rnn(inputs, hidden_state)
            hy, cy = outputs_state_stage[0], outputs_state_stage[1]
            outputs_stage = hy.unsqueeze(0)
            state_stage = (hy, cy)
        else:
            if self.use_checkpoint:
                rnn_wrappers = [None, self.rnn1_wrapper,
                                self.rnn2_wrapper, self.rnn3_wrapper]
                outputs_state_stage = checkpoint(
                    rnn_wrappers[i], inputs, hidden_state, self.dummy_tensor)
            else:
                outputs_state_stage = rnn(inputs, hidden_state)
            hy = outputs_state_stage[0]
            outputs_stage = hy.unsqueeze(0)
            state_stage = hy

        return outputs_stage, state_stage

    def forward(self, inputs, state_stages):
        """Encode the input sequence through all three spatial scales.

        Parameters
        ----------
        inputs : Tensor, shape (S, B, C, H, W)
            Full-resolution input sequence (SWP window of S timesteps).
        state_stages : list[Tensor]
            Previous GRU hidden states for stages 1, 2, 3 (in order).

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Updated hidden states for stages 1, 2, 3 — passed to decoder
            as skip connections.
        """
        hidden_states = []

        for i in range(1, self.blocks + 1):
            # Each stage's input is the PREVIOUS stage's output (downsampled)
            inputs, state_stage = self.forward_by_stage(
                i, inputs,
                state_stages[i - 1],
                getattr(self, "stage" + str(i)),
                getattr(self, "rnn" + str(i)),
            )
            hidden_states.append(state_stage)

        return tuple(hidden_states)
