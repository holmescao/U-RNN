"""
decoder.py — Multi-scale Skip-ConvGRU decoder for U-RNN.

The decoder mirrors the encoder's 3-stage hierarchy but in reverse order
(coarsest → finest) and uses the encoder's hidden states as skip connections.

Processing order:
  Stage 3 (¼ resolution): no feature input from previous stage; GRU starts
                           from zeros (or previous decoder state), receives
                           encoder_state[2] as skip connection.
  Stage 2 (½ resolution): receives stage3's deconv output + encoder_state[1].
  Stage 1 (full resolution): receives stage2's deconv output + encoder_state[0].

The key architectural detail is that each decoder stage receives the encoder
state concatenated with the decoder's own previous hidden state as the GRU
hidden state argument (see ``CGRU_cell`` for the Skip-ConvGRU equations).

Each stage's structure:
  Skip-ConvGRU → ConvTranspose2d (upsampling, except the final stage)

The decoder hidden states carry the flood dynamics across the SWP windows;
the encoder states supply fresh rainfall-terrain forcing at each timestep.
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


class ModuleWrapperIgnores2ndArg_gru(nn.Module):
    """Checkpoint-compatible wrapper for the decoder ConvGRU.

    Argument order is (hx, x) because ``torch.utils.checkpoint.checkpoint``
    requires the first positional argument to carry gradients; ``hx``
    (the combined encoder+decoder hidden state) does so here.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, hx, x, dummy_arg=None):
        assert dummy_arg is not None, "dummy_arg is required but was None"
        return self.module(x, hx)


class Decoder(nn.Module):
    """Multi-scale decoder with Skip-ConvGRU cells and transposed convolutions.

    Parameters
    ----------
    clstm : bool
        Unused for the decoder (always uses GRU). Kept for API symmetry.
    subnets : list[dict]
        Layer specs for the three ConvTranspose2d upsampling stages.
        Index 0 → stage 3 (deepest); index 2 → stage 1 (shallowest).
    rnns : list[CGRU_cell]
        Three Skip-ConvGRU cells.  Index order matches ``subnets``.
    use_checkpoint : bool
        Enable gradient checkpointing.
    """

    def __init__(self, clstm, subnets, rnns, use_checkpoint):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        # subnets[0] / rnns[0] correspond to the DEEPEST (stage 3) decoder stage
        self.stage3 = make_layers(subnets[0])  # ¼ res → ½ res (ConvTranspose2d)
        self.stage2 = make_layers(subnets[1])  # ½ res → full res
        self.stage1 = make_layers(subnets[2])  # full res → full res (final conv)
        self.rnn3 = rnns[0]
        self.rnn2 = rnns[1]
        self.rnn1 = rnns[2]

        self.clstm = clstm
        self.use_checkpoint = use_checkpoint
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        # Checkpoint wrappers
        self.stage1_wrapper = ModuleWrapperIgnores2ndArg_cnn(self.stage1)
        self.stage2_wrapper = ModuleWrapperIgnores2ndArg_cnn(self.stage2)
        self.stage3_wrapper = ModuleWrapperIgnores2ndArg_cnn(self.stage3)
        self.rnn1_wrapper = ModuleWrapperIgnores2ndArg_gru(self.rnn1)
        self.rnn2_wrapper = ModuleWrapperIgnores2ndArg_gru(self.rnn2)
        self.rnn3_wrapper = ModuleWrapperIgnores2ndArg_gru(self.rnn3)

    def forward_by_stage(self, i, inputs, encoder_states, decoder_states=None):
        """Process one decoder stage: Skip-ConvGRU → ConvTranspose2d.

        The encoder state (from the matching encoder scale) is concatenated
        with the decoder's own previous hidden state and passed as the GRU
        hidden state.  Inside ``CGRU_cell``, the encoder state acts as a
        direct skip connection to the candidate computation.

        Parameters
        ----------
        i : int
            Stage index (1 = full, 2 = half, 3 = quarter resolution).
        inputs : Tensor or None
            Feature map from the previous (coarser) decoder stage,
            shape (1, B, F, H', W').  None for the deepest stage (stage 3).
        encoder_states : Tensor, shape (B, F, H', W')
            Encoder hidden state for the matching spatial scale.
        decoder_states : Tensor or None
            Decoder hidden state from the previous timestep.  Defaults to
            zeros (matching encoder_states shape) on the first timestep.

        Returns
        -------
        inputs : Tensor, shape (1, B, F_out, H'', W'')
            Updated feature map after GRU + transposed convolution.
        state_stage : Tensor, shape (B, F, H', W')
            New decoder hidden state for this stage.
        """
        if decoder_states is None:
            # Zero-initialise decoder state for the deepest stage's first step
            decoder_states = torch.zeros_like(encoder_states).cuda()

        # Combine encoder skip and decoder hidden state into 2F-channel hidden state
        state = torch.cat((encoder_states, decoder_states), dim=1)

        # Skip-ConvGRU: state = cat(enc, dec); inputs = feature from previous stage
        if self.use_checkpoint:
            rnn_wrappers = {1: self.rnn1_wrapper,
                            2: self.rnn2_wrapper,
                            3: self.rnn3_wrapper}
            outputs_state_stage = checkpoint(
                rnn_wrappers[i], state, inputs, self.dummy_tensor)
        else:
            rnn = getattr(self, f"rnn{i}")
            outputs_state_stage = rnn(inputs, state)

        # GRU output is the updated decoder hidden state (F channels)
        hy = outputs_state_stage[0]
        inputs = hy.unsqueeze(0)  # restore sequence dimension: (1, B, F, H', W')
        state_stage = hy

        # Transposed convolution: upsample to the next spatial scale
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))

        if self.use_checkpoint:
            stage_wrappers = {1: self.stage1_wrapper,
                              2: self.stage2_wrapper,
                              3: self.stage3_wrapper}
            inputs = checkpoint(stage_wrappers[i], inputs, self.dummy_tensor)
        else:
            stage = getattr(self, f"stage{i}")
            inputs = stage(inputs)

        inputs = torch.reshape(
            inputs,
            (seq_number, batch_size, inputs.size(1), inputs.size(2), inputs.size(3)),
        )

        return inputs, state_stage

    def forward(self, encoder_states, decoder_states):
        """Decode encoder states through all three spatial scales.

        Processing order: stage 3 (deepest / ¼ res) → stage 2 → stage 1 (full res).
        Each stage passes its feature output to the next stage as ``inputs``.

        Parameters
        ----------
        encoder_states : tuple[Tensor, Tensor, Tensor]
            Hidden states from encoder stages 1, 2, 3.
        decoder_states : list[Tensor]
            Previous decoder hidden states: [dec_state_deepest, ..., dec_state_shallowest]
            i.e. [stage3_prev, stage2_prev, stage1_prev].

        Returns
        -------
        inputs : Tensor, shape (B, S, F, H, W)
            Full-resolution output feature maps for the head.
        hidden_states : tuple[Tensor, Tensor, Tensor]
            Updated decoder hidden states (same ordering as input).
        """
        hidden_states = []

        # Start from the deepest (¼ resolution) stage with no input feature
        # encoder_states[-1] = encoder_state at ¼ resolution (stage 3)
        inputs, state = self.forward_by_stage(
            3, None,
            encoder_states[-1],   # enc stage 3 (¼ res)
            decoder_states[0],    # dec stage 3 previous state
        )
        hidden_states.append(state)

        # Stage 2 and stage 1: iterate back toward full resolution
        for i in list(range(1, self.blocks))[::-1]:  # [2, 1] for 3 blocks
            inputs, state = self.forward_by_stage(
                i, inputs,
                encoder_states[i - 1],        # matching encoder scale
                decoder_states[self.blocks - i],  # matching decoder previous state
            )
            hidden_states.append(state)

        # Transpose from (S, B, F, H, W) to (B, S, F, H, W) for the head
        inputs = inputs.transpose(0, 1)

        return inputs, tuple(hidden_states)
