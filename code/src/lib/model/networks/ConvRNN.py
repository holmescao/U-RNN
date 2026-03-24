"""
ConvRNN.py — Convolutional Recurrent Cells for U-RNN.

Implements two variants of the Skip-ConvGRU cell used in U-RNN
(Cao et al., J. Hydrology 2025):

  • Encoder ConvGRU  — standard ConvGRU; hidden state h has shape (B, F, H, W).
  • Decoder Skip-ConvGRU — the key architectural innovation.  The decoder cell
    receives the ENCODER state e_t (skip connection) concatenated with its own
    previous hidden state d_{t-1} as a combined 2F-channel input.  This lets
    each decoder stage directly see the current-timestep encoder features,
    which represent the rainfall + terrain forcing at the matching scale.

    Decoder gate equations (all operations are spatial convolutions):
        combined   = cat(x_t, e_t, d_{t-1})          # shape: (B, in_ch+2F, H, W)
        z_t, r_t   = split( σ(W_z * combined) )       # update and reset gates
        candidate  = tanh(W_h * cat(x_t, e_t, r_t⊙d_{t-1}))
        d_t        = (1 - z_t) ⊙ d_{t-1} + z_t ⊙ candidate

    The encoder state e_t acts as a *fixed skip input* — it influences both the
    gate and the candidate but is NOT updated by the GRU equations.  Only d_t
    (num_features channels) is returned as the new hidden state.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class ModuleWrapperIgnores2ndArg(nn.Module):
    """Wraps any nn.Module so it can be used with gradient checkpointing.

    ``torch.utils.checkpoint.checkpoint`` requires all inputs to be Tensors.
    The standard workaround is to pass a ``dummy_tensor`` (requires_grad=True)
    as a second argument that is ignored inside the module.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, dummy_arg=None):
        assert dummy_arg is not None
        return self.module(x)


class CGRU_cell(nn.Module):
    """Convolutional GRU cell for the U-RNN encoder or decoder.

    When ``module="encoder"`` this is a standard ConvGRU cell.
    When ``module="decoder"`` this is the Skip-ConvGRU described in the paper:
    the hidden state passed in is ``cat(encoder_state, decoder_state)`` and the
    encoder state acts as a direct skip connection into the candidate computation.

    Parameters
    ----------
    use_checkpoint : bool
        Enable gradient checkpointing (trades compute for GPU memory).
    shape : tuple[int, int]
        Spatial dimensions (H, W) of the feature maps at this scale.
    input_channels : int
        Channels of the incoming feature map x_t.
        • Encoder: conv output channels of the same stage.
        • Decoder: deconv output channels of the *previous* (coarser) stage.
    filter_size : int
        Kernel size for all convolutional gates (typically 5).
    num_features : int
        Number of hidden-state channels F (GRU output channels).
    module : str
        ``"encoder"`` or ``"decoder"`` — controls the Skip-ConvGRU variant.
    """

    def __init__(self, use_checkpoint, shape, input_channels, filter_size,
                 num_features, module):
        super(CGRU_cell, self).__init__()
        self.shape = shape
        self.input_channels = int(input_channels)
        self.filter_size = filter_size
        self.num_features = int(num_features)
        self.padding = (filter_size - 1) // 2
        self.module = module

        if self.module == "encoder":
            # Standard ConvGRU: gates take cat(x_t, h_{t-1})
            conv_input_channels = self.input_channels + self.num_features
        elif self.module == "decoder":
            # Skip-ConvGRU: gates take cat(x_t, e_t, d_{t-1})
            # hidden_state passed in = cat(enc_state [F], dec_state [F]) = 2F channels
            conv_input_channels = self.input_channels + 2 * self.num_features

        conv_output_channels = 2 * self.num_features  # update gate z + reset gate r

        # conv1: produces update gate z and reset gate r (each F channels)
        self.conv1 = nn.Sequential(
            nn.Conv2d(conv_input_channels, conv_output_channels,
                      self.filter_size, 1, self.padding),
            nn.GroupNorm(conv_output_channels // 32, conv_output_channels),
        )
        # conv2: produces candidate hidden state h̃ (F channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv_input_channels, self.num_features,
                      self.filter_size, 1, self.padding),
            nn.GroupNorm(self.num_features // 32, self.num_features),
        )

        self.use_checkpoint = use_checkpoint
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        self.conv1_module_wrapper = ModuleWrapperIgnores2ndArg(self.conv1)
        self.conv2_module_wrapper = ModuleWrapperIgnores2ndArg(self.conv2)

    def forward(self, inputs=None, hidden_state=None, seq_len=1):
        """Process ``seq_len`` timesteps through the ConvGRU cell.

        Parameters
        ----------
        inputs : Tensor or None
            Shape ``(S, B, in_ch, H, W)``.  If None (decoder stage-3 init),
            x_t is treated as zeros.
        hidden_state : Tensor or None
            • Encoder: previous hidden state h_{t-1}, shape ``(B, F, H, W)``.
              Defaults to zeros when None.
            • Decoder: ``cat(encoder_state, decoder_state)`` from the current
              timestep, shape ``(B, 2F, H, W)``.
        seq_len : int
            Number of timesteps to unroll (usually 1 for the decoder;
            seq_num for the encoder during pre-warming).

        Returns
        -------
        Tensor
            Stacked hidden states, shape ``(S, B, F, H, W)``.
        """
        if hidden_state is None:
            # Zero-initialize encoder hidden state at the start of each sequence
            htprev = torch.zeros(inputs.size(1), self.num_features,
                                 self.shape[0], self.shape[1]).cuda()
        else:
            htprev = hidden_state  # (B, F) for encoder or (B, 2F) for decoder

        output_inner = []

        for index in range(seq_len):
            if inputs is None:
                # Decoder stage-3: no input from previous stage (deepest level)
                x = torch.zeros(htprev.size(0), self.input_channels,
                                self.shape[0], self.shape[1]).cuda()
            else:
                x = inputs[index, ...]  # (B, in_ch, H, W)

            # ── Gate computation ──────────────────────────────────────────
            # Encoder: combined = cat(x_t, h_{t-1})
            # Decoder: combined = cat(x_t, cat(e_t, d_{t-1}))  [= cat(x_t, htprev)]
            combined_1 = torch.cat((x, htprev), 1)
            if self.use_checkpoint:
                gates = checkpoint(self.conv1_module_wrapper, combined_1,
                                   self.dummy_tensor)
            else:
                gates = self.conv1(combined_1)

            zgate, rgate = torch.split(gates, self.num_features, dim=1)
            z = torch.sigmoid(zgate)  # update gate
            r = torch.sigmoid(rgate)  # reset gate

            # ── Candidate hidden state h̃ ──────────────────────────────────
            if self.module == "encoder":
                # Standard ConvGRU: h̃ = tanh(W * cat(x_t, r ⊙ h_{t-1}))
                combined_2 = torch.cat((x, r * htprev), 1)
            elif self.module == "decoder":
                # Skip-ConvGRU: split htprev into encoder state (e_t) and
                # decoder state (d_{t-1}); apply reset gate only to d_{t-1}.
                # h̃ = tanh(W * cat(x_t, e_t, r ⊙ d_{t-1}))
                etprev, dtprev = torch.split(htprev, self.num_features, dim=1)
                combined_2 = torch.cat((x, etprev, r * dtprev), 1)

            if self.use_checkpoint:
                ht = checkpoint(self.conv2_module_wrapper, combined_2,
                                self.dummy_tensor)
            else:
                ht = self.conv2(combined_2)
            ht = torch.tanh(ht)

            # ── Hidden state update ───────────────────────────────────────
            if self.module == "encoder":
                # h_t = (1 - z) ⊙ h_{t-1} + z ⊙ h̃
                htnext = (1 - z) * htprev + z * ht
            elif self.module == "decoder":
                # Update only the decoder component d_{t-1} → d_t
                # (encoder state e_t is a pass-through skip; it is NOT updated here)
                htnext = (1 - z) * dtprev + z * ht  # shape: (B, F, H, W)

            output_inner.append(htnext)
            htprev = htnext  # for encoder multi-step unrolling

        return torch.stack(output_inner)  # (S, B, F, H, W)
