"""
model.py — U-RNN top-level Encoder-Decoder model.

``ED`` (Encoder-Decoder) is the main model class.  It chains three components:

  Encoder  → multi-scale ConvGRU encoder, 3 stages (full, ½, ¼ resolution)
  Decoder  → multi-scale Skip-ConvGRU decoder, 3 stages (¼, ½, full)
  Head     → YOLOXHead: dual-branch prediction (flood depth + wet/dry extent)

The model processes one SWP window of ``seq_num`` timesteps per forward pass.
Hidden states are explicitly passed between calls so the pre-warming stage can
propagate states from t=0 to the window start without storing gradients.
"""

from torch import nn
from src.lib.model.networks.decoder import Decoder
from src.lib.model.networks.encoder import Encoder
from src.lib.model.networks.head.flood_head import YOLOXHead


class ED(nn.Module):
    """U-RNN Encoder-Decoder model for spatiotemporal urban flood nowcasting.

    Architecture (Cao et al., J. Hydrology 2025):
      • 3-stage encoder: Conv→Skip-ConvGRU→AvgPool (scales: 1×, ½×, ¼×)
      • 3-stage decoder: ConvGRU→ConvTranspose2d  (scales: ¼×, ½×, 1×)
        Skip connections pass each encoder stage's hidden state directly into
        the matching decoder stage's ConvGRU (see ``ConvRNN.CGRU_cell``).
      • YOLOXHead: two independent branches predicting
          – flood depth in mm (regression, leaky-ReLU output)
          – wet/dry probability per cell (classification, sigmoid output)

    The classification output is used during inference to zero out depth
    predictions in cells classified as dry (``correction_depth`` in the head).
    During training, both heads contribute to the combined ``FocalBCE_and_WMSE``
    loss via the ``pred`` dict returned by ``main.accumulate_predictions``.

    Parameters
    ----------
    clstm_flag : bool
        If True, use ConvLSTM cells instead of ConvGRU (experimental).
    encoder_params : list
        ``[subnets, rnns]`` — convolutional and recurrent layers for the encoder.
    decoder_params : list
        ``[subnets, rnns]`` — convolutional and recurrent layers for the decoder.
    cls_thred : float
        Wet/dry probability threshold used in ``correction_depth`` (default 0.5).
    use_checkpoint : bool
        Enable gradient checkpointing to reduce peak GPU memory.
    """

    def __init__(self, clstm_flag, encoder_params, decoder_params,
                 cls_thred=0.5, use_checkpoint=True,
                 input_height=500, input_width=500):
        super().__init__()
        self.encoder = Encoder(clstm_flag,
                               encoder_params[0], encoder_params[1],
                               use_checkpoint=use_checkpoint)
        self.decoder = Decoder(clstm_flag,
                               decoder_params[0], decoder_params[1],
                               use_checkpoint=use_checkpoint)
        self.head = YOLOXHead(cls_thred, use_checkpoint=use_checkpoint,
                              input_height=input_height, input_width=input_width)

    def forward(self,
                input_t,
                prev_encoder_state1, prev_encoder_state2, prev_encoder_state3,
                prev_decoder_state1, prev_decoder_state2, prev_decoder_state3):
        """Forward pass for one SWP window.

        Parameters
        ----------
        input_t : Tensor, shape (B, S, C, H, W)
            Input window of S timesteps.  C = historical_nums*2+3 channels
            (normalised rainfall history + DEM + imperviousness + manhole).
        prev_encoder_state{1,2,3} : Tensor
            Encoder GRU hidden states from the end of the previous window
            (or zeros at the start of the sequence / after pre-warming).
        prev_decoder_state{1,2,3} : Tensor
            Decoder GRU hidden states, same provenance.

        Returns
        -------
        reg_output_t : Tensor, shape (B, S, H, W)
            Predicted flood depth (normalised to [0,1]) after applying the
            classification mask (dry cells are zeroed out by the head).
        encoder_state_t{1,2,3} : Tensor
            Updated encoder hidden states at the end of the window.
        decoder_state_t{1,2,3} : Tensor
            Updated decoder hidden states.
        """
        prev_encoder_state = [prev_encoder_state1,
                               prev_encoder_state2,
                               prev_encoder_state3]
        prev_decoder_state = [prev_decoder_state1,
                               prev_decoder_state2,
                               prev_decoder_state3]

        # Encoder expects (S, B, C, H, W); input_t arrives as (B, S, C, H, W)
        input_t = input_t.permute(1, 0, 2, 3, 4)

        # Multi-scale encoding: produces 3 encoder hidden states at ×1, ×½, ×¼
        encoder_state_t = self.encoder(input_t, prev_encoder_state)

        # Multi-scale decoding with skip connections; output shape: (B, S, F, H, W)
        output_t, decoder_state_t = self.decoder(encoder_state_t, prev_decoder_state)

        # Dual-output head: reg (depth) + cls (wet/dry probability)
        # Head applies correction_depth: zeroes out depth where cls < cls_thred
        output_t = self.head(output_t)  # (S, B, 2, H, W)

        # Extract regression output; squeeze channel dim to match label (B, S, H, W)
        reg_output_t = output_t[:, :, 0]  # depth channel (index 0)

        # Unpack states for explicit state passing (required for SWP pre-warming)
        encoder_state_t1, encoder_state_t2, encoder_state_t3 = encoder_state_t
        decoder_state_t1, decoder_state_t2, decoder_state_t3 = decoder_state_t

        return (reg_output_t,
                encoder_state_t1, encoder_state_t2, encoder_state_t3,
                decoder_state_t1, decoder_state_t2, decoder_state_t3)
