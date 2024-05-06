from torch import nn
from src.lib.model.networks.decoder import Decoder
from src.lib.model.networks.encoder import Encoder
from src.lib.model.networks.head.flood_head import YOLOXHead


class ED(nn.Module):
    """
    ED (Encoder-Decoder) model which processes input through an encoder, a decoder,
    and a head for final output. It's structured to handle sequential input for
    tasks such as segmentation or object detection in spatiotemporal data.
    """

    def __init__(self, clstm_flag, encoder_params, decoder_params, cls_thred=0.5, use_checkpoint=True):
        """
        Initializes the ED model with encoder, decoder, and head components.

        Parameters:
        - clstm_flag: Flag to use LSTM (True) or GRU (False) in encoder and decoder.
        - encoder_params: Parameters for the encoder (list of subnets, list of RNNs).
        - decoder_params: Parameters for the decoder (list of subnets, list of RNNs).
        - cls_thred: Threshold for classification head.
        - use_checkpoint: Use gradient checkpointing to save memory.
        """
        super().__init__()

        self.encoder = Encoder(clstm_flag,
                               encoder_params[0],
                               encoder_params[1],
                               use_checkpoint=use_checkpoint)
        self.decoder = Decoder(clstm_flag,
                               decoder_params[0],
                               decoder_params[1],
                               use_checkpoint=use_checkpoint)
        self.head = YOLOXHead(cls_thred, use_checkpoint=use_checkpoint)

    def forward(self, input_t,
                prev_encoder_state1, prev_encoder_state2, prev_encoder_state3,
                prev_decoder_state1, prev_decoder_state2, prev_decoder_state3
                ):
        """
        Forward pass of the ED model, processing inputs through the encoder, decoder, and head.

        Parameters:
        - input_t: Input tensor of shape (B, S, C, H, W).
        - prev_encoder_state1, prev_encoder_state2, prev_encoder_state3: Previous states for encoder layers.
        - prev_decoder_state1, prev_decoder_state2, prev_decoder_state3: Previous states for decoder layers.

        Returns:
        - tuple: Output tensors from the model and the new states for encoder and decoder.
        """
        prev_encoder_state = [prev_encoder_state1,
                              prev_encoder_state2, prev_encoder_state3]
        prev_decoder_state = [prev_decoder_state1,
                              prev_decoder_state2, prev_decoder_state3]

        input_t = input_t.permute(1, 0, 2, 3, 4)  # to S,B,C,64,64

        encoder_state_t = self.encoder(input_t, prev_encoder_state)

        # (B, S, F, H, W)
        output_t, decoder_state_t = self.decoder(
            encoder_state_t, prev_decoder_state)  # (B, S, F, H, W)

        output_t = self.head(output_t)

        # Separate regression and classification outputs
        reg_output_t, cls_output_t = output_t[:, :, 0:1], output_t[:, :, 1:]

        # Unpack new states for encoder and decoder to pass back
        encoder_state_t1, encoder_state_t2, encoder_state_t3 = encoder_state_t
        decoder_state_t1, decoder_state_t2, decoder_state_t3 = decoder_state_t

        return reg_output_t,  encoder_state_t1, encoder_state_t2, encoder_state_t3, decoder_state_t1, decoder_state_t2, decoder_state_t3
