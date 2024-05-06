from collections import OrderedDict
from src.lib.model.networks.ConvRNN import CGRU_cell


def get_network_params(use_checkpoint):
    """
    Configures and returns the parameters for both encoder and decoder networks in a
    convolutional GRU-based architecture. This configuration includes defining
    layer specifications and GRU cell parameters.

    Parameters:
    - use_checkpoint: Boolean indicating whether to use gradient checkpointing for saving memory.

    Returns:
    - tuple: Contains two lists of OrderedDicts. The first list corresponds to the encoder configuration,
             and the second list corresponds to the decoder configuration. Each list includes configurations
             for convolutional layers followed by ConvGRU cells reflecting the architectural stages.
    """
    # Global configuration settings
    input_height = 500
    input_width = 500
    filter_size = 1
    encoder_filter_size = 1
    decoder_filter_size = 1

    # Encoder parameters with convolutional layers and ConvGRU cells
    convgru_encoder_params = [
        [
            # First stage: convolutional layers
            OrderedDict({
                "conv1_leaky_1": [63, 16, encoder_filter_size, 1, 0]
            }),
            # Second stage: convolutional layer followed by average pooling
            OrderedDict({
                "conv2_leaky_1": [64, 64, encoder_filter_size, 1, 0],
                "avgpool": [2, 2, 0]
            }),
            # Third stage: similar to the second
            OrderedDict({
                "conv3_leaky_1": [96, 96, encoder_filter_size, 1, 0],
                "avgpool": [2, 2, 0]
            }),
        ],
        [
            # Corresponding ConvGRU cells for each convolutional stage
            CGRU_cell(use_checkpoint=use_checkpoint, shape=(input_height, input_width),
                      input_channels=16, filter_size=filter_size, num_features=64, module="encoder"),
            CGRU_cell(use_checkpoint=use_checkpoint, shape=(input_height//2, input_width//2),
                      input_channels=64, filter_size=filter_size, num_features=96, module="encoder"),
            CGRU_cell(use_checkpoint=use_checkpoint, shape=(input_height//4, input_width//4),
                      input_channels=96, filter_size=filter_size, num_features=96, module="encoder"),
        ],
    ]

    # Decoder parameters with deconvolutional layers and ConvGRU cells
    convgru_decoder_params = [
        [
            # Deconvolutional layers to upsample back to the original dimensions
            OrderedDict({
                "deconv1_leaky_1": [96, 96, decoder_filter_size+1, 2, 0]
            }),
            OrderedDict({
                "deconv2_leaky_1": [96, 96, decoder_filter_size+1, 2, 0]
            }),
            # Final convolutional layer to produce the output
            OrderedDict({
                "conv3_leaky_1": [64, 16, decoder_filter_size, 1, 0]
            }),
        ],
        [
            # Corresponding ConvGRU cells for each deconvolutional stage
            CGRU_cell(use_checkpoint=use_checkpoint, shape=(input_height//4, input_width//4),
                      input_channels=96, filter_size=filter_size, num_features=96, module="decoder"),
            CGRU_cell(use_checkpoint=use_checkpoint, shape=(input_height//2, input_width//2),
                      input_channels=96, filter_size=filter_size, num_features=96, module="decoder"),
            CGRU_cell(use_checkpoint=use_checkpoint, shape=(input_height, input_width),
                      input_channels=96, filter_size=filter_size, num_features=64, module="decoder"),
        ],
    ]

    return convgru_encoder_params, convgru_decoder_params
