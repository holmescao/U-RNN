from collections import OrderedDict
from src.lib.model.networks.ConvRNN import CGRU_cell, CLSTM_cell

# build model
# in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4]
input_heigh = 300
input_width = 300
# input_heigh = 64
# input_width = 64
# input_heigh = 364
# input_width = 508
filter_size = 1
use_checkpoint = True
encoder_filter_size = 1
decoder_filter_size = 1

convlstm_encoder_params = [
    [
        # (input channels)
        OrderedDict({
            "conv1_leaky_1": [6, 16, encoder_filter_size, 1, 0],
        }),
        OrderedDict({
            "conv2_leaky_1": [64, 64, encoder_filter_size, 1, 0],
            "avgpool": [2, 2, 0]
        }),
        OrderedDict({
            "conv3_leaky_1": [96, 96, encoder_filter_size, 1, 0],
            "avgpool": [2, 2, 0]
        }),
    ],
    [
        CLSTM_cell(use_checkpoint=use_checkpoint,
                   shape=(input_heigh, input_width),
                   input_channels=16,
                   filter_size=filter_size,
                   num_features=64),
        CLSTM_cell(use_checkpoint=use_checkpoint,
                   shape=(input_heigh // 2, input_width // 2),
                   input_channels=64,
                   filter_size=filter_size,
                   num_features=96),
        CLSTM_cell(use_checkpoint=use_checkpoint,
                   shape=(input_heigh // 4, input_width // 4),
                   input_channels=96,
                   filter_size=filter_size,
                   num_features=96),
    ],
]

convlstm_decoder_params = [
    [
        OrderedDict(
            {"deconv1_leaky_1": [96, 96, decoder_filter_size + 1, 2, 0]}),
        OrderedDict(
            {"deconv2_leaky_1": [96, 96, decoder_filter_size + 1, 2, 0]}),
        OrderedDict({"conv3_leaky_1": [64, 16, decoder_filter_size, 1, 0]}),
    ],
    [
        CLSTM_cell(use_checkpoint=use_checkpoint,
                   shape=(input_heigh // 4, input_width // 4),
                   input_channels=96,
                   filter_size=filter_size,
                   num_features=96),
        CLSTM_cell(use_checkpoint=use_checkpoint,
                   shape=(input_heigh // 2, input_width // 2),
                   input_channels=96,
                   filter_size=filter_size,
                   num_features=96),
        CLSTM_cell(use_checkpoint=use_checkpoint,
                   shape=(input_heigh, input_width),
                   input_channels=96,
                   filter_size=filter_size,
                   num_features=64),
    ],
]

convgru_encoder_params = [
    [
        OrderedDict({
            "conv1_leaky_1": [6, 16, encoder_filter_size, 1, 0],
            "conv1_leaky_2": [16, 16, encoder_filter_size, 1, 0],
        }),
        OrderedDict({
            "conv2_leaky_1": [64, 64, encoder_filter_size, 1, 0],
            "conv2_leaky_2": [64, 64, encoder_filter_size, 1, 0],
            "avgpool": [2, 2, 0]
        }),
        OrderedDict({
            "conv3_leaky_1": [96, 96, encoder_filter_size, 1, 0],
            "conv3_leaky_2": [96, 96, encoder_filter_size, 1, 0],
            "avgpool": [2, 2, 0]
        }),
    ],
    [
        CGRU_cell(use_checkpoint=use_checkpoint,
                  shape=(input_heigh, input_width),
                  input_channels=16,
                  filter_size=filter_size,
                  num_features=64),
        CGRU_cell(use_checkpoint=use_checkpoint,
                  shape=(input_heigh // 2, input_width // 2),
                  input_channels=64,
                  filter_size=filter_size,
                  num_features=96),
        CGRU_cell(use_checkpoint=use_checkpoint,
                  shape=(input_heigh // 4, input_width // 4),
                  input_channels=96,
                  filter_size=filter_size,
                  num_features=96),
    ],
]

convgru_decoder_params = [
    [
        OrderedDict({
            "deconv1_leaky_1": [96, 96, decoder_filter_size + 1, 2, 0],
            "conv1_de_leaky_1": [96, 96, decoder_filter_size, 1, 0],
        }),
        OrderedDict({
            "deconv2_leaky_1": [96, 96, decoder_filter_size + 1, 2, 0],
            "conv2_de_leaky_2": [96, 96, decoder_filter_size, 1, 0],
        }),
        OrderedDict({
            "conv3_leaky_1": [64, 16, decoder_filter_size, 1, 0],
            "conv3_de_leaky_2": [16, 16, decoder_filter_size, 1, 0],
        }),
    ],
    [
        CGRU_cell(use_checkpoint=use_checkpoint,
                  shape=(input_heigh // 4, input_width // 4),
                  input_channels=96,
                  filter_size=filter_size,
                  num_features=96),
        CGRU_cell(use_checkpoint=use_checkpoint,
                  shape=(input_heigh // 2, input_width // 2),
                  input_channels=96,
                  filter_size=filter_size,
                  num_features=96),
        CGRU_cell(use_checkpoint=use_checkpoint,
                  shape=(input_heigh, input_width),
                  input_channels=96,
                  filter_size=filter_size,
                  num_features=64),
    ],
]
