from collections import OrderedDict
from src.lib.model.networks.ConvRNN import CGRU_cell, CLSTM_cell


# build model
# in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4]
# input_heigh = 500
# input_width = 500
input_heigh = 64
input_width = 64
filter_size = 5
use_checkpoint = True
encoder_filter_size = 3
decoder_filter_size = 3


convlstm_encoder_params = [
    [
        # (input channels)
        OrderedDict({"conv1_leaky_1": [6, 16, encoder_filter_size, 1, 0],
                     }),
        OrderedDict({"conv2_leaky_1": [64, 64, encoder_filter_size, 1, 0],
                     "avgpool": [2, 2, 0]}),
        OrderedDict({"conv3_leaky_1": [96, 96, encoder_filter_size, 1, 0],
                     "avgpool": [2, 2, 0]}),
    ],
    [
        CLSTM_cell(use_checkpoint=use_checkpoint, shape=(input_heigh, input_width),
                   input_channels=16,
                   filter_size=filter_size, num_features=64),
        CLSTM_cell(use_checkpoint=use_checkpoint, shape=(input_heigh//2, input_width//2),
                   input_channels=64,
                   filter_size=filter_size, num_features=96),
        CLSTM_cell(use_checkpoint=use_checkpoint, shape=(input_heigh//4, input_width//4),
                   input_channels=96,
                   filter_size=filter_size, num_features=96),
    ],
]

convlstm_decoder_params = [
    [
        OrderedDict(
            {"deconv1_leaky_1": [96, 96, decoder_filter_size+1, 2, 0]}),
        OrderedDict(
            {"deconv2_leaky_1": [96, 96, decoder_filter_size+1, 2, 0]}),
        OrderedDict(
            {"conv3_leaky_1": [64, 16, decoder_filter_size, 1, 0]}
        ),
    ],
    [
        CLSTM_cell(use_checkpoint=use_checkpoint, shape=(input_heigh//4, input_width//4),
                   input_channels=96,
                   filter_size=filter_size, num_features=96),
        CLSTM_cell(use_checkpoint=use_checkpoint, shape=(input_heigh//2, input_width//2),
                   input_channels=96,
                   filter_size=filter_size, num_features=96),
        CLSTM_cell(use_checkpoint=use_checkpoint, shape=(input_heigh, input_width),
                   input_channels=96,
                   filter_size=filter_size, num_features=64),
    ],
]


convgru_encoder_params = [
    [
        # v2
        OrderedDict({"conv1_leaky_1": [1, 16, encoder_filter_size, 1, 1],
                     }),
        OrderedDict({"conv2_leaky_1": [64, 64, encoder_filter_size, 2, 1],
                     }),
        OrderedDict({"conv3_leaky_1": [96, 96, encoder_filter_size, 2, 1],
                     }),
        # v1
        # OrderedDict({"conv1_leaky_1": [5, 16, encoder_filter_size, 1, 0],
        #              }),
        # OrderedDict({"conv2_leaky_1": [64, 64, encoder_filter_size, 1, 0],
        #              "avgpool": [2, 2, 0]}),
        # OrderedDict({"conv3_leaky_1": [96, 96, encoder_filter_size, 1, 0],
        #              "avgpool": [2, 2, 0]}),
    ],
    [

        CGRU_cell(use_checkpoint=use_checkpoint, shape=(input_heigh, input_width),
                  input_channels=16,
                  filter_size=filter_size, num_features=64, module="encoder"),
        CGRU_cell(use_checkpoint=use_checkpoint, shape=(input_heigh//2, input_width//2),
                  input_channels=64,
                  filter_size=filter_size, num_features=96, module="encoder"),
        CGRU_cell(use_checkpoint=use_checkpoint, shape=(input_heigh//4, input_width//4),
                  input_channels=96,
                  filter_size=filter_size, num_features=96, module="encoder"),
    ],
]

convgru_decoder_params = [
    [
        # v2
        OrderedDict(
            {"deconv1_leaky_1": [96, 96, decoder_filter_size+1, 2, 1]}),
        OrderedDict(
            {"deconv2_leaky_1": [96, 96, decoder_filter_size+1, 2, 1]}),
        OrderedDict(
            {"conv3_leaky_1": [64, 16, decoder_filter_size, 1, 1]}),

        # v1
        # OrderedDict(
        #     {"deconv1_leaky_1": [96, 96, decoder_filter_size+1, 2, 0]}),
        # OrderedDict(
        #     {"deconv2_leaky_1": [96, 96, decoder_filter_size+1, 2, 0]}),
        # OrderedDict(
        #     {"conv3_leaky_1": [64, 16, decoder_filter_size, 1, 0]}
        # ),
    ],
    [
        CGRU_cell(use_checkpoint=use_checkpoint, shape=(input_heigh//4, input_width//4),
                  input_channels=96,
                  filter_size=filter_size, num_features=96, module="decoder"),
        CGRU_cell(use_checkpoint=use_checkpoint, shape=(input_heigh//2, input_width//2),
                  input_channels=96,
                  filter_size=filter_size, num_features=96, module="decoder"),
        CGRU_cell(use_checkpoint=use_checkpoint, shape=(input_heigh, input_width),
                  input_channels=96,
                  filter_size=filter_size, num_features=64, module="decoder"),
    ],
]
