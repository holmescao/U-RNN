from collections import OrderedDict
from src.lib.model.networks.ConvRNN import CGRU_cell


def get_network_params(use_checkpoint, input_height=500, input_width=500,
                       input_channels=63, net_cfg=None):
    """
    Build and return encoder and decoder parameter lists for the U-RNN model.

    All channel sizes are read from ``net_cfg`` (loaded from
    ``configs/network.yaml``) so that no numeric constants are hardcoded here.
    When ``net_cfg`` is None the function falls back to the legacy defaults,
    which are identical to the published architecture.

    Parameters
    ----------
    use_checkpoint : bool
        Enable gradient checkpointing inside CGRU cells.
    input_height : int
        Spatial height of the input grid (pixels).
    input_width : int
        Spatial width of the input grid (pixels).
    input_channels : int
        Number of input channels to the first encoder conv layer.
        Computed as ``historical_nums * 2 + 3``; default 63 (historical_nums=30).
    net_cfg : dict or None
        Loaded network config (output of ``load_net_config``).
        If None, built-in defaults matching the published architecture are used.

    Returns
    -------
    tuple
        (convgru_encoder_params, convgru_decoder_params)
    """
    # ── Resolve channel sizes from YAML or fall back to built-in defaults ──
    if net_cfg is not None:
        enc_conv_chs = net_cfg["encoder"]["conv_out_channels"]   # [16, 64, 96]
        enc_gru_chs  = net_cfg["encoder"]["gru_channels"]        # [64, 96, 96]
        down_factors = net_cfg["encoder"]["downsample_factors"]  # [1,  2,  2]
        enc_filter   = net_cfg["encoder"]["filter_size"]         # 1

        dec_gru_chs  = net_cfg["decoder"]["gru_channels"]        # [96, 96, 64]
        dec_conv_chs = net_cfg["decoder"]["conv_out_channels"]   # [96, 96, 16]
        up_factors   = net_cfg["decoder"]["upsample_factors"]    # [2,  2,  1]
        dec_filter   = net_cfg["decoder"]["filter_size"]         # 1
    else:
        # Published architecture defaults (identical to original hardcoded values)
        enc_conv_chs = [16, 64, 96]
        enc_gru_chs  = [64, 96, 96]
        down_factors = [1,  2,  2]
        enc_filter   = 1

        dec_gru_chs  = [96, 96, 64]
        dec_conv_chs = [96, 96, 16]
        up_factors   = [2,  2,  1]
        dec_filter   = 1

    n = len(enc_gru_chs)  # number of encoder/decoder stages

    # ── Compute per-stage spatial dimensions ──────────────────────────────────
    # Cumulative downsampling scale at the END of each encoder stage.
    scales = []
    s = 1
    for f in down_factors:
        s *= f
        scales.append(s)

    enc_spatial = [
        (input_height // scales[k], input_width // scales[k])
        for k in range(n)
    ]
    dec_spatial = [
        (input_height // scales[n - 1 - k], input_width // scales[n - 1 - k])
        for k in range(n)
    ]

    # ── Build encoder params ───────────────────────────────────────────────────
    # Encoder conv layers: input_channels → enc_conv_chs[0],
    #                      enc_gru_chs[k-1] → enc_conv_chs[k]  (k>=1)
    enc_conv_in = [input_channels] + enc_gru_chs[:-1]

    encoder_convs = []
    for k in range(n):
        d = OrderedDict()
        d[f"conv{k+1}_leaky_1"] = [enc_conv_in[k], enc_conv_chs[k], enc_filter, 1, 0]
        if down_factors[k] > 1:
            d["avgpool"] = [down_factors[k], down_factors[k], 0]
        encoder_convs.append(d)

    encoder_grus = [
        CGRU_cell(
            use_checkpoint=use_checkpoint,
            shape=enc_spatial[k],
            input_channels=enc_conv_chs[k],
            filter_size=enc_filter,
            num_features=enc_gru_chs[k],
            module="encoder",
        )
        for k in range(n)
    ]

    convgru_encoder_params = [encoder_convs, encoder_grus]

    # ── Build decoder params ───────────────────────────────────────────────────
    # Decoder deconv/conv layers: enc_gru_chs[n-1-k] → dec_conv_chs[k]
    dec_conv_in = [enc_gru_chs[n - 1 - k] for k in range(n)]

    decoder_convs = []
    for k in range(n):
        d = OrderedDict()
        if up_factors[k] > 1:
            # ConvTranspose2d (deconv): kernel = filter+1, stride = upsample_factor
            d[f"deconv{k+1}_leaky_1"] = [
                dec_conv_in[k], dec_conv_chs[k], dec_filter + 1, up_factors[k], 0
            ]
        else:
            d[f"conv{k+1}_leaky_1"] = [
                dec_conv_in[k], dec_conv_chs[k], dec_filter, 1, 0
            ]
        decoder_convs.append(d)

    # GRU at stage k receives the deconv output of stage k-1 as its x input.
    # k=0 (deepest): receives zeros placeholder → same channel count as stage0 deconv output.
    # k>0: receives the previous stage's deconv output = dec_conv_chs[k-1].
    dec_gru_input_chs = [dec_conv_chs[0]] + [dec_conv_chs[k - 1] for k in range(1, n)]

    decoder_grus = [
        CGRU_cell(
            use_checkpoint=use_checkpoint,
            shape=dec_spatial[k],
            input_channels=dec_gru_input_chs[k],
            filter_size=dec_filter,
            num_features=dec_gru_chs[k],
            module="decoder",
        )
        for k in range(n)
    ]

    convgru_decoder_params = [decoder_convs, decoder_grus]

    return convgru_encoder_params, convgru_decoder_params
