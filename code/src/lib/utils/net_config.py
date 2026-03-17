"""
src/lib/utils/net_config.py
============================
Utility for loading and querying the network architecture YAML config.

Usage
-----
    from src.lib.utils.net_config import load_net_config, get_state_shapes

    net_cfg = load_net_config("configs/network.yaml")

    # State shapes for initialize_states / prepare_initial_states
    shapes = get_state_shapes(net_cfg, input_height=500, input_width=500)
    # shapes = [(1, 64, 500, 500), (1, 96, 250, 250), ...]

    # Input channels to the network (derived from historical_nums)
    input_channels = get_input_channels(net_cfg, historical_nums=30)
"""

import os
import yaml


# Default path: relative to this file (src/lib/utils/), go up 3 levels to reach the code root
_DEFAULT_CFG = os.path.join(os.path.dirname(__file__),
                             "..", "..", "..", "configs", "network.yaml")


def load_net_config(cfg_path=None):
    """
    Load the network architecture YAML config.

    Parameters
    ----------
    cfg_path : str or None
        Path to the YAML file.  If None, uses the project default at
        ``configs/network.yaml`` (relative to the code root).

    Returns
    -------
    dict
        Parsed YAML content under the ``model`` key.
    """
    if cfg_path is None:
        cfg_path = os.path.normpath(_DEFAULT_CFG)

    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(
            f"Network config not found: {cfg_path}\n"
            f"Make sure configs/network.yaml exists or pass --net_config."
        )

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return raw["model"]


def get_state_shapes(net_cfg, input_height, input_width):
    """
    Compute the (C, H, W) shapes for all 6 encoder+decoder hidden states.

    The spatial dimension at each stage is derived from the encoder's
    ``downsample_factors``:
        stage 0 → full resolution
        stage 1 → full / cumulative_scale_after_stage0
        stage 2 → full / cumulative_scale_after_stage1
    The decoder mirrors the encoder in reverse order.

    Parameters
    ----------
    net_cfg : dict
        Loaded model config (output of ``load_net_config``).
    input_height : int
        Spatial height of the input grid.
    input_width : int
        Spatial width of the input grid.

    Returns
    -------
    list of tuple
        Six (batch=1, channels, H, W) shape tuples:
        [enc_state1, enc_state2, enc_state3, dec_state1, dec_state2, dec_state3]
    """
    enc_gru_chs = net_cfg["encoder"]["gru_channels"]        # [64, 96, 96]
    dec_gru_chs = net_cfg["decoder"]["gru_channels"]        # [96, 96, 64]
    down_factors = net_cfg["encoder"]["downsample_factors"]  # [1, 2, 2]

    n_stages = len(enc_gru_chs)
    assert len(dec_gru_chs) == n_stages, "encoder/decoder stage counts must match"
    assert len(down_factors) == n_stages

    # Cumulative spatial scale for each stage.
    # Stage k state lives at (H / scale_k, W / scale_k).
    # The downsampling at stage k is applied BEFORE the GRU, so the GRU state
    # at stage k already sees the downsampled resolution.
    scale = 1
    spatial = []
    for f in down_factors:
        scale *= f
        spatial.append(scale)

    # Build encoder state shapes (stage 0, 1, 2 → scale 1, 2, 4 for [1,2,2])
    enc_shapes = [
        (1, enc_gru_chs[k], input_height // spatial[k], input_width // spatial[k])
        for k in range(n_stages)
    ]

    # Decoder mirrors encoder in reverse: decoder stage 0 is at the deepest level
    dec_shapes = [
        (1, dec_gru_chs[k], input_height // spatial[n_stages - 1 - k],
         input_width // spatial[n_stages - 1 - k])
        for k in range(n_stages)
    ]

    return enc_shapes + dec_shapes


def get_input_channels(net_cfg, historical_nums):
    """
    Compute the total number of input channels to the network.

    The input tensor concatenates:
        - historical_nums rainfall fields
        - historical_nums cumulative-rainfall fields
        - 1 DSM / DEM map
        - 1 impervious surface map
        - 1 drainage manhole map
    Total = historical_nums * 2 + 3

    Parameters
    ----------
    net_cfg : dict
        Loaded model config (unused, kept for API consistency).
    historical_nums : int
        Number of past rainfall timesteps used as input features.

    Returns
    -------
    int
    """
    return historical_nums * 2 + 3
