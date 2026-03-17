import torch
import os
from pathlib import Path
import sys
from src.lib.utils.net_config import get_state_shapes


def initialize_environment_variables():
    """
    Initializes and prints relevant environment variables for distributed processing.
    Sets up the Python environment by appending the project's root directory to sys.path if it's not already included.
    Returns the local rank, rank, and world size for distributed computation.

    Returns:
    - tuple: Contains local_rank, rank, and world_size as integers.
    """
    file_path = Path(__file__).resolve()
    root_path = file_path.parents[0]

    if str(root_path) not in sys.path:
        sys.path.append(str(root_path))
    # Update root_path to be a relative path
    root_path = Path(os.path.relpath(root_path, Path.cwd()))

    local_rank = int(os.getenv("LOCAL_RANK", -1))
    rank = int(os.getenv("RANK", -1))
    world_size = int(os.getenv("WORLD_SIZE", -1))

    # Only print DDP rank info when actually running in distributed mode
    if local_rank != -1:
        print(f"DDP: LOCAL_RANK={local_rank}, RANK={rank}, WORLD_SIZE={world_size}")

    return local_rank, rank


def to_device(inputs, device):
    """
    Transfer all tensors in the provided dictionary to the specified device.

    Parameters:
    - inputs: Dictionary of tensors.
    - device: The device to transfer tensors to.

    Returns:
    - inputs: Dictionary with all tensors transferred to the specified device.
    """
    return {key: value.to(device) for key, value in inputs.items()}


def initialize_states(device, input_height=500, input_width=500, net_cfg=None):
    """
    Initializes or resets the encoder and decoder hidden states to zero tensors.

    The shape of each state is derived from ``net_cfg`` (network architecture
    config loaded from ``configs/network.yaml``) and the runtime spatial
    dimensions ``input_height`` / ``input_width``.  When ``net_cfg`` is None
    the function falls back to the legacy 500×500 defaults so that callers
    that have not yet been updated continue to work.

    Parameters
    ----------
    device : torch.device
        Target device for the state tensors.
    input_height : int
        Spatial height of the input grid.
    input_width : int
        Spatial width of the input grid.
    net_cfg : dict or None
        Loaded network config (output of ``load_net_config``).
        If None, uses hardcoded legacy defaults (500×500, channels 64/96/96/96/96/64).

    Returns
    -------
    tuple of 6 torch.Tensor
        (enc_state1, enc_state2, enc_state3, dec_state1, dec_state2, dec_state3)
    """
    if net_cfg is not None:
        shapes = get_state_shapes(net_cfg, input_height, input_width)
    else:
        # Legacy fallback — identical to the original hardcoded values
        H, W = input_height, input_width
        shapes = [
            (1, 64, H,     W    ),
            (1, 96, H // 2, W // 2),
            (1, 96, H // 4, W // 4),
            (1, 96, H // 4, W // 4),
            (1, 96, H // 2, W // 2),
            (1, 64, H,     W    ),
        ]

    states = tuple(
        torch.zeros(shape, device=device, dtype=torch.float32)
        for shape in shapes
    )
    return states
