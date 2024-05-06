import torch
import os
from pathlib import Path
import sys


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

    # Output for debugging purposes
    print("LOCAL RANK:", local_rank)
    print("RANK:", rank)
    print("WORLD SIZE:", world_size)

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


def initialize_states(device):
    """
    Initializes or resets the states for the encoder and decoder.

    Parameters:
    - device: The device to which the state tensors are to be moved.

    Returns:
    - Tuple of tensors representing the initialized states.
    """
    prev_encoder_state1 = torch.zeros(
        (1, 64, 500, 500)).to(device, dtype=torch.float32)
    prev_encoder_state2 = torch.zeros(
        (1, 96, 250, 250)).to(device, dtype=torch.float32)
    prev_encoder_state3 = torch.zeros(
        (1, 96, 125, 125)).to(device, dtype=torch.float32)

    prev_decoder_state1 = torch.zeros(
        (1, 96, 125, 125)).to(device, dtype=torch.float32)
    prev_decoder_state2 = torch.zeros(
        (1, 96, 250, 250)).to(device, dtype=torch.float32)
    prev_decoder_state3 = torch.zeros(
        (1, 64, 500, 500)).to(device, dtype=torch.float32)

    return prev_encoder_state1, prev_encoder_state2, prev_encoder_state3, \
        prev_decoder_state1, prev_decoder_state2, prev_decoder_state3
