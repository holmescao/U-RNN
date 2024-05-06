import torch
import os
import platform
import subprocess
import torch.distributed as dist
from contextlib import contextmanager
import time
import glob
from pathlib import Path
import re


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    A context manager to synchronize distributed processes at the beginning and end.

    Parameters:
    - local_rank: The local rank of the process.
    """
    if local_rank not in [-1, 0]:
        dist.barrier()
    yield
    if local_rank == 0:
        dist.barrier()


def device_count():
    """
    Safely counts the number of available CUDA devices.

    Returns:
    - The count of available CUDA devices.
    """
    assert platform.system() in (
        'Linux', 'Windows'), 'device_count() only supported on Linux or Windows'
    try:
        cmd = 'nvidia-smi -L | wc -l' if platform.system() == 'Linux' else 'nvidia-smi -L | find /c /v ""'
        return int(subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().strip())
    except subprocess.CalledProcessError:
        return 0


def select_device(device='', batch_size=0, newline=True):
    """
    Selects and sets the appropriate computing device for running the model.

    Parameters:
    - device: Device identifier.
    - batch_size: Batch size to validate device compatibility.
    - newline: Whether to append a newline at the end of the output string.

    Returns:
    - A torch.device object configured to use a specific device.
    """
    s = f'Python-{platform.python_version()} torch-{torch.__version__} '
    device = str(device).strip().lower().replace(
        'cuda:', '').replace('none', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    mps = device == 'mps'  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:

        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:  # non-cpu device requested
        # set environment variable - must be before assert is_available()
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and torch.cuda.is_available():  # prefer GPU if available

        devices = device.split(',') if device else '0'
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            # bytes to MB
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"
        arg = 'cuda:0'
    # prefer MPS if available
    elif not cpu and getattr(torch, 'has_mps', False) and torch.backends.mps.is_available():
        s += 'MPS\n'
        arg = 'mps'
    else:  # revert to CPU
        s += 'CPU\n'
        arg = 'cpu'

    if not newline:
        s = s.rstrip()

    return torch.device(arg)


def time_sync():
    """
    Synchronizes across all CUDA devices and returns the current time.

    Returns:
    - Current synchronized time.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """
    Increments a file or directory path to prevent overwriting by adding a numeric suffix.

    Parameters:
    - path: The original file or directory path.
    - exist_ok: If True, does not increment if the path already exists.
    - sep: Separator between original name and numeric suffix.
    - mkdir: If True, creates the directory if it does not exist.

    Returns:
    - A new incremented Path object.
    """
    path = Path(path)
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")
        matches = [re.search(rf"{re.escape(path.stem)}{sep}(\d+)", d)
                   for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        path = Path(f"{path}{sep}{n}{suffix}")
    dir = path if path.suffix == '' else path.parent
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)
    return path
