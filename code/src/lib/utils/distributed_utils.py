import os
import torch
import torch.distributed as dist


def init_distributed_mode(args):
    """
    Initializes distributed mode for PyTorch if possible, sets device, and creates process group.

    Parameters:
    - args: A namespace object that must include dist_url, and will be populated with
            rank, world_size, gpu, distributed, and dist_backend attributes.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode.')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'  # NCCL is recommended for Nvidia GPUs.

    print(
        f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()


def cleanup():
    """
    Destroys the distributed process group, cleaning up distributed environment.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def is_dist_avail_and_initialized():
    """
    Checks if the distributed environment is available and initialized.

    Returns:
    - bool: True if the distributed environment is both available and initialized.
    """
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    """
    Returns the world size in the distributed environment.

    Returns:
    - int: The world size, or 1 if distributed processing is not available or initialized.
    """
    if is_dist_avail_and_initialized():
        return dist.get_world_size()
    return 1


def get_rank():
    """
    Returns the current process rank within the distributed environment.

    Returns:
    - int: The rank, or 0 if distributed processing is not available or initialized.
    """
    if is_dist_avail_and_initialized():
        return dist.get_rank()
    return 0


def is_main_process():
    """
    Determines if the current process is the main process (rank 0).

    Returns:
    - bool: True if the current process is the main process, otherwise False.
    """
    return get_rank() == 0


def reduce_value(value, average=True):
    """
    Reduces a value across all processes in the distributed environment.

    Parameters:
    - value: A tensor to be reduced across all processes.
    - average: If True, averages the value, otherwise, sums up.

    Returns:
    - The reduced (and possibly averaged) value.
    """
    world_size = get_world_size()
    if world_size < 2:
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size
        return value
