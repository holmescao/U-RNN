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


@contextmanager    # 这个是上下文管理器
def torch_distributed_zero_first(local_rank: int):
    """train.py
    用于处理模型进行分布式训练时同步问题
    基于torch.distributed.barrier()函数的上下文管理器，为了完成数据的正常同步操作（yolov5中拥有大量的多线程并行操作）
    Decorator to make all processes in distributed training wait for each local_master to do something.
    :params local_rank: 代表当前进程号  0代表主进程  1、2、3代表子进程
    """
    if local_rank not in [-1, 0]:
        # 如果执行create_dataloader()函数的进程不是主进程，即rank不等于0或者-1，
        # 上下文管理器会执行相应的torch.distributed.barrier()，设置一个阻塞栅栏，
        # 让此进程处于等待状态，等待所有进程到达栅栏处（包括主进程数据处理完毕）；
        dist.barrier()
    yield  # yield语句 中断后执行上下文代码，然后返回到此处继续往下执行
    if local_rank == 0:
        # 如果执行create_dataloader()函数的进程是主进程，其会直接去读取数据并处理，
        # 然后其处理结束之后会接着遇到torch.distributed.barrier()，
        # 此时，所有进程都到达了当前的栅栏处，这样所有进程就达到了同步，并同时得到释放。
        dist.barrier()


def device_count():
    # Returns number of CUDA devices available. Safe version of torch.cuda.device_count(). Supports Linux and Windows
    assert platform.system() in (
        'Linux', 'Windows'), 'device_count() only supported on Linux or Windows'
    try:
        cmd = 'nvidia-smi -L | wc -l' if platform.system() == 'Linux' else 'nvidia-smi -L | find /c /v ""'  # Windows
        return int(subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1])
    except Exception:
        return 0


def select_device(device='', batch_size=0, newline=True):
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    s = f'Python-{platform.python_version()} torch-{torch.__version__} '
    device = str(device).strip().lower().replace(
        'cuda:', '').replace('none', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    mps = device == 'mps'  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        # force torch.cuda.is_available() = False
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:  # non-cpu device requested
        # set environment variable - must be before assert is_available()
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and torch.cuda.is_available():  # prefer GPU if available
        # range(torch.cuda.device_count())  # i.e. 0,1,6,7
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
    # LOGGER.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    return torch.device(arg)


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """根据文件夹中已有的文件名，自动获得新路径或文件名"""
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        # 获取文件后缀
        suffix = path.suffix
        # 去掉后缀的path
        path = path.with_suffix('')
        # 获取所有以{path}{sep}开头的文件
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # 在dirs中找到以数字结尾的文件
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        # 获取dirs文件结尾的数字
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        # 最大的数字+1
        n = max(i) + 1 if i else 2  # increment number
        # 设置新文件的文件名
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    # 获取文件路径并创建
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path
