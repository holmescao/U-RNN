import torch.distributed as dist
from src.lib.utils import select_device
import math
import traceback
import logging
import random
from test import test
import wandb
from src.lib.utils import exp_record
import os
import time
import datetime
import numpy as np
from tqdm import tqdm
from src.lib.model.earlystopping import SaveBestModel
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
from src.lib.dataset.Dynamic2DFlood import (
    Dynamic2DFlood,
    preprocess_inputs,
    MinMaxScaler,
)
from src.lib.model.networks.net_params import get_network_params
from src.lib.model.networks.model import ED
from src.lib.utils.net_config import load_net_config, get_input_channels
from src.lib.model.networks.losses import select_loss_function
from config import ArgumentParsers
from src.lib.utils.general import initialize_environment_variables, to_device, initialize_states


def init_torch_seeds(seed=0):
    """
    Set the seeds for torch and configure CUDA for determinism and reproducibility.

    Parameters:
    - seed: Integer seed for initializing random number generators.
    """
    torch.manual_seed(seed)
    # This is safe even if CUDA is not available
    torch.cuda.manual_seed_all(seed)

    # Ensure reproducibility
    # Might slow down, but necessary for reproducibility
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_seeds(random_seed):
    """
    Initialize random seeds across Python and PyTorch to ensure consistent behavior in runs.

    Parameters:
    - random_seed: The seed value to use for all random number generators.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    init_torch_seeds(random_seed)


def init_device(device, batch_size, local_rank):
    """
    Initializes the computational device based on the specified settings and prepares the environment for distributed training if applicable.

    Parameters:
    - device: A string specifying the requested device type ('auto', 'cpu', 'cuda', etc.). If 'auto', the device will be selected based on available hardware and the batch size.
    - batch_size: The batch size used for the training, which helps in automatic device selection.
    - local_rank: The rank of the device on the local machine, used in distributed training to set the specific GPU.

    Returns:
    - torch.device: Configured device object suitable for tensor computations.
    """
    device = select_device(
        device, batch_size)  # Assume select_device is defined to handle 'auto'

    if local_rank != -1:
        assert torch.cuda.device_count() > local_rank, "Insufficient CUDA devices for DDP command."
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl" if dist.is_nccl_available() else "gloo"
        dist.init_process_group(backend=backend)

    return device


def init_expr_path(exp_root, rank):
    """
    Creates an experiment directory based on the specified root path, and initializes a timestamp file to record the start of the experiment.

    Parameters:
    - exp_root: Path to the root directory where the experiment's data and logs should be stored.
    - rank: The rank of the process. The timestamp file is created only by the process with rank 0 or when rank is -1 (not in a distributed setting).

    Returns:
    - str: Path to the timestamp file created within the experiment directory. This file contains the start timestamp of the experiment.
    """
    os.makedirs(exp_root, exist_ok=True)
    timestamp_save_path = os.path.join(exp_root, "timestamp.txt")
    if rank in {-1, 0}:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        with open(timestamp_save_path, "w") as f:
            f.write(timestamp)
    return timestamp_save_path


def correction_seq_num(seq_num, window_size, full_window_size=False):
    """
    Correct the sequence number based on the window size and training configuration.

    Parameters:
    - seq_num: The proposed number of sequences.
    - window_size: The size of the window for processing.
    - full_window_size: Flag to indicate whether to use the full window size.

    Returns:
    - Corrected sequence number.
    """
    return window_size if full_window_size else min(seq_num, window_size)


def correction_window_size(window_size, rain_len, event_len, all_seq_train=False, train_event=False):
    """
    Adjust the window size based on the length of the event and training settings.

    Parameters:
    - window_size: Initial window size.
    - rain_len: Length of the rainfall data.
    - event_len: Length of the event data.
    - all_seq_train: Flag indicating if the entire sequence should be used for training.
    - train_event: Flag indicating if the event-based training is enabled.

    Returns:
    - Adjusted window size.
    """
    sample_length = event_len if train_event else rain_len
    return sample_length if all_seq_train else min(window_size, sample_length)


def get_start_loc(rain_len, window_size, event_len, train_event=False):
    """
    Determine the starting location for processing based on the length of the data and the window size.

    Parameters:
    - rain_len: Length of the rainfall data.
    - window_size: Size of the window for processing.
    - event_len: Length of the event data.
    - train_event: Flag indicating if the event-based training is enabled.

    Returns:
    - Starting location for processing.
    """
    sample_length = event_len if train_event else rain_len

    loc = 0
    if sample_length - window_size > 0:
        loc = np.random.randint(
            0, sample_length - window_size, size=1, dtype=int)[0]

    return loc


def split_iter_index(start_loc, seq_num, window_size):
    """
    Split the processing window into indices for iterative processing.

    Parameters:
    - start_loc: Starting location for the window.
    - seq_num: Number of sequences.
    - window_size: Size of the window.

    Returns:
    - List of indices for processing.
    """
    iter_indexes = list(range(start_loc, start_loc + window_size, seq_num))
    if window_size % seq_num > 0:
        iter_indexes[-1] = start_loc + window_size - seq_num

    return iter_indexes


def WarmUpCosineAnneal_v2(optimizer, warm_up_iter, T_max, lr_max, lr_min):
    """
    Create a scheduler with a warm-up phase followed by cosine annealing of the learning rate.

    Parameters:
    - optimizer: Optimizer to apply the scheduler to.
    - warm_up_iter: Number of iterations for the warm-up phase.
    - T_max: Maximum number of iterations.
    - lr_max: Maximum learning rate during warm-up.
    - lr_min: Minimum learning rate after warm-up.

    Returns:
    - A configured learning rate scheduler.
    """
    def lambda0(cur_iter):
        if cur_iter < warm_up_iter:
            return cur_iter / warm_up_iter * lr_max / 0.1
        else:
            return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos((cur_iter - warm_up_iter) / (T_max - warm_up_iter) * math.pi)) / 0.1
    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)

def WarmUpCosineAnneal(optimizer, warm_up_iter, T_max, lr_max, lr_min):
    """
    Create a scheduler with a warm-up phase followed by cosine annealing of the learning rate.

    Parameters:
    - optimizer: Optimizer to apply the scheduler to.
    - warm_up_iter: Number of iterations for the warm-up phase.
    - T_max: Maximum number of iterations (total epochs).
    - lr_max: Maximum learning rate (must equal the optimizer's initial lr).
    - lr_min: Minimum learning rate floor after cosine decay.

    Returns:
    - A configured LambdaLR learning rate scheduler.
    """
    min_ratio = lr_min / lr_max

    def lr_lambda(cur_iter):
        # Warm-up: multiplier ramps linearly from 0 → 1
        if cur_iter < warm_up_iter:
            return float(cur_iter) / float(max(1, warm_up_iter))
        # Guard: clamp at min_ratio once T_max is exceeded
        elif cur_iter > T_max:
            return min_ratio
        # Cosine annealing: multiplier decays from 1 → min_ratio
        else:
            progress = (cur_iter - warm_up_iter) / max(1, T_max - warm_up_iter)
            return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def lr_schedule(optimizer, schedule_name, lr, warm_up_iter, epochs, lr_min=1e-4, factor=0.9, patience=10):
    """
    Configure a learning rate scheduler based on specified settings.

    Parameters:
    - optimizer: Optimizer for which the scheduler will be set.
    - schedule_name: Type of scheduler to use ('ReduceLROnPlateau' or 'WarmUpCosineAnneal').
    - lr: Initial learning rate or maximum learning rate for warm-up.
    - warm_up_iter: Number of warm-up iterations, applicable for 'WarmUpCosineAnneal'.
    - epochs: Total number of epochs, used as T_max for 'WarmUpCosineAnneal'.

    Returns:
    - scheduler: Configured learning rate scheduler.
    """
    # Select the appropriate scheduler based on the name
    if schedule_name == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=factor,
            patience=patience,
            cooldown=0,
            verbose=True,
            min_lr=lr_min
        )
    elif schedule_name == "WarmUpCosineAnneal":
        scheduler = WarmUpCosineAnneal(
            optimizer,
            warm_up_iter=warm_up_iter,
            T_max=epochs,
            lr_max=lr,
            lr_min=lr_min
        )
    elif schedule_name == "WarmUpCosineAnneal_v2":
        scheduler = WarmUpCosineAnneal_v2(
            optimizer,
            warm_up_iter=warm_up_iter,
            T_max=epochs,
            lr_max=lr,
            lr_min=lr_min
        )
    else:
        raise ValueError(
            f"Unsupported learning rate scheduler: {schedule_name}")

    return scheduler


def load_model(args, device, local_rank, rank):
    """
    Load or initialize the model, along with its optimizer. Optionally resume from a checkpoint.

    Parameters:
    - args: Configuration parameters with model settings and paths.
    - device: The device to deploy the model on (CPU/GPU).
    - local_rank: The rank of the device on the local machine, used in distributed training to set the specific GPU.
    - rank: The rank of the process. The timestamp file is created only by the process with rank 0 or when rank is -1 (not in a distributed setting).

    Returns:
    - net: The loaded or newly initialized model.
    - optimizer: The optimizer for the model.
    - cur_epoch: The current epoch to start/resume training.
    """

    net = ED(
        args.clstm,
        args.model_params["encoder_params"],
        args.model_params["decoder_params"],
        args.cls_thred,
        args.use_checkpoint,
        input_height=args.input_height,
        input_width=args.input_width,
    )

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    if getattr(args, "pretrain_path", "") and os.path.isfile(args.pretrain_path):
        # ── Fine-tune: load weights only, skip mismatched layers, start from epoch 0 ──
        print(f"==> Fine-tuning from pretrained checkpoint: {args.pretrain_path}")
        ckpt = torch.load(args.pretrain_path, map_location="cpu")
        src_state = ckpt.get("state_dict", ckpt)
        # Strip 'module.' prefix from DDP-saved checkpoints
        src_state = {(k[7:] if k.startswith("module.") else k): v
                     for k, v in src_state.items()}
        dst_state = net.state_dict()
        matched, skipped = {}, []
        for k, v in src_state.items():
            if k in dst_state and dst_state[k].shape == v.shape:
                matched[k] = v
            else:
                skipped.append(f"{k}  src={tuple(v.shape)}  dst={tuple(dst_state[k].shape) if k in dst_state else 'missing'}")
        dst_state.update(matched)
        net.load_state_dict(dst_state)
        print(f"   Loaded {len(matched)}/{len(src_state)} layers. "
              f"Skipped {len(skipped)}: {skipped}")
        cur_epoch = 0
        if rank in {-1, 0}:
            os.makedirs(args.save_model_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.save_model_dir, "checkpoint_0_99999.pth.tar")
            torch.save({"state_dict": net.state_dict()}, checkpoint_path)
            print("saved model!")
        if rank != -1:
            dist.barrier()
    elif args.resume and os.path.isdir(args.save_model_dir):
        model_names = os.listdir(args.save_model_dir)
        model_name = sorted(
            model_names, key=lambda x: int(
                x.replace("checkpoint_", "").split("_")[0])
        )[-1]
        model_path = os.path.join(args.save_model_dir, model_name)
        # load existing model
        print("==> loading existing model")
        model_info = torch.load(model_path, map_location="cpu")
        print("loaded model:%s" % (model_path))

        state_dict = {}
        for k, v in model_info["state_dict"].items():
            if k[:7] == "module.":
                state_dict[k[7:]] = v
            else:
                state_dict[k] = v
        net.load_state_dict(state_dict)

        optimizer.load_state_dict(model_info["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        cur_epoch = model_info["epoch"] + 1

        torch.cuda.empty_cache()
    else:
        checkpoint_path = os.path.join(
            args.save_model_dir, "checkpoint_0_99999.pth.tar"
        )

        if rank in {-1, 0}:
            os.makedirs(args.save_model_dir, exist_ok=True)
            torch.save({"state_dict": net.state_dict()}, checkpoint_path)
            print("saved model!")
        if rank != -1:
            dist.barrier()

        model_info = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(model_info["state_dict"])

        cur_epoch = 0
    net = net.to(device)
    print(f"cur_epoch:{cur_epoch}")

    # Convert to DDP
    cuda = device.type != "cpu"
    if cuda and rank != -1:
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[local_rank], output_device=local_rank
        )

    return net, optimizer, cur_epoch


def set_optimizer(optimizer, args):
    """
    Setup the optimizer and the learning rate scheduler based on training arguments.

    Parameters:
    - optimizer: The optimizer for the model.
    - args: Training configuration and parameters.

    Returns:
    - scheduler: Configured learning rate scheduler.
    - lossfunction: Configured loss function based on provided settings.
    """
    # Set the learning rate scheduler
    scheduler = lr_schedule(optimizer, args.schedule_name,
                            args.lr, args.warm_up_iter, args.epochs,
                            lr_min=args.lr_min, factor=args.factor, patience=int(args.patience))

    # Select and configure the loss function
    lossfunction = select_loss_function(args.loss_name, args.reduction)

    return scheduler, lossfunction


def get_window(args, inputs, label):
    """
    Determine the starting location and sequence number for processing input data in batches.

    Parameters:
    - args: Configuration parameters for the model and training.
    - inputs: Input dataset, typically including features like rainfall data.
    - label: Ground truth labels corresponding to the inputs.

    Returns:
    - ind: Starting index for the current batch.
    - cur_iter: Current iteration number, initialized to 0.
    - iter_indexes: List of indices indicating the start of each batch segment.
    """
    rain_len = inputs["rainfall"].shape[1]
    event_len = label.shape[1]

    args.window_size = correction_window_size(
        args.window_size, rain_len, event_len, args.all_seq_train, args.train_event)
    loc = get_start_loc(rain_len, args.window_size,
                        event_len, args.train_event)
    args.seq_num = correction_seq_num(
        args.seq_num, args.window_size, args.full_window_size)

    iter_indexes = split_iter_index(loc, args.seq_num, args.window_size)
    if args.wind_random:
        random.shuffle(iter_indexes)

    return loc, 0, iter_indexes


def update_losses(losses, iter_loss):
    """
    Updates the tracked losses with new values from the current computation.

    Parameters:
    - losses: Dictionary of current loss values.
    - iter_loss: Dictionary tracking all loss values over iterations.

    Returns:
    - Updated iter_loss dictionary.
    """
    for k, v in losses.items():
        iter_loss[k].append(v.item())
    return iter_loss


def initialize_loss_tracking():
    """
    Initializes dictionaries for tracking various loss metrics.

    Returns:
    - Dictionary with lists to track each type of loss.
    """
    return {"loss": [], "loss_reg": [], "loss_reg_label": [], "loss_reg_pred": [], "loss_cls": []}


def prepare_labels(label, index, args, device):
    """
    Prepares label data for processing based on the current index and sequence number.

    Parameters:
    - label: Tensor containing label data.
    - index: Current index in the dataset.
    - args: Argument namespace containing configuration like sequence length.
    - device: The device to which tensors are to be moved.

    Returns:
    - Prepared label tensor moved to the specified device.
    """
    iter_label = label[:, index:index + args.seq_num]
    return iter_label.to(device, dtype=torch.float32)


def classify_outputs(output, threshold):
    """
    Classifies outputs based on a threshold value.

    Parameters:
    - output: Model output tensor to be classified.
    - threshold: Scalar value for thresholding the outputs.

    Returns:
    - Classified output tensor.
    """
    return torch.where(output >= threshold, 1, 0)


def update_progress(batch, cur_iter, iter_indexes, iter_loss, epoch):
    """
    Updates the training progress displayed on the progress bar.

    Parameters:
    - batch: The batch progress bar instance from tqdm or similar.
    - cur_iter: Current iteration number.
    - iter_indexes: Total number of iterations.
    - iter_loss: Dictionary containing tracked loss values.
    - epoch: Current epoch number.
    """
    batch.set_postfix({
        "iter": f"{cur_iter}/{len(iter_indexes)}",
        "total": f"{np.average(iter_loss['loss']):.9f}",
        "epoch": f"{epoch:02d}"
    })


def accumulate_predictions(pred, output, cls_thred):
    """
    Accumulates predictions by updating or initializing the predictions dictionary.

    Parameters:
    - pred: Dictionary of accumulated predictions or None if uninitialized.
    - output: Model output tensor to be accumulated.
    - cls_thred: Threshold for classification of outputs.

    Returns:
    - Updated predictions dictionary.
    """
    output_cls = classify_outputs(output, cls_thred)
    if pred is None:
        pred = {"reg": output, "cls": output_cls}
    else:
        pred["reg"] = torch.cat((pred["reg"], output), dim=1)
        pred["cls"] = torch.cat((pred["cls"], output_cls), dim=1)
    return pred


def prewarming(ind, net, inputs, device,
               prev_encoder_state1, prev_encoder_state2, prev_encoder_state3,
               prev_decoder_state1, prev_decoder_state2, prev_decoder_state3,
               historical_nums=30, rain_max=6.0, cumsum_rain_max=250.0):
    """SWP pre-warming: propagate GRU hidden states to the window start without storing gradients.

    The Sliding Window Pre-warming (SWP) paradigm (Cao et al., J. Hydrology 2025)
    addresses the difficulty of training on long sequences (≥360 timesteps) with
    limited GPU memory.  Instead of zero-initialising each training window, SWP
    first runs the model gradient-free from t=0 to t=ind-1 to obtain a physically
    meaningful initial state for the window.

    This "pre-warming" is critical for long-sequence generalisation because:
      • The GRU hidden states capture accumulated flood dynamics (ponded water,
        saturated surfaces) that cannot be recovered from instantaneous rainfall.
      • Training the model to start from its own imperfect states (rather than
        ground-truth states) improves robustness to error propagation at test time.
      • ``torch.no_grad()`` ensures the pre-warming pass consumes no GPU memory
        beyond the hidden state tensors themselves (no gradient tape stored).

    Parameters
    ----------
    ind : int
        Window start index.  Pre-warming covers timesteps [0, ind).
    net : ED
        The U-RNN model.
    inputs : dict
        Raw input tensors for the full event sequence.
    device : torch.device
        Target device.
    prev_encoder_state{1,2,3} : Tensor
        Zero-initialised encoder states (reset before each SWP window).
    prev_decoder_state{1,2,3} : Tensor
        Zero-initialised decoder states.

    Returns
    -------
    Tuple of 6 Tensors
        (enc_state1, enc_state2, enc_state3, dec_state1, dec_state2, dec_state3)
        evaluated at timestep ind-1, ready to initialise the training window.
    """
    with torch.no_grad():
        # Forward pass without gradient recording: memory cost = O(1) in states
        for t in range(0, ind):
            input_t = preprocess_inputs(t, inputs, device, nums=historical_nums,
                                        rain_max=rain_max, cumsum_rain_max=cumsum_rain_max)
            _, \
                prev_encoder_state1, prev_encoder_state2, prev_encoder_state3, \
                prev_decoder_state1, prev_decoder_state2, prev_decoder_state3 = net(input_t,
                                                                                    prev_encoder_state1, prev_encoder_state2, prev_encoder_state3,
                                                                                    prev_decoder_state1, prev_decoder_state2, prev_decoder_state3)

    return prev_encoder_state1, prev_encoder_state2, prev_encoder_state3, \
        prev_decoder_state1, prev_decoder_state2, prev_decoder_state3


def process_window(ind, args, net, inputs, device, optimizer, cls_thred=0,
                   prev_states=None):
    """Execute one SWP training window: initialise states → forward ``seq_num`` steps.

    Supports two initial-state strategies controlled by ``args.prewarming``:

    **prewarming=True (original SWP, paper-accurate):**
      1. Zero-initialise all GRU hidden states.
      2. Pre-warm gradient-free from t=0 to t=ind for a physically meaningful
         initial state (see ``prewarming`` for the rationale).
      3. Forward ``seq_num`` steps WITH gradient tracking.

    **prewarming=False (fast mode, default):**
      1. First window (prev_states=None): zero-initialise states.
      2. Subsequent windows: reuse the final hidden states from the previous
         window — valid because only one gradient update separates consecutive
         windows, so state drift is small.
      3. Forward ``seq_num`` steps WITH gradient tracking.
      Eliminates the O(ind) gradient-free warm-up pass, giving a significant
      speedup for long sequences.  Accuracy is marginally lower but acceptable.

    Parameters
    ----------
    ind : int
        Window start index (timestep within the SWP iteration schedule).
    args : argparse.Namespace
        Configuration (seq_num, historical_nums, rain/flood max values,
        prewarming flag, etc.).
    net : ED
        The U-RNN model (in train mode).
    inputs : dict
        Raw input tensors for the full event.
    device : torch.device
        Target device.
    optimizer : torch.optim.Optimizer
        Gradient will be zeroed at the start; caller calls ``optimizer.step()``.
    cls_thred : float
        Classification threshold for binarising the cls output.
    prev_states : tuple or None
        ``((enc1, enc2, enc3), (dec1, dec2, dec3))`` — detached hidden states
        from the previous window's last timestep.  ``None`` for the first
        window (zeros are used).  Only consulted when ``args.prewarming=False``.

    Returns
    -------
    pred : dict
        ``{"reg": Tensor(B, S, H, W), "cls": Tensor(B, S, H, W)}``
    final_states : tuple
        ``((enc1, enc2, enc3), (dec1, dec2, dec3))`` — detached hidden states
        at the last timestep; pass as ``prev_states`` for the next window.
    """
    prev_encoder_state1, prev_encoder_state2, prev_encoder_state3, \
        prev_decoder_state1, prev_decoder_state2, prev_decoder_state3 = initialize_states(
            device, input_height=args.input_height, input_width=args.input_width,
            net_cfg=args.net_cfg)

    optimizer.zero_grad()

    if args.prewarming:
        # ── Original SWP: gradient-free pre-warm from t=0 to t=ind ──────────
        if ind > 0:
            prev_encoder_state1, prev_encoder_state2, prev_encoder_state3, \
                prev_decoder_state1, prev_decoder_state2, prev_decoder_state3 = prewarming(
                    ind, net, inputs, device,
                    prev_encoder_state1, prev_encoder_state2, prev_encoder_state3,
                    prev_decoder_state1, prev_decoder_state2, prev_decoder_state3,
                    historical_nums=args.historical_nums,
                    rain_max=args.rain_max,
                    cumsum_rain_max=args.cumsum_rain_max)
    else:
        # ── Fast mode: reuse previous window's final states (skip pre-warm) ──
        if prev_states is not None:
            (prev_encoder_state1, prev_encoder_state2, prev_encoder_state3), \
                (prev_decoder_state1, prev_decoder_state2, prev_decoder_state3) = prev_states

    pred = None
    for t in range(ind, ind + args.seq_num):
        input_t = preprocess_inputs(t, inputs, device, nums=args.historical_nums,
                                    rain_max=args.rain_max, cumsum_rain_max=args.cumsum_rain_max)

        output, \
            prev_encoder_state1, prev_encoder_state2, prev_encoder_state3, \
            prev_decoder_state1, prev_decoder_state2, prev_decoder_state3 = net(input_t,
                                                                                prev_encoder_state1, prev_encoder_state2, prev_encoder_state3,
                                                                                prev_decoder_state1, prev_decoder_state2, prev_decoder_state3)

        pred = accumulate_predictions(pred, output, cls_thred)

    # Detach final states so they can be reused without accumulating graph refs
    final_states = (
        (prev_encoder_state1.detach(), prev_encoder_state2.detach(), prev_encoder_state3.detach()),
        (prev_decoder_state1.detach(), prev_decoder_state2.detach(), prev_decoder_state3.detach()),
    )

    return pred, final_states


def model_forward(args, rank, batch, epoch, net, inputs, label, device, optimizer, lossfunction):
    """Execute all SWP windows for one training sample (one event × location).

    Implements the Sliding Window Pre-warming (SWP) training paradigm:

    For a sequence of length ``window_size`` and backward-pass window of
    ``seq_num`` steps, the SWP schedule produces ~window_size/seq_num windows.
    Each window:
      1. Obtains initial GRU states (via pre-warm or from previous window).
      2. Runs ``seq_num`` forward steps with gradient tracking.
      3. Computes FocalBCE_and_WMSE loss and calls ``backward()``.
      4. Steps the optimizer.

    When ``args.prewarming=False`` (fast mode), the final hidden states of
    each window are carried forward to the next window, eliminating the
    O(ind) gradient-free warm-up pass.

    This gives O(seq_num) gradient memory per backward pass regardless of the
    total sequence length (vs O(window_size) for full BPTT), enabling training
    on 360-step, 500×500 sequences on a single 24 GB GPU.

    Parameters
    ----------
    args : argparse.Namespace
        Configuration: seq_num, window_size, historical_nums, prewarming, etc.
    rank : int
        DDP process rank (-1 for single-GPU).
    batch : tqdm
        Progress bar instance for per-iteration updates.
    epoch : int
        Current epoch (controls curriculum loss weighting).
    net : ED
        U-RNN model in train mode.
    inputs : dict
        Raw input tensors for the full event sequence.
    label : Tensor, shape (B, T, H, W)
        Normalised ground-truth flood depth for the full sequence.
    device : torch.device
        Target device.
    optimizer : torch.optim.Optimizer
    lossfunction : FocalBCE_and_WMSE

    Returns
    -------
    optimizer : torch.optim.Optimizer
        (Unchanged reference; kept for API consistency.)
    iter_loss : dict
        Per-key lists of loss values across all windows in this sample.
    """
    ind, cur_iter, iter_indexes = get_window(args, inputs, label)
    iter_loss = initialize_loss_tracking()

    prev_states = None  # carries final hidden states between windows (fast mode)

    # sliding window-based training (with or without pre-warming)
    for ind in iter_indexes:
        iter_label = prepare_labels(label, ind, args, device)
        pred, prev_states = process_window(
            ind, args, net, inputs, device, optimizer,
            prev_states=prev_states)
        # loss
        losses = lossfunction(pred, iter_label, epoch)
        losses["loss"].backward()
        iter_loss = update_losses(losses, iter_loss)
        # update
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        optimizer.step()
        cur_iter += 1
        # print
        if rank in {-1, 0}:
            update_progress(batch, cur_iter, iter_indexes, iter_loss, epoch)

    return optimizer, iter_loss


def train_one_epoch(args, rank, net, optimizer, scheduler, lossfunction, trainLoader, device, epoch):
    """
    Conduct training over one epoch of data.

    Parameters:
    - args: Configuration parameters for the training process.
    - rank: The rank of the process. The timestamp file is created only by the process with rank 0 or when rank is -1 (not in a distributed setting).
    - net: The neural network model to train.
    - optimizer: Optimization algorithm.
    - scheduler: Learning rate scheduler.
    - lossfunction: Loss function used for training.
    - trainLoader: DataLoader for the training dataset.
    - device: Device on which to perform computations.
    - epoch: Current epoch number.
    - flood_max: Maximum flood value for scaling purposes.

    Returns:
    - optimizer: Updated optimizer after an epoch of training.
    - upload_info: Dictionary containing training metrics.
    """
    net.train()
    pbar = tqdm(enumerate(trainLoader), total=len(trainLoader),
                leave=False) if rank in {-1, 0} else enumerate(trainLoader)

    for step, (inputs, labels, _) in pbar:
        inputs = to_device(inputs, device)
        labels = MinMaxScaler(labels, args.flood_max, 0)
        optimizer, iter_loss = model_forward(
            args, rank, pbar, epoch, net, inputs, labels, device, optimizer, lossfunction)

    # Synchronize all processes in case of distributed training
    if rank != -1 and device.type != "cpu":
        torch.cuda.synchronize(device)

    # Aggregate learning information
    upload_info = {"lr": optimizer.param_groups[0]["lr"]}
    upload_info.update({k: np.average(v) for k, v in iter_loss.items()})

    scheduler = scheduler_update(args, scheduler, upload_info)

    return optimizer, upload_info


def print_epoch_train_info(args, epoch, upload_info, epoch_start_time, epoch_end_time):
    """
    Log and print training information for the epoch.

    Parameters:
    - args: Configuration parameters containing flags and settings.
    - epoch: The current epoch number.
    - upload_info: Dictionary containing training metrics.
    - epoch_start_time: Start time of the epoch.
    - epoch_end_time: End time of the epoch.
    """
    if args.upload:
        wandb.log(upload_info)

    epoch_len = len(str(args.epochs))
    formatted_info = " | ".join(
        f"{key}:{value:.9f}" for key, value in upload_info.items())
    log_info = f"[{epoch+1:>{epoch_len}}/{args.epochs:>{epoch_len}}] {formatted_info} | time:{epoch_end_time - epoch_start_time:.2f} sec"

    logging.info(log_info)
    print(log_info)


def scheduler_update(args, scheduler, upload_info):
    """
    Update the learning rate scheduler based on the training progress.

    Parameters:
    - args: Configuration parameters.
    - scheduler: Learning rate scheduler.
    - upload_info: Dictionary containing metrics from the training process.

    Returns:
    - scheduler: Updated scheduler.
    """
    if args.schedule_name == "ReduceLROnPlateau":
        scheduler.step(upload_info["loss"])
    elif args.schedule_name in ("WarmUpCosineAnneal", "WarmUpCosineAnneal_v2"):
        scheduler.step()

    return scheduler


def train(args, device, local_rank, rank, trainLoader, train_sampler, testLoader):
    """
    Main function to execute the training loop for a deep learning model.

    Parameters:
    - args: Namespace containing all configuration parameters for training.
    - device: The computational device (GPU/CPU) to use for training.
    - local_rank: The rank of the device on the local machine, used in distributed training to set the specific GPU.
    - rank: The rank of the process. The timestamp file is created only by the process with rank 0 or when rank is -1 (not in a distributed setting).
    - trainLoader: DataLoader for the training dataset.
    - train_sampler: Sampler for distributing the data across multiple processes.
    - testLoader: DataLoader for the testing dataset.
    """
    # Initialize random seeds for reproducibility
    init_seeds(args.random_seed)

    # Load model and optimizer, and initialize the current epoch counter
    net, optimizer, cur_epoch = load_model(args, device, local_rank, rank)

    # Set up the optimizer and loss function
    scheduler, lossfunction = set_optimizer(optimizer, args)

    # Initialize best model saving mechanism if running in the main process
    save_best_model = SaveBestModel(verbose=True) if rank in {-1, 0} else None

    # Begin training loop
    for epoch in range(cur_epoch, args.epochs):
        # Set epoch for distributed training
        if rank != -1:
            trainLoader.sampler.set_epoch(epoch)

        # Start timer for the epoch
        epoch_start_time = time.time()

        # Run training for one epoch and capture optimization metrics
        optimizer, upload_info = train_one_epoch(
            args=args,
            rank=rank,
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            lossfunction=lossfunction,
            trainLoader=trainLoader,
            device=device,
            epoch=epoch
        )

        # End timer for the epoch
        epoch_end_time = time.time()

        # In the main process, perform testing and log training stats
        if rank in {-1, 0}:
            if (epoch + 1) % 50 == 0 and args.test:
                test(args, device, testLoader, epoch)

            # Print and log training and validation statistics
            print_epoch_train_info(
                args, epoch, upload_info, epoch_start_time, epoch_end_time)

            # Save the model if it's the best one seen so far
            save_best_model(upload_info["loss"], net,
                            optimizer, epoch, args.save_model_dir)


def load_dataset(args, local_rank, rank):
    """
    Load training and testing datasets and prepare data loaders for distributed training if required.

    Parameters:
    - args: Configuration namespace containing all required parameters.
    - local_rank: The rank of the device on the local machine, used in distributed training to set the specific GPU.
    - rank: The rank of the process. The timestamp file is created only by the process with rank 0 or when rank is -1 (not in a distributed setting).

    Returns:
    - trainLoader: DataLoader for training data.
    - train_sampler: Sampler for distributing training data across multiple processes.
    - testLoader: DataLoader for testing data.
    """
    # Initialize training and testing datasets
    trainvalFolder = Dynamic2DFlood(
        data_root=args.data_root,
        split="train",
        event_list_file=getattr(args, "train_list_file", None),
        duration=args.duration,
        location=getattr(args, "location", ""),
    )
    testFolder = Dynamic2DFlood(
        data_root=args.data_root,
        split="test",
        event_list_file=getattr(args, "test_list_file", None),
        duration=args.duration,
        location=getattr(args, "location", ""),
    )

    # Determine the number of data loader workers
    num_workers = args.num_workers
    if rank in {-1, 0}:
        print(f"Using {num_workers} dataloader workers every process")

    # Set up distributed training samplers if applicable
    train_sampler = None
    if local_rank != -1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            trainvalFolder, shuffle=True)

    test_sampler = None
    if local_rank != -1:
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            testFolder, shuffle=False)

    # Define batch size adjusted for number of CUDA devices
    effective_batch_size = args.batch_size

    # Configure data loaders for training and testing datasets
    trainLoader = torch.utils.data.DataLoader(
        trainvalFolder,
        batch_size=effective_batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True
    )

    testLoader = torch.utils.data.DataLoader(
        testFolder,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=test_sampler,
        pin_memory=True,
        drop_last=False
    )

    return trainLoader, train_sampler, testLoader


def load_model_params(args):
    """
    Load model parameters for the encoder and decoder from predefined settings.

    Parameters:
    - args: Configuration namespace containing all required parameters.

    Returns:
    - args: Updated argument object including model parameters and net_cfg.
    """
    net_cfg = load_net_config(args.net_config)
    input_channels = get_input_channels(net_cfg, args.historical_nums)
    params = get_network_params(args.use_checkpoint, args.input_height, args.input_width,
                                input_channels=input_channels, net_cfg=net_cfg)
    # Assign predefined model parameters to the args
    args.model_params = {
        "encoder_params": params[0],
        "decoder_params": params[1],
    }
    args.net_cfg = net_cfg
    return args


def RecordsPrepare(args, rank):
    """
    Prepare logging and experiment recording facilities.

    Parameters:
    - args: Configuration namespace containing all required parameters.
    - rank: The rank of the process. The timestamp file is created only by the process with rank 0 or when rank is -1 (not in a distributed setting).

    Returns:
    - run: Initialized Weights & Biases (wandb) run object if enabled.
    - log_filename: Path to the log file.
    """
    run = None
    log_filename = ""
    if rank in {-1, 0}:  # Only execute in the main process
        expr_dir = os.path.join(args.exp_root, args.timestamp)
        os.makedirs(expr_dir, exist_ok=True)
        if args.upload:
            run = exp_record.wandb_init(
                job_type="Training",
                id=args.timestamp,
                name=args.timestamp,
                config=args,
                project=args.exp_name,
            )
        log_filename = exp_record.init_logging(expr_dir)

    return run, log_filename


def update_opts(args, timestamp_save_path):
    """
    Update configuration options based on experiment timestamp.

    Parameters:
    - args: Configuration namespace containing all required parameters.
    - timestamp_save_path: Path to the file where the experiment's timestamp is saved.

    Returns:
    - args: Updated argument object with paths containing the actual timestamp.
    """
    with open(timestamp_save_path, "r") as file:
        timestamp = file.readline().strip()
    print("Experiment start! Now: ", timestamp)

    # Replace placeholder with actual timestamp in all relevant paths
    attributes_to_update = ['timestamp', 'save_loss_dir', 'save_model_dir',
                            'save_dir_flood_maps', 'save_fig_dir', 'exp_dir']
    for attr in attributes_to_update:
        setattr(args, attr, getattr(args, attr).replace("timestamp", timestamp))

    return args


def main(exp_root, timestamp_save_path, local_rank, rank):
    """
    Main execution function to setup and run the training and testing processes.

    Parameters:
    - exp_root: The root directory for the experiment.
    - timestamp_save_path: Path to the file where the experiment's timestamp is saved.
    - local_rank: The rank of the device on the local machine, used in distributed training to set the specific GPU.
    - rank: The rank of the process. The timestamp file is created only by the process with rank 0 or when rank is -1 (not in a distributed setting).
    """
    # Parse command line arguments or configuration files
    args = ArgumentParsers(exp_root)

    # Initialize the computational device for the experiment
    device = init_device(args.device, args.batch_size, local_rank)

    # Update runtime options based on the timestamp and potentially other configurations
    args = update_opts(args, timestamp_save_path)

    # Load datasets and model parameters
    trainLoader, train_sampler, testLoader = load_dataset(
        args, local_rank, rank)
    args = load_model_params(args)

    # Prepare for experiment record-keeping (logs, visualization)
    run, log_filename = RecordsPrepare(args, rank)

    # Print all resolved configuration parameters for traceability
    if rank in {-1, 0}:
        print("\n" + "="*60)
        print("EXPERIMENT CONFIG")
        print("="*60)
        for k, v in sorted(vars(args).items()):
            if not k.startswith("_") and not callable(v):
                print(f"  {k}: {v}")
        print("="*60 + "\n")

    try:
        # Begin the training process
        train(args, device, local_rank, rank,
              trainLoader, train_sampler, testLoader)

        # Optionally execute testing if specified in arguments
        if rank in {-1, 0} and args.test:
            test(args, device, testLoader, args.epochs)

        # If configured, finalize the experiment record-keeping
        if rank in {-1, 0} and args.upload:
            run.finish()

    except Exception as e:
        # Handle exceptions by logging them to a file and standard error
        if rank in {-1, 0}:  # Only perform in the main process
            with open(log_filename, "a+") as log_file:
                traceback.print_exc(file=log_file)
            traceback.print_exc()


if __name__ == "__main__":
    exp_root = "../exp"
    local_rank, rank = initialize_environment_variables()

    timestamp_save_path = init_expr_path(exp_root, rank)

    main(exp_root, timestamp_save_path, local_rank, rank)
