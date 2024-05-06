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
    sample_length = window_size - start_loc

    iter_indexes = list(range(start_loc, sample_length, seq_num))
    if sample_length % seq_num > 0:
        iter_indexes[-1] = window_size - seq_num

    return iter_indexes


def WarmUpCosineAnneal(optimizer, warm_up_iter, T_max, lr_max, lr_min):
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


def lr_schedule(optimizer, schedule_name, lr, warm_up_iter, epochs):
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
            factor=0.9,
            patience=10,
            cooldown=0,
            verbose=True,
            min_lr=5e-5
        )
    elif schedule_name == "WarmUpCosineAnneal":
        scheduler = WarmUpCosineAnneal(
            optimizer,
            warm_up_iter=warm_up_iter,
            T_max=epochs,
            lr_max=lr,
            lr_min=1e-4
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
    )

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    if args.resume and os.path.isdir(args.save_model_dir):
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
            if not os.path.isdir(args.save_model_dir):
                os.makedirs(args.save_model_dir)
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
                            args.lr, args.warm_up_iter, args.epochs)

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
               prev_decoder_state1, prev_decoder_state2, prev_decoder_state3):
    """
    Pre-warms the model states up to a specific index to stabilize predictions.

    Parameters:
    - ind: Index up to which to warm the model.
    - net: The neural network model.
    - inputs: Dictionary of input data tensors.
    - device: The device to use for computations.
    - prev_encoder_state1, prev_encoder_state2, prev_encoder_state3: Encoder states.
    - prev_decoder_state1, prev_decoder_state2, prev_decoder_state3: Decoder states.

    Returns:
    - Tuple of updated encoder and decoder states.
    """
    with torch.no_grad():
        for t in range(0, ind):
            input_t = preprocess_inputs(t, inputs, device)
            _, \
                prev_encoder_state1, prev_encoder_state2, prev_encoder_state3, \
                prev_decoder_state1, prev_decoder_state2, prev_decoder_state3 = net(input_t,
                                                                                    prev_encoder_state1, prev_encoder_state2, prev_encoder_state3,
                                                                                    prev_decoder_state1, prev_decoder_state2, prev_decoder_state3)

    return prev_encoder_state1, prev_encoder_state2, prev_encoder_state3, \
        prev_decoder_state1, prev_decoder_state2, prev_decoder_state3


def process_window(ind, args, net, inputs, device, optimizer, cls_thred=0):
    """
    Process a window of inputs through the network to produce predictions.

    Parameters:
    - ind: Starting index of the window.
    - args: Configuration parameters including sequence number.
    - net: Neural network model.
    - inputs: Input data for the model.
    - device: Computational device (CPU or GPU).
    - optimizer: Optimizer used for training.
    - cls_thred: Classification threshold for binary output.

    Returns:
    - pred: Predictions from the model for the current window.
    """
    prev_encoder_state1, prev_encoder_state2, prev_encoder_state3, \
        prev_decoder_state1, prev_decoder_state2, prev_decoder_state3 = initialize_states(
            device)

    pred = None
    optimizer.zero_grad()

    # Pre-warming before current window
    if ind > 0:
        prev_encoder_state1, prev_encoder_state2, prev_encoder_state3, \
            prev_decoder_state1, prev_decoder_state2, prev_decoder_state3 = prewarming(ind, net, inputs, device,
                                                                                       prev_encoder_state1, prev_encoder_state2, prev_encoder_state3,
                                                                                       prev_decoder_state1, prev_decoder_state2, prev_decoder_state3)

    for t in range(ind, ind + args.seq_num):
        input_t = preprocess_inputs(t, inputs, device)

        output, \
            prev_encoder_state1, prev_encoder_state2, prev_encoder_state3, \
            prev_decoder_state1, prev_decoder_state2, prev_decoder_state3 = net(input_t,
                                                                                prev_encoder_state1, prev_encoder_state2, prev_encoder_state3,
                                                                                prev_decoder_state1, prev_decoder_state2, prev_decoder_state3)

        pred = accumulate_predictions(pred, output, cls_thred)

    return pred


def model_forward(args, rank, batch, epoch, net, inputs, label, device, optimizer, lossfunction):
    """
    Sliding window-based pre-warming training

    Parameters:
    - args: Configuration and runtime arguments.
    - rank: The rank of the process. The timestamp file is created only by the process with rank 0 or when rank is -1 (not in a distributed setting).
    - batch: Current batch information, typically from a progress tracking utility like tqdm.
    - epoch: The current epoch number.
    - net: Neural network model.
    - inputs: Input data for the model.
    - label: Labels corresponding to the inputs.
    - device: Device on which the model is running.
    - optimizer: Optimizer for updating model weights.
    - lossfunction: Function to compute the loss between predictions and labels.

    Returns:
    - optimizer: Updated optimizer after processing the inputs.
    - iter_loss: Dictionary of loss statistics for the current epoch.
    """
    ind, cur_iter, iter_indexes = get_window(args, inputs, label)
    iter_loss = initialize_loss_tracking()

    # sliding window-based pre-warming training paradigm
    for ind in iter_indexes:
        iter_label = prepare_labels(label, ind, args, device)
        pred = process_window(ind, args, net, inputs, device, optimizer)
        # loss
        losses = lossfunction(pred, iter_label, epoch)
        losses["loss"].backward()
        iter_loss = update_losses(losses, iter_loss)
        # update
        optimizer.step()
        cur_iter += 1
        # print
        if rank in {-1, 0}:
            update_progress(batch, cur_iter, iter_indexes, iter_loss, epoch)

    return optimizer, iter_loss


def train_one_epoch(args, rank, net, optimizer, scheduler, lossfunction, trainLoader, device, epoch, flood_max=5000):
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
        labels = MinMaxScaler(labels, flood_max, 0)
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
    elif args.schedule_name == "WarmUpCosineAnneal":
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
                test(args, device, testLoader, epoch, upload=args.upload)

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
    trainvalFolder = Dynamic2DFlood(data_root=args.data_root, split="train")
    testFolder = Dynamic2DFlood(data_root=args.data_root, split="test")

    # Determine the number of data loader workers
    num_workers = 1
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
    effective_batch_size = 1
    # effective_batch_size = args.batch_size // torch.cuda.device_count()

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
    - args: Updated argument object including model parameters.
    """

    params = get_network_params(args.use_checkpoint)
    # Assign predefined model parameters to the args
    args.model_params = {
        "encoder_params": params[0],
        "decoder_params": params[1],
    }
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

    try:
        # Begin the training process
        train(args, device, local_rank, rank,
              trainLoader, train_sampler, testLoader)

        # Optionally execute testing if specified in arguments
        if rank in {-1, 0} and args.test:
            test(args, device, testLoader, args.epochs, upload=args.upload)

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
    exp_root = "./exp"
    local_rank, rank = initialize_environment_variables()

    timestamp_save_path = init_expr_path(exp_root, rank)

    main(exp_root, timestamp_save_path, local_rank, rank)
