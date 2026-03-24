from matplotlib.colors import ListedColormap, Normalize
import matplotlib.pyplot as plt
import torch.distributed as dist
from src.lib.utils import select_device
import wandb
from config import ArgumentParsers
import time
import numpy as np
from tqdm import tqdm
import torch
from src.lib.model.networks.net_params import get_network_params
from src.lib.model.networks.model import ED
from src.lib.utils.net_config import load_net_config, get_input_channels, get_state_shapes
import os
import pandas as pd
from src.lib.dataset.Dynamic2DFlood import (Dynamic2DFlood, preprocess_inputs,
                                            MinMaxScaler, r_MinMaxScaler)
from src.lib.utils.general import initialize_environment_variables, to_device, initialize_states


def initialize_environment():
    from src.lib.utils.common_runtime import cuda_call
    from cuda import cuda, cudart
    import pycuda.driver as pydrcuda
    import tensorrt as trt
    import pycuda.autoinit  # This automatically initializes CUDA and creates a context
    import src.lib.utils.common as common

    return cuda, cudart, pydrcuda, trt, common, cuda_call


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


def _do_inference_base(inputs, outputs, stream, execute_async_func):
    """
    Handles the low-level inference execution on GPU using asynchronous CUDA calls.

    Parameters:
    - inputs: List of input data objects with attributes 'device', 'host', and 'nbytes'.
    - outputs: List of output data objects with attribute 'device'.
    - stream: CUDA stream for asynchronous execution.
    - execute_async_func: Function to trigger asynchronous execution.

    Returns:
    - Tuple of list of device pointers for outputs and the output objects themselves.
    """
    # Transfer input data to the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    for inp in inputs:
        cuda_call(cudart.cudaMemcpyAsync(
            inp.device, inp.host, inp.nbytes, kind, stream))

    # Run inference.
    execute_async_func()

    # Synchronize the stream to ensure all computations are done before proceeding.
    cuda_call(cudart.cudaStreamSynchronize(stream))

    # Return device pointers of outputs directly along with the outputs themselves.
    return ([out.device for out in outputs], outputs)


def do_inference(context, engine, bindings, inputs, outputs, stream):
    """
    Configures and runs inference in an asynchronous manner, setting up necessary
    bindings and utilizing an execution callback.

    Parameters:
    - context: Inference context for the neural network.
    - engine: The engine object that handles the neural network.
    - bindings: Tensor addresses in GPU memory.
    - inputs: List of input data objects.
    - outputs: List of output data objects.
    - stream: CUDA stream for handling asynchronous operations.

    Returns:
    - Result of the base inference function, including device pointers to outputs.
    """
    def execute_async_func():
        context.execute_async_v3(stream_handle=stream)

    # Setup tensor addresses in the context according to the engine specifications.
    num_io = engine.num_io_tensors
    for i in range(num_io):
        tensor_name = engine.get_tensor_name(i)
        context.set_tensor_address(tensor_name, bindings[i])

    return _do_inference_base(inputs, outputs, stream, execute_async_func)


class TensorRTInference:
    """
    Manages the TensorRT inference process including initialization, input preparation,
    and execution of inference steps.
    """

    def __init__(self, engine_path, T, input_height=500, input_width=500, net_cfg=None):
        """
        Initializes the inference engine, creates the execution context, and allocates
        necessary buffers and streams.

        Parameters:
        - engine_path: Path to the serialized TensorRT engine.
        - T: Number of time steps to process in the inference loop.
        - input_height: Spatial height of the model input grid.
        - input_width: Spatial width of the model input grid.
        - net_cfg: Loaded network architecture config dict (from network.yaml).
        """
        self.T = T
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(
            self.engine)

        self.H, self.W = input_height, input_width
        self.net_cfg = net_cfg
        self.output_size_in_bytes = self.H * \
            self.W * np.dtype(np.float32).itemsize

    @staticmethod
    def load_engine(engine_path):
        """
        Loads a serialized TensorRT engine.

        Parameters:
        - engine_path: Path to the serialized TensorRT engine file.

        Returns:
        - The deserialized engine.
        """
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def prepare_initial_data(self, input_ts, start_index=0, batch_size=100):
        """
        Prepares and uploads data for a batch of timesteps to the GPU.

        Parameters:
        - input_ts: Initial input data for T timesteps.
        - start_index: The starting index for the batch.
        - batch_size: The number of timesteps in each batch.
        """
        batch_end = min(start_index + batch_size, len(input_ts))
        actual_batch_size = batch_end - start_index
        single_input_size = input_ts[0].nbytes
        self.device_input_ts = pydrcuda.mem_alloc(
            actual_batch_size * single_input_size)

        # Upload batch input data to GPU
        for i in range(start_index, batch_end):
            offset = (i - start_index) * single_input_size
            device_data_ptr = int(self.device_input_ts) + offset
            pydrcuda.memcpy_htod(device_data_ptr, input_ts[i])

    def prepare_initial_states(self):
        """
        Initializes and allocates memory for state variables on the GPU. This function
        sets up initial states needed for both encoder and decoder components of the model,
        with predefined shapes for each state tensor.

        The states are initialized to zero and uploaded to the GPU to prepare for inference
        or further processing. Each state variable's memory is managed through PyCUDA,
        ensuring that the necessary GPU resources are allocated.

        Attributes:
            device_states (list): A list that stores device pointers to the state tensors
                                allocated on the GPU. Each element in the list is a
                                PyCUDA device allocation that points to the initialized state.
        """
        # Initialize and allocate state variables on GPU
        state_shapes = get_state_shapes(self.net_cfg, self.H, self.W) \
            if self.net_cfg is not None else [
                (1, 64, self.H, self.W), (1, 96, self.H//2, self.W//2), (1, 96, self.H//4, self.W//4),
                (1, 96, self.H//4, self.W//4), (1, 96, self.H//2, self.W//2), (1, 64, self.H, self.W)
            ]
        self.device_states = []
        for shape in state_shapes:
            state = np.zeros(shape, dtype=np.float32)
            device_state = pydrcuda.mem_alloc(state.nbytes)
            pydrcuda.memcpy_htod(device_state, state)
            self.device_states.append(device_state)

    def inferStep(self, input_ts, start_index=0, batch_size=100):
        """
        Perform inference steps for a batch of time instances.

        Parameters:
        - input_ts: List of new inputs for the T timesteps.
        - start_index: The starting index for the batch.
        - batch_size: The number of timesteps in each batch.

        Returns:
        - List of device pointers to the output buffers for each timestep in the batch.
        """
        batch_end = min(start_index + batch_size, len(input_ts))
        actual_batch_size = batch_end - start_index
        single_input_size = input_ts[0].nbytes
        output_buffers = [pydrcuda.mem_alloc(
            self.output_size_in_bytes) for _ in range(actual_batch_size)]

        for t in range(start_index, batch_end):
            self.bindings[0] = int(self.device_input_ts) + \
                (t - start_index) * single_input_size

            output_data, outputs_origin = do_inference(
                self.context, self.engine, self.bindings, self.inputs, self.outputs, self.stream)

            # Store output in separate GPU buffers to avoid overwriting
            pydrcuda.memcpy_dtod(
                output_buffers[t - start_index], outputs_origin[0].device, self.output_size_in_bytes)

            self.bindings[1:7] = output_data[1:7]

        if hasattr(self, 'device_input_ts'):
            self.device_input_ts.free()

        return output_buffers

    def output_spec(self):
        """
        Retrieves specifications for the output tensors of the network.

        Returns:
        - List of tuples containing the shape and datatype of each output tensor.
        """
        return [(o['shape'], o['dtype']) for o in self.outputs]

    def cuda_to_npy(self, output_data):
        """
        Copies output data from CUDA buffers to host memory and converts to numpy arrays.

        Parameters:
        - output_data: List of CUDA device pointers to output data.

        Returns:
        - Numpy array of output data reshaped to the expected dimensions.
        """
        host_output_data = [np.empty((self.H, self.W), dtype=np.float32)
                            for _ in range(len(output_data))]
        for i, buffer in enumerate(output_data):
            pydrcuda.memcpy_dtoh(host_output_data[i], buffer)

        return np.array(host_output_data).reshape(-1, self.H, self.W)


def Inference_with_TensorRT(inference, inputs, device, batch_size=350,
                            historical_nums=30, rain_max=6.0, cumsum_rain_max=250.0):
    """
    Perform inference using a TensorRT model for a given set of inputs.

    Parameters:
    - inference: TensorRT engine
    - inputs: Dictionary of input tensors.
    - device: The device on which preprocessing should be executed.

    Returns:
    - Numpy array of the output data after inference.
    """
    # Determine the number of frames based on the rainfall input tensor
    Frames = inputs['rainfall'].shape[1]

    # Preprocess each frame input and convert them to NumPy arrays for TensorRT processing
    full_sequence_inputs = [preprocess_inputs(
        f, inputs, device,
        nums=historical_nums,
        rain_max=rain_max,
        cumsum_rain_max=cumsum_rain_max
    ).cpu().numpy() for f in range(Frames)]

    # Initialize the TensorRT inference engine
    inference.prepare_initial_states()
    inference.prepare_initial_data(
        full_sequence_inputs, 0, batch_size)

    output_data_batch = []
    # Execute inference and measure the duration
    test_start_time = time.time()
    for start_index in range(0, Frames, batch_size):
        if start_index > 0:
            inference.prepare_initial_data(
                full_sequence_inputs, start_index, batch_size)
        # infer
        output_data_batch.append(
            inference.inferStep(
                full_sequence_inputs, start_index, batch_size))

    test_duration = time.time() - test_start_time

    # Log the duration of the test
    print(
        f"Test completed in {test_duration:.2f} sec for {Frames} steps, {test_duration / Frames:.2f} sec/step")

    # Convert CUDA device output data to NumPy array
    output_data = []
    for output in output_data_batch:
        output_data.append(inference.cuda_to_npy(output))
    output_data = np.concatenate(output_data, axis=0)

    return output_data


def Inference(net, inputs, device, historical_nums=30, rain_max=6.0, cumsum_rain_max=250.0,
              input_height=500, input_width=500, net_cfg=None):
    """
    Performs inference over all time steps in the input data using the specified network.

    Parameters:
    - net: The neural network model to use for inference.
    - inputs: A dictionary containing the input data with key 'rainfall' pointing to a tensor of shape (T, H, W).
    - historical_nums: Number of past rainfall timesteps used as input features.
    - rain_max: Max rainfall intensity for normalization.
    - cumsum_rain_max: Max cumulative rainfall for normalization.
    - input_height: Spatial height of the model input grid.
    - input_width: Spatial width of the model input grid.
    - net_cfg: Loaded network architecture config dict (from network.yaml).

    Returns:
    - output_data: List of outputs from the network for each time step.
    """

    with torch.no_grad():
        net.eval()
        Frames = inputs['rainfall'].shape[1]

        prev_encoder_state1, prev_encoder_state2, prev_encoder_state3, \
            prev_decoder_state1, prev_decoder_state2, prev_decoder_state3 = initialize_states(
                device, input_height=input_height, input_width=input_width, net_cfg=net_cfg)

        output_data = []
        test_start_time = time.time()
        t = 0
        for t in range(Frames):
            input_t = preprocess_inputs(t, inputs, device,
                                        nums=historical_nums,
                                        rain_max=rain_max,
                                        cumsum_rain_max=cumsum_rain_max)
            output, \
                prev_encoder_state1, prev_encoder_state2, prev_encoder_state3, \
                prev_decoder_state1, prev_decoder_state2, prev_decoder_state3 = net(input_t,
                                                                                    prev_encoder_state1, prev_encoder_state2, prev_encoder_state3,
                                                                                    prev_decoder_state1, prev_decoder_state2, prev_decoder_state3)

            output_data.append(output)
        # Calculate and print test duration
        test_duration = time.time() - test_start_time
        print(
            f"Test completed in {test_duration:.2f} sec for {Frames} steps, {test_duration / Frames:.2f} sec/step")

        # post-process
        output_data = [output.cpu().numpy()[0, 0] for output in output_data]
        output_data = np.array(output_data)

    return output_data


def load_net(args, device):
    # Initialize model
    net = ED(args.clstm,
             args.model_params["encoder_params"],
             args.model_params["decoder_params"],
             args.cls_thred,
             args.use_checkpoint,
             input_height=args.input_height,
             input_width=args.input_width)

    # Load the model
    print("loading model...")
    model_names = os.listdir(args.save_model_dir)
    model_name = sorted(
        model_names,
        key=lambda x: int(x.replace("checkpoint_", "").split("_")[0]))[-1]
    model_path = os.path.join(args.save_model_dir, model_name)
    model_info = torch.load(model_path, map_location=torch.device('cpu'))
    print("loaded model:%s" % (model_path))
    state_dict = {}
    for k, v in model_info["state_dict"].items():
        if k[:7] == "module.":
            state_dict[k[7:]] = v
        else:
            state_dict[k] = v
    net.load_state_dict(state_dict)
    net = net.to(device)

    return net


def test(args, device, testLoader, cur_epoch=99999):
    """
    Runs the test phase for the model on provided data.

    Parameters:
    - args: Configuration and runtime arguments.
    - device: Device on which to run the test.
    - testLoader: DataLoader providing test datasets.
    - cur_epoch: Current epoch of model training.
    """
    # Set random seeds for reproducibility
    random_seed = args.random_seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(random_seed)
    else:
        torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.benchmark = False

    if not args.trt:
        net = load_net(args, device)
    else:
        trt_net = TensorRTInference(os.path.join(
            args.trt_model_dir, "URNN.trt"), T=args.window_size,
            input_height=args.input_height, input_width=args.input_width,
            net_cfg=args.net_cfg)

    save_epoch = os.path.join(args.save_fig_dir, "epoch@"+str(cur_epoch))
    if not os.path.exists(save_epoch):
        os.makedirs(save_epoch, exist_ok=True)

    # ── Evaluation loop ─────────────────────────────────────────────────────
    all_metrics = {}
    print("Start testing...")
    batch = tqdm(testLoader, leave=False, total=len(testLoader))
    for _, (inputs, target_vars, event_dir) in enumerate(batch):
        print("event_dir:", event_dir)
        inputs = to_device(inputs, device)

        if not args.trt:
            print("Infer with PyTorch...")
            output_data = Inference(net, inputs, device,
                                    historical_nums=args.historical_nums,
                                    rain_max=args.rain_max,
                                    cumsum_rain_max=args.cumsum_rain_max,
                                    input_height=args.input_height,
                                    input_width=args.input_width,
                                    net_cfg=args.net_cfg)
        else:
            print("Infer with TensorRT...")
            output_data = Inference_with_TensorRT(trt_net, inputs, device,
                                                  historical_nums=args.historical_nums,
                                                  rain_max=args.rain_max,
                                                  cumsum_rain_max=args.cumsum_rain_max)

        # De-normalise predictions to mm
        output_mm = r_MinMaxScaler(output_data, max=args.flood_max, min=0)

        # ── Compute metrics ─────────────────────────────────────────────────
        # target_vars shape: (B, T, H, W) in mm; B=1 from DataLoader
        gt_mm = target_vars[0].numpy()   # (T, H, W) in mm
        event_name = os.path.basename(event_dir[0])
        metrics = compute_metrics(output_mm, gt_mm, flood_thres=args.flood_thres)
        all_metrics[event_name] = metrics
        print(f"  {event_name}: R2={metrics['R2']:.4f}  RMSE={metrics['RMSE']:.4f}m  "
              f"MAE={metrics['MAE']:.4f}m  CSI={metrics['CSI']:.4f}")

        post_process(output_mm, gt_mm, save_epoch, event_dir, viz_time_points=args.viz_time_points)

    # ── Aggregate and save metrics ───────────────────────────────────────────
    if all_metrics:
        save_metrics(all_metrics, args.save_metric_dir, cur_epoch)


def post_process(output_mm, gt_mm, save_epoch, event_dir, viz_time_points=None):
    """
    Applies post-processing to model output (already de-normalised to mm),
    and visualises a 3-row comparison (Reference / U-RNN / Error).

    Parameters:
    - output_mm: numpy array (T, H, W) in mm — model predictions, de-normalised.
    - gt_mm: numpy array (T, H, W) in mm — ground-truth flood depths.
    - save_epoch: Base directory where the output visualisations will be saved.
    - event_dir: Directory path containing event data, used to derive the event name.
    - viz_time_points: List of time step indices for visualisation.
    """
    if viz_time_points is None:
        viz_time_points = [0, 119, 239, 359]

    # Extract the event name from the path
    event_name = os.path.basename(event_dir[0])
    save_epoch_sample = os.path.join(save_epoch, event_name)

    # Ensure the target directory exists, creating it if necessary
    os.makedirs(save_epoch_sample, exist_ok=True)

    plot_spatial_distributions(
        output_mm.copy(), gt_mm.copy(), time_points=viz_time_points, save_path=save_epoch_sample)


def plot_spatial_distributions(pred_mm, gt_mm, time_points=[0, 50, 99], save_path="./"):
    """
    Plot 3-row comparison at specified time points:
      Row 0 — Reference (ground truth)
      Row 1 — U-RNN (predictions)
      Row 2 — Absolute error  |pred - gt|

    Rows 0–1 share a fixed 0–2 m colorbar (jet, zero = white).
    Row 2 uses a fixed 0–0.3 m colorbar (Reds, zero = white).

    Parameters:
    - pred_mm : np.ndarray (T, H, W), model predictions in mm.
    - gt_mm   : np.ndarray (T, H, W), ground-truth flood depths in mm.
    - time_points : list of int, timestep indices to visualise.
    - save_path : str, directory for the output image.
    """
    num_cols = len(time_points)
    fig, axes = plt.subplots(3, num_cols, figsize=(5 * num_cols + 2.0, 13))
    if num_cols == 1:
        axes = axes.reshape(3, 1)

    fontsize = 15
    tick_fontsize = fontsize - 2

    # ── Convert mm → m ──────────────────────────────────────────────────────
    pred_m  = pred_mm  / 1000.0
    gt_m    = gt_mm    / 1000.0
    error_m = np.abs(pred_m - gt_m)

    # ── Colormaps ────────────────────────────────────────────────────────────
    depth_cmap_arr = plt.cm.jet(np.linspace(0, 1, 256))
    depth_cmap_arr[0] = (1, 1, 1, 1)          # zero → white
    depth_cmap = ListedColormap(depth_cmap_arr)
    depth_cmap.set_bad('white')

    error_cmap_arr = plt.cm.Reds(np.linspace(0, 1, 256))
    error_cmap_arr[0] = (1, 1, 1, 1)
    error_cmap = ListedColormap(error_cmap_arr)
    error_cmap.set_bad('white')

    depth_norm = Normalize(vmin=0.0, vmax=2.0)
    error_norm = Normalize(vmin=0.0, vmax=0.3)

    row_labels = ['Reference', 'U-RNN', 'Error']
    datasets   = [gt_m,       pred_m,  error_m]
    cmaps      = [depth_cmap, depth_cmap, error_cmap]
    norms      = [depth_norm, depth_norm, error_norm]

    im_depth = None
    im_error = None

    for row_idx, (data, cmap, norm, row_label) in enumerate(
            zip(datasets, cmaps, norms, row_labels)):
        data_plot = data.copy()
        data_plot[data_plot == 0] = np.nan

        for col_idx, t in enumerate(time_points):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(data_plot[t], cmap=cmap, norm=norm, aspect='auto')
            if row_idx == 0:
                ax.set_title(f'{t + 1} min', fontsize=fontsize)
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=fontsize)
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, width=1)
            for spine in ax.spines.values():
                spine.set_linewidth(1)
            if row_idx < 2:
                im_depth = im
            else:
                im_error = im

    # ── Colorbars ────────────────────────────────────────────────────────────
    plt.subplots_adjust(right=0.87, hspace=0.35, wspace=0.25)
    fig.canvas.draw()   # fix axis positions before reading .get_position()

    # Shared depth colorbar spanning rows 0–1
    pos0 = axes[0, -1].get_position()
    pos1 = axes[1, -1].get_position()
    cbar_ax1 = fig.add_axes([0.89, pos1.y0, 0.015, pos0.y1 - pos1.y0])
    cb1 = fig.colorbar(im_depth, cax=cbar_ax1)
    cb1.set_label('Water depth (m)', size=fontsize)
    cb1.ax.tick_params(labelsize=tick_fontsize)

    # Error colorbar for row 2
    pos2 = axes[2, -1].get_position()
    cbar_ax2 = fig.add_axes([0.89, pos2.y0, 0.015, pos2.height])
    cb2 = fig.colorbar(im_error, cax=cbar_ax2)
    cb2.set_label('Absolute error (m)', size=fontsize)
    cb2.ax.tick_params(labelsize=tick_fontsize)

    plt.savefig(f"{save_path}/water_depth_spatial_temporal.png",
                bbox_inches="tight", dpi=150)
    plt.close()


def compute_metrics(pred_mm, gt_mm, flood_thres=150.0):
    """Compute six evaluation metrics for one flood event.

    Both inputs are in millimetres (mm), shape (T, H, W).

    Metrics
    -------
    R2      : coefficient of determination (all space × time)
    MSE     : mean squared error (m²)
    RMSE    : root MSE (m)
    MAE     : mean absolute error (m)
    PeakR2  : R² restricted to the timestep of maximum spatial-mean depth
    CSI     : Critical Success Index (temporal-max flood extent)

    Parameters
    ----------
    pred_mm : np.ndarray, (T, H, W)
        Model predictions in mm.
    gt_mm : np.ndarray, (T, H, W)
        Ground-truth flood depths in mm.
    flood_thres : float
        Wet/dry threshold in mm (default 150 mm = 0.15 m).
    """
    # ── Convert to metres for standard metrics ──────────────────────────────
    import torch as _torch
    if isinstance(pred_mm, _torch.Tensor):
        pred_mm = pred_mm.cpu().numpy()
    if isinstance(gt_mm, _torch.Tensor):
        gt_mm = gt_mm.cpu().numpy()
    pred_m = pred_mm / 1000.0
    gt_m   = gt_mm   / 1000.0

    # ── R² ─────────────────────────────────────────────────────────────────
    ss_res = np.sum((pred_m - gt_m) ** 2)
    ss_tot = np.sum((gt_m - gt_m.mean()) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-10)

    # ── MSE / RMSE / MAE ────────────────────────────────────────────────────
    mse  = float(np.mean((pred_m - gt_m) ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(pred_m - gt_m)))

    # ── PeakR²: R² at the timestep of maximum spatial-mean ground-truth ────
    spatial_mean_gt = gt_m.mean(axis=(1, 2))   # (T,)
    t_peak = int(np.argmax(spatial_mean_gt))
    pred_p = pred_m[t_peak].ravel()
    gt_p   = gt_m[t_peak].ravel()
    ss_res_p = np.sum((pred_p - gt_p) ** 2)
    ss_tot_p = np.sum((gt_p - gt_p.mean()) ** 2)
    peak_r2 = float(1.0 - ss_res_p / (ss_tot_p + 1e-10))

    # ── CSI: binary flood extent from temporal maximum depth ────────────────
    pred_max = pred_mm.max(axis=0)   # (H, W) in mm
    gt_max   = gt_mm.max(axis=0)     # (H, W) in mm
    pred_flooded = pred_max > flood_thres
    gt_flooded   = gt_max   > flood_thres
    tp = int(np.sum( pred_flooded &  gt_flooded))
    fp = int(np.sum( pred_flooded & ~gt_flooded))
    fn = int(np.sum(~pred_flooded &  gt_flooded))
    csi = float(tp / (tp + fp + fn + 1e-10))

    return {
        "R2":     float(r2),
        "MSE":    mse,
        "RMSE":   rmse,
        "MAE":    mae,
        "PeakR2": peak_r2,
        "CSI":    csi,
    }


def save_metrics(all_metrics, save_metric_dir, cur_epoch):
    """Save per-event metrics and summary statistics to an Excel file.

    Parameters
    ----------
    all_metrics : dict[str, dict]
        ``{event_name: {"R2": ..., "MSE": ..., "RMSE": ..., "MAE": ...,
                        "PeakR2": ..., "CSI": ...}}``
    save_metric_dir : str
        Directory where the Excel file will be written.
    cur_epoch : int
        Epoch number (embedded in the filename).
    """
    os.makedirs(save_metric_dir, exist_ok=True)

    rows = []
    for event_name, m in all_metrics.items():
        rows.append({"Event": event_name, **m})

    # Summary row (mean ± std)
    cols = ["R2", "MSE", "RMSE", "MAE", "PeakR2", "CSI"]
    summary_mean = {c: np.mean([r[c] for r in rows]) for c in cols}
    summary_std  = {c: np.std( [r[c] for r in rows]) for c in cols}
    rows.append({"Event": "MEAN",  **summary_mean})
    rows.append({"Event": "STD",   **summary_std})

    df = pd.DataFrame(rows)
    excel_path = os.path.join(save_metric_dir, f"metrics_epoch{cur_epoch}.xlsx")
    df.to_excel(excel_path, index=False, float_format="%.6f")
    print(f"\n[Metrics] Saved → {excel_path}")

    # ── Print summary table ─────────────────────────────────────────────────
    print(f"\n{'Event':<30s}  {'R2':>7s}  {'RMSE(m)':>9s}  {'MAE(m)':>8s}  {'PeakR2':>7s}  {'CSI':>6s}")
    print("-" * 80)
    for event_name, m in all_metrics.items():
        print(f"{event_name:<30s}  {m['R2']:7.4f}  {m['RMSE']:9.4f}  {m['MAE']:8.4f}  {m['PeakR2']:7.4f}  {m['CSI']:6.4f}")
    print("-" * 80)
    print(f"{'MEAN':<30s}  {summary_mean['R2']:7.4f}  {summary_mean['RMSE']:9.4f}  "
          f"{summary_mean['MAE']:8.4f}  {summary_mean['PeakR2']:7.4f}  {summary_mean['CSI']:6.4f}")
    print(f"{'STD':<30s}  {summary_std['R2']:7.4f}  {summary_std['RMSE']:9.4f}  "
          f"{summary_std['MAE']:8.4f}  {summary_std['PeakR2']:7.4f}  {summary_std['CSI']:6.4f}\n")



def load_dataset(args, local_rank):
    """
    Loads the test dataset using the Dynamic2DFlood dataset class with a DataLoader.

    Parameters:
    - args: Configuration namespace with data_root, test_list_file, etc.
    - local_rank: DDP rank; -1 for single-GPU.

    Returns:
    - A DataLoader for the test dataset.
    """
    test_folder = Dynamic2DFlood(
        data_root=args.data_root,
        split="test",
        event_list_file=getattr(args, "test_list_file", None),
        duration=args.duration,
        location=getattr(args, "location", ""),
    )

    # Create a distributed sampler if running in a distributed environment, otherwise, use None
    test_sampler = (
        None
        if local_rank == -1
        else torch.utils.data.distributed.DistributedSampler(test_folder, shuffle=False)
    )

    test_loader = torch.utils.data.DataLoader(
        test_folder,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        sampler=test_sampler,
        pin_memory=True,
        drop_last=False
    )

    return test_loader


def label_reg2cls(reg_targets):
    """
    Converts regression targets to binary class labels based on the presence of water.

    Parameters:
    - reg_targets: A tensor of regression targets where positive values indicate water presence.

    Returns:
    - A tensor where each element is 1 if water is present, otherwise 0.
    """
    return (reg_targets > 0).float()


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


if __name__ == "__main__":
    exp_root = "../exp"
    local_rank, rank = initialize_environment_variables()
    args = ArgumentParsers(exp_root)
    # load dataset & model
    args = load_model_params(args)
    device = init_device(args.device, args.batch_size, local_rank)
    testLoader = load_dataset(args, local_rank)

    if args.trt:
        cuda, cudart, pydrcuda, trt, common, cuda_call = initialize_environment()
    test(args, device, testLoader, cur_epoch=100)
