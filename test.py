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
import os
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

    def __init__(self, engine_path, T):
        """
        Initializes the inference engine, creates the execution context, and allocates
        necessary buffers and streams.

        Parameters:
        - engine_path: Path to the serialized TensorRT engine.
        - T: Number of time steps to process in the inference loop.
        """
        self.T = T
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(
            self.engine)

        # Define the output size in bytes, assuming float32 with HxW of 500x500
        self.H, self.W = 500, 500
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
        state_shapes = [
            (1, 64, self.H, self.W), (1, 96, self.H//2, self.W//2), (1,
                                                                     96, self.H//4, self.W//4),  # Encoder states
            (1, 96, self.H//4, self.W//4), (1, 96, self.H//2, self.W//2), (1,
                                                                           64, self.H, self.W)   # Decoder states
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


def Inference_with_TensorRT(inference, inputs, device, batch_size=350):
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
        f, inputs, device).cpu().numpy() for f in range(Frames)]

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


def Inference(net, inputs):
    """
    Performs inference over all time steps in the input data using the specified network.

    Parameters:
    - net: The neural network model to use for inference.
    - inputs: A dictionary containing the input data with key 'rainfall' pointing to a tensor of shape (T, H, W).

    Returns:
    - output_data: List of outputs from the network for each time step.
    """

    with torch.no_grad():
        net.eval()
        Frames = inputs['rainfall'].shape[1]

        prev_encoder_state1, prev_encoder_state2, prev_encoder_state3, \
            prev_decoder_state1, prev_decoder_state2, prev_decoder_state3 = initialize_states(
                device)

        output_data = []
        test_start_time = time.time()
        t = 0
        for t in range(Frames):
            input_t = preprocess_inputs(t, inputs, device)
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
        output_data = [output.cpu().numpy()[0, 0, 0] for output in output_data]
        output_data = np.array(output_data)

    return output_data


def load_net(args):
    # Initialize model
    net = ED(args.clstm,
             args.model_params["encoder_params"],
             args.model_params["decoder_params"],
             args.cls_thred,
             args.use_checkpoint)

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


def test(args, device, testLoader, cur_epoch=99999, flood_max=5000):
    """
    Runs the test phase for the model on provided data.

    Parameters:
    - args: Configuration and runtime arguments.
    - device: Device on which to run the test.
    - testLoader: DataLoader providing test datasets.
    - cur_epoch: Current epoch of model training.
    - flood_max: Maximum scale value for normalization.
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
        net = load_net(args)
    else:
        trt_net = TensorRTInference(os.path.join(
            args.trt_model_dir, "URNN.trt"), T=args.window_size)

    save_epoch = os.path.join(args.save_fig_dir, "epoch@"+str(cur_epoch))
    if not os.path.exists(save_epoch):
        os.makedirs(save_epoch, exist_ok=True)

    # Start test
    print("Start testing...")
    batch = tqdm(testLoader, leave=False, total=len(testLoader))
    for _, (inputs, _, event_dir) in enumerate(batch):
        print("event_dir:", event_dir)
        inputs = to_device(inputs, device)

        if not args.trt:
            print("Infer with PyTorch...")
            output_data = Inference(net, inputs)
        else:
            print("Infer with TensorRT...")
            output_data = Inference_with_TensorRT(trt_net, inputs, device)

        post_process(output_data, save_epoch, event_dir, flood_max)


def post_process(output_data, save_epoch, event_dir, flood_max=5000):
    """
    Applies post-processing to the output data from inference, normalizes it, and visualizes
    the results at specified time points.

    Parameters:
    - output_data: The raw output data from the model.
    - save_epoch: Base directory where the output visualizations will be saved.
    - event_dir: Directory path containing event data, used to derive the event name.
    - flood_max: The maximum flood value for reverse normalization.

    Returns:
    - None
    """
    # Reverse Min-Max scaling on the output data to bring it back to the original scale
    output_data = r_MinMaxScaler(output_data, max=flood_max, min=0)

    # Extract the event name from the path
    event_name = os.path.basename(event_dir[0])
    save_epoch_sample = os.path.join(save_epoch, event_name)

    # Ensure the target directory exists, creating it if necessary
    os.makedirs(save_epoch_sample, exist_ok=True)

    # Specify the time points for visualizations
    time_points = [0, 119, 239, 359]
    plot_spatial_distributions(
        output_data, time_points=time_points, save_path=save_epoch_sample)


def plot_spatial_distributions(data, time_points=[0, 50, 99], unit=1000, save_path="./"):
    """
    Plot spatial distributions for specified time points in a data array, with a unified color scale
    and a single colorbar for all subplots. Zero values are shown in white.

    Parameters:
    - data: numpy array with shape (T, H, W), where T is the number of time points,
            H is the height, and W is the width of the spatial distribution.
    - time_points: list of indices for the time points to plot.
    - unit: A divisor to scale the data values.
    - save_path: directory path where the plot image will be saved.
    """
    num_plots = len(time_points)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots+1, 5))

    fontsize = 15
    tick_fontsize = fontsize-2

    # Create a new colormap from the jet colormap
    cmap = plt.cm.jet(np.linspace(0, 1, 256))
    # Set the first color entry to white for zero values
    cmap[0] = (1, 1, 1, 1)
    new_cmap = ListedColormap(cmap)
    new_cmap.set_bad(color='white')

    data /= unit

    # Determine the global min and max
    global_min = np.min(data)
    global_max = np.max(data)

    data[data == 0] = np.nan

    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable

    # Configure each subplot
    for ax, t in zip(axes, time_points):
        im = ax.imshow(data[t], cmap=new_cmap, aspect='auto',
                       norm=Normalize(vmin=global_min, vmax=global_max))
        ax.set_title(f'{t+1} min', fontsize=fontsize)
        ax.tick_params(axis='both', which='major',
                       labelsize=tick_fontsize, width=1)
        for spine in ax.spines.values():
            spine.set_linewidth(1)

    # Place a single colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Water depth (m)', size=fontsize)
    # Set font size for colorbar ticks
    cbar.ax.tick_params(labelsize=tick_fontsize)

    plt.savefig(f"{save_path}/water_depth_spatial_temporal.png",
                bbox_inches="tight")

    plt.close()


def load_dataset(args, local_rank):
    """
    Loads the test dataset using the Dynamic2DFlood dataset class with a DataLoader.

    Parameters:
    - args: A namespace or other object with attribute 'data_root' specifying the root directory of the data.
    - local_rank: The rank of the device on the local machine, used in distributed training to set the specific GPU.

    Returns:
    - A DataLoader for the test dataset.
    """
    test_folder = Dynamic2DFlood(data_root=args.data_root, split="test")

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
    - args: Updated argument object including model parameters.
    """
    params = get_network_params(args.use_checkpoint)
    # Assign predefined model parameters to the args
    args.model_params = {
        "encoder_params": params[0],
        "decoder_params": params[1],
    }
    return args


if __name__ == "__main__":
    timestamp = "20240202_162801_962166"
    exp_root = "./exp"
    local_rank, rank = initialize_environment_variables()
    args = ArgumentParsers(exp_root, timestamp=timestamp)
    # load dataset & model
    args = load_model_params(args)
    device = init_device(args.device, args.batch_size, local_rank)
    testLoader = load_dataset(args, local_rank)

    if args.trt:
        cuda, cudart, pydrcuda, trt, common, cuda_call = initialize_environment()
    test(args, device, testLoader, cur_epoch=100)
