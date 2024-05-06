import tensorrt as trt
import torch.onnx
from config import ArgumentParsers
import numpy as np
from tqdm import tqdm
import torch
from src.lib.model.networks.model import ED
import os
from src.lib.dataset.Dynamic2DFlood import preprocess_inputs
from pathlib import Path
import sys
from test import load_model_params, init_device, load_dataset
from src.lib.utils.general import initialize_environment_variables, to_device, initialize_states


def build_engine(onnx_file_path, engine_file_path):
    """
    Builds a TensorRT engine from an ONNX file.

    Parameters:
    - onnx_file_path: Path to the ONNX file.
    - engine_file_path: Path where the TensorRT engine will be saved.

    Returns:
    - The built TensorRT engine, or None if building fails.
    """
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    if not os.path.exists(onnx_file_path):
        print(
            f"ONNX file {onnx_file_path} not found, please generate it first.")
        return None

    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network() as network, \
            builder.create_builder_config() as config, \
            trt.OnnxParser(network, TRT_LOGGER) as parser, \
            trt.Runtime(TRT_LOGGER) as runtime:

        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, 20 * (1 << 30))  # 10GB

        print(f"Loading ONNX file from path {onnx_file_path}...")
        with open(onnx_file_path, "rb") as model:
            print("Beginning ONNX file parsing")
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # Define the optimization profile and set dimensions for the input layer
        profile = builder.create_optimization_profile()
        input_shape = (1, 1, 63, 500, 500)
        profile.set_shape("input_t", input_shape, input_shape, input_shape)
        config.add_optimization_profile(profile)

        print(
            f"Building an engine from file {onnx_file_path}; this may take a while...")
        plan = builder.build_serialized_network(network, config)
        if not plan:
            print("Failed to build the engine.")
            return None

        engine = runtime.deserialize_cuda_engine(plan)
        print("Completed creating Engine")

        # Save the engine to a file
        with open(engine_file_path, "wb") as f:
            f.write(plan)
        return engine


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

    # Start test
    print("Start testing...")

    with torch.no_grad():
        net.eval()
        batch = tqdm(testLoader, leave=False, total=len(testLoader))
        for i, (inputs, _, _) in enumerate(batch):
            if i > 0:
                break
            inputs = to_device(inputs, device)
            input0 = preprocess_inputs(0, inputs, device)

            prev_encoder_state1, prev_encoder_state2, prev_encoder_state3, \
                prev_decoder_state1, prev_decoder_state2, prev_decoder_state3 = initialize_states(
                    device)

            # Export PyTorch to ONNX
            if not os.path.exists(args.trt_model_dir):
                os.makedirs(args.trt_model_dir, exist_ok=True)
            save_tensorrt_model_path = os.path.join(
                args.trt_model_dir, "URNN.onnx")
            torch.onnx.export(net,
                              (input0,
                               prev_encoder_state1, prev_encoder_state2, prev_encoder_state3,
                               prev_decoder_state1, prev_decoder_state2, prev_decoder_state3,
                               ),
                              save_tensorrt_model_path,
                              export_params=True,
                              opset_version=11,
                              do_constant_folding=True,
                              input_names=['input_t',
                                           'prev_encoder_state1', 'prev_encoder_state2', 'prev_encoder_state3',
                                           'prev_decoder_state1', 'prev_decoder_state2', 'prev_decoder_state3',
                                           ],
                              output_names=['output_h',
                                            'encoder_state_t1', 'encoder_state_t2', 'encoder_state_t3',
                                            'decoder_state_t1', 'decoder_state_t2', 'decoder_state_t3',
                                            ],
                              dynamic_axes={'input_t': {0: 'batch_size'},
                                            'output_h': {0: 'batch_size'}})

            # Export ONNX to TensorRT
            save_engine_path = os.path.join(args.trt_model_dir, "URNN.trt")
            build_engine(save_tensorrt_model_path, save_engine_path)


if __name__ == "__main__":
    timestamp = "20240202_162801_962166"
    exp_root = "./exp"

    local_rank, rank = initialize_environment_variables()
    args = ArgumentParsers(exp_root, timestamp=timestamp)
    # load dataset & model
    args = load_model_params(args)
    device = init_device(args.device, args.batch_size, local_rank)
    testLoader = load_dataset(args, local_rank)

    test(args, device, testLoader)
