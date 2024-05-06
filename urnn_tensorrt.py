import matplotlib.pyplot as plt
import torch.onnx
import torch.distributed as dist
from src.lib.utils import select_device
import wandb
from config import ArgumentParsers
import time
import numpy as np
from tqdm import tqdm
import torch
from src.lib.model.networks.net_params_w_cls_16_64_96_k1 import (
    convgru_encoder_params,
    convgru_decoder_params,
)
from src.lib.model.networks.model import ED
import os
from src.lib.dataset.Dynamic2DFlood import (Dynamic2DFlood, preprocess_inputs,
                                            MinMaxScaler, r_MinMaxScaler)
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", -1))
print("LOCAL RANK:", LOCAL_RANK)
print("RANK:", RANK)
print("WORLD_SIZE:", WORLD_SIZE)


def to_device(inputs, device):
    for key in inputs.keys():
        inputs[key] = inputs[key].to(device)

    return inputs


import time
import numpy as np
import common
import tensorrt as trt
from cuda import cuda, cudart
import pycuda.driver as pydrcuda
import pycuda.autoinit

np.random.seed(42)

def check_cuda_err(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))

def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res


def _do_inference_base(inputs, outputs, stream, execute_async_func):
    # Transfer input data to the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    [cuda_call(cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, kind, stream)) for inp in inputs]
    
    # Run inference.
    execute_async_func()
    
    # Synchronize the stream to ensure all computations are done.
    cuda_call(cudart.cudaStreamSynchronize(stream))
    
    # Return device pointers of outputs directly.
    return [out.device for out in outputs], outputs


def do_inference(context, engine, bindings, inputs, outputs, stream):
    def execute_async_func():
        context.execute_async_v3(stream_handle=stream)
    # Setup context tensor address.
    num_io = engine.num_io_tensors
    for i in range(num_io):
        context.set_tensor_address(engine.get_tensor_name(i), bindings[i])
    return _do_inference_base(inputs, outputs, stream, execute_async_func)


class TensorRTInference:
    def __init__(self, engine_path, T):
        self.T = T
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)

        H = 500  # 输出高度
        W = 500  # 输出宽度
        self.output_size_in_bytes = H * W * 4  # float32类型占4个字节

    @staticmethod
    def load_engine(engine_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def prepare_initial_data(self,input_ts):
        # # Generate initial input_t for T steps
        self.input_ts = input_ts
        # 计算单步输入数据的大小
        single_input_size = self.input_ts[0].nbytes
        # 为所有步骤的输入数据分配内存
        self.device_input_ts = pydrcuda.mem_alloc(self.T * single_input_size)
        
        # 将所有输入数据复制到 GPU
        for i in range(self.T):
            offset = i * single_input_size
            device_data_ptr = int(self.device_input_ts) + offset
            pydrcuda.memcpy_htod(device_data_ptr, self.input_ts[i])

        
        # State initialization
        state_shapes = [
            (1, 64, 500, 500), (1, 96, 250, 250), (1, 96, 125, 125),  # Encoder states
            (1, 96, 125, 125), (1, 96, 250, 250), (1, 64, 500, 500)   # Decoder states
        ]
        self.device_states = []
        for shape in state_shapes:
            state = np.zeros(shape, dtype=np.float32)
            device_state = pydrcuda.mem_alloc(state.nbytes)
            pydrcuda.memcpy_htod(device_state, state)
            self.device_states.append(device_state)


    def inferStep(self, input_ts):
        self.input_ts = input_ts

        single_input_size = self.input_ts[0].nbytes
        output_buffers = [pydrcuda.mem_alloc(self.output_size_in_bytes) for _ in range(self.T)]  # 为每个时刻的输出分配单独的GPU内存

        for t in range(self.T):
            self.bindings[0] = int(self.device_input_ts) + t * single_input_size

            output_data, outputs_origin = do_inference(self.context, self.engine, self.bindings, self.inputs, self.outputs, self.stream)

            # 将输出保存到独立的GPU缓冲区，避免覆盖
            pydrcuda.memcpy_dtod(output_buffers[t], outputs_origin[0].device, self.output_size_in_bytes)

            self.bindings[1:7] = output_data[1:7]

        return output_buffers


    def inferStep_v1(self, input_ts):
        self.input_ts=input_ts
        outputs = []
        single_input_size = self.input_ts[0].nbytes
        
        # # Bind these states to self.bindings
        for idx, device_state in enumerate(self.device_states):
            self.bindings[idx+1] = device_state

        for t in range(self.T):     
            # 设置当前步骤的输入数据指针
            self.bindings[0] = int(self.device_input_ts) + t * single_input_size
            
            # Perform inference
            output_data,outputs_origin = do_inference(self.context, self.engine, self.bindings, self.inputs, self.outputs, self.stream)
            outputs.append(outputs_origin[0])
            
            # Update state bindings for the next iteration
            for idx in range(6): 
                self.bindings[1+ idx] = output_data[1 + idx]  # 更新新的状态到bindings
        
        return outputs

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs
    
    def cuda_to_npy(self, output_data):
        # 一次性从所有GPU缓冲区复制数据到CPU
        host_output_data = [np.empty((500, 500), dtype=np.float32) for _ in range(len(output_data))]  # 假设输出形状为 H x W
        for i, buffer in enumerate(output_data):
            pydrcuda.memcpy_dtoh(host_output_data[i], buffer)

        output_data = np.array(host_output_data)

        output_data = output_data.reshape(-1, 500, 500)
        
        return output_data


def test(args, device, testLoader, cur_epoch, upload=False):
    """
    main function to run the training
    """
    
    """开始测试"""
    print("start test...")
    
    save_epoch = os.path.join(args.save_fig_dir,"epoch@"+str(cur_epoch))
    if not os.path.exists(save_epoch):
        os.makedirs(save_epoch, exist_ok=True)

    batch = tqdm(testLoader, leave=False, total=len(testLoader))
    # 按batch_size来遍历
    for i, (inputs, label,event_dir) in enumerate(batch):
        print("event_dir:",event_dir)
        if i > 0:
            break
        inputs = to_device(inputs, device)
        label = label.to(device, dtype=torch.float32)
        flood_max = 5 * 1000
        label = MinMaxScaler(label, flood_max, 0)
        # 随机获取连续N帧
        Frames = inputs['rainfall'].shape[1]  # 获取降雨时长
        input_gen = []
        for f in range(Frames):
            input_t = preprocess_inputs(f, inputs, device)
            input_gen.append(input_t.cpu().numpy())
      
        inference = TensorRTInference("net_model.trt", T=Frames)
        # 数据初始化
        inference.prepare_initial_data(input_gen)

        start = time.time()
        output_data = inference.inferStep(input_gen)
        print(f"use time: {time.time()-start}")
        # 输出后处理
        output_data = inference.cuda_to_npy(output_data)
        # 结果可视化
        plot_spatial_distributions(output_data,time_points=[0, 120-1, 240-1, 359])


def plot_spatial_distributions(data, time_points= [0, 50, 99]):
    """绘制指定时间点的空间分布图。

    参数:
    - data: 一个形状为 (T, H, W) 的 numpy 数组。
    - time_points: 一个包含要绘制的时间点索引的列表。
    """
    num_plots = len(time_points)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    
    if num_plots == 1:
        axes = [axes]  # 如果只有一个图，确保 axes 是可迭代的
    
    for ax, t in zip(axes, time_points):
        im = ax.imshow(data[t], cmap='viridis', aspect='auto')
        ax.set_title(f'Time = {t+1}')
        fig.colorbar(im, ax=ax)
    
    plt.tight_layout()
    # plt.show()
    plt.savefig("spatialtemporal_tensorrt.png")
    plt.close()

def plot_max_spatial_distribution(output_data, cmap='viridis', figsize=(10, 8), title='Max Spatial Distribution Over Time'):
    """
    Plots the maximum spatial distribution over time for a given 3D NumPy array.
    
    Parameters:
        output_data (numpy.ndarray): A 3D NumPy array with shape (T, H, W) where T is time, H is height, and W is width.
        cmap (str): Color map for the heatmap.
        figsize (tuple): Figure size for the plot.
        title (str): Title of the plot.
    """
    if not isinstance(output_data, np.ndarray) or output_data.ndim != 3:
        raise ValueError("output_data must be a 3D NumPy array")
    
    # Calculate the maximum values across the time dimension
    max_values = np.max(output_data, axis=0)
    
    # Create the plot
    plt.figure(figsize=figsize)
    im = plt.imshow(max_values, cmap=cmap, interpolation='nearest')
    plt.colorbar(im)  # Add a color bar to a plot
    plt.title(title)
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.savefig("max_spatial_distribution.png")
    plt.close()

def load_dataset(args):
    testFolder = Dynamic2DFlood(data_root=args.data_root, split="test")

    
    test_sampler = (
        None
        if LOCAL_RANK == -1
        else torch.utils.data.distributed.DistributedSampler(testFolder, shuffle=False)
    )

    testLoader = torch.utils.data.DataLoader(
        testFolder,
        batch_size=1,
        # batch_size=args.batch_size // torch.cuda.device_count(), # 分布式训练必须的
        shuffle=False,
        num_workers=1,
        sampler=test_sampler,
        pin_memory=True,
        drop_last=False,
    )


    return testLoader


def load_model_params(args):
    
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params

    # 初始化wandb，包括启动、参数保存
    args.model_params = {
        "encoder_params": encoder_params,
        "decoder_params": decoder_params
    }
    return args


def label_reg2cls(reg_targets):
    """
    只分为有洪水1，无洪水0
    depth>0的为有洪水，否则为无洪水
    """
    return (reg_targets > 0).float()

def init_device(device, batch_size):
    device = select_device(device, batch_size)
    if LOCAL_RANK != -1:
        assert (
            torch.cuda.device_count() > LOCAL_RANK
        ), "insufficient CUDA devices for DDP command"
        print("torch.cuda.device_count():", torch.cuda.device_count())
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    return device

if __name__ == "__main__":

    print("="*10)
    # print("Exp: loss_name@%s, reduction@%s" % (loss_name, reduction))
    # 设置参数
    timestamp = "20240224_180730_484545"
    args = ArgumentParsers(exp_root='../../exp', timestamp=timestamp)
    # load dataset、model
    args = load_model_params(args)
    device = init_device(args.device, args.batch_size)
    testLoader = load_dataset(args)
    # test
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test(
        args, device, testLoader,999996,
    )
