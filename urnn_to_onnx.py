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


def test(args, device, testLoader, cur_epoch, upload=False):
    """
    main function to run the training
    """
    # seed
    random_seed = args.random_seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(random_seed)
    else:
        torch.cuda.manual_seed(random_seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = True

    # net构建
    net = ED(args.clstm,
             args.model_params["encoder_params"],
             args.model_params["decoder_params"],
             args.cls_thred,
             args.use_checkpoint)

    # load model
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
    
    """开始测试"""
    print("start test...")
    
    save_epoch = os.path.join(args.save_fig_dir,"epoch@"+str(cur_epoch))
    if not os.path.exists(save_epoch):
        os.makedirs(save_epoch, exist_ok=True)
    # r2 = []
    with torch.no_grad():
        net.eval()

        batch = tqdm(testLoader, leave=False, total=len(testLoader))
        # 按batch_size来遍历
        for i, (inputs, label,event_dir) in enumerate(batch):
            print("event_dir:",event_dir)
            # if i > 0:
            #     break
            inputs = to_device(inputs, device)
            label = label.to(device, dtype=torch.float32)

            flood_max = 5 * 1000
            label = MinMaxScaler(label, flood_max, 0)

            # 随机获取连续N帧
            Frames = inputs['rainfall'].shape[1]  # 获取降雨时长
            event_Frames = label.shape[1]

            loc = 0
            t = loc
            prev_encoder_state1 = torch.zeros((1, 64, 500, 500)).cuda()
            prev_encoder_state2 = torch.zeros((1, 96, 250, 250)).cuda()
            prev_encoder_state3 = torch.zeros((1, 96, 125, 125)).cuda()

            prev_decoder_state1 = torch.zeros((1, 96, 125, 125)).cuda()
            prev_decoder_state2 = torch.zeros((1, 96, 250, 250)).cuda()
            prev_decoder_state3 = torch.zeros((1, 64, 500, 500)).cuda()

            input_gen = []
            for f in range(Frames):
                input_gen.append(preprocess_inputs(f, inputs, device))
            
            input_t = input_gen[0]
            # 准备ONNX导出
            torch.onnx.export(net, 
                            (input_t, 
                             prev_encoder_state1,prev_encoder_state2,prev_encoder_state3,
                             prev_decoder_state1,prev_decoder_state2,prev_decoder_state3,
                             ), 
                            "net_model.onnx", 
                            export_params=True, 
                            opset_version=11, 
                            do_constant_folding=True,  # 是否执行常量折叠优化
                            input_names=['input_t', 
                                         'prev_encoder_state1', 'prev_encoder_state2', 'prev_encoder_state3', 
                                         'prev_decoder_state1','prev_decoder_state2','prev_decoder_state3',
                                         ],
                            output_names=['output_h',
                                           'encoder_state_t1','encoder_state_t2','encoder_state_t3',
                                             'decoder_state_t1','decoder_state_t2','decoder_state_t3',
                                             ],
                            dynamic_axes={'input_t': {0: 'batch_size'},  # 可以添加更多的动态轴
                                            'output_h': {0: 'batch_size'}})


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
