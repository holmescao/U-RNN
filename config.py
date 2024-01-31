import torch
import argparse
import os
import datetime

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# conda activate cnnlstm
# CUDA_VISIBLE_DEVICES=1 python main.py --device 1 -batch_size 1
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port 2122 main.py  --device 1 -batch_size 1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 2135 main.py --device 0,1,2,3,4,5,6,7 -batch_size 8
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 2131 main.py --device 0,1 -batch_size 2
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=7 main.py --device 1,2,3,4,5,6,7 -batch_size 7


def ArgumentParsers(
    exp_root="../../exp",
    timestamp="timestamp",
):
    description = """
    输入: 
    - 将输入区域为500x500的网格; 16个样本，train:test=12:4
    - 所有变量都用最大最小归一化；DEM_max=最值, 降雨_max=峰值，内涝_max=峰值3500mm，不透水性_max=0.95，排水口_max=1
    - 去除机理:flow_volumn
    - loc=0; window_size改为事件长度; seq_num=45, 顺序窗口
    - 将完整的一场降雨-内涝事件送进去训练（包括降雨期间和降雨结束的退水过程）

    输出: 当前时刻的洪水
    模型: GRU, 卷积核1x1；stride=2改为1，用平均池化来做下采样;(16_64_96); 
        cls和reg走branch+4层;
        Conv的head:cls不加GN,reg不加GN;ED不加GN;
        用cls修正reg
    损失函数: 新的FocalBCE_and_WMSE@0.8; 不上限1e+5; reg*2*4.5; 
    head激活函数：acts = ["silu"] * 3 + ["sigmoid", "lrelu"];
    模型优化方案: 
    - lr=1e-2(multi), patience=10, factor=0.9, min_lr=1e-4; epoch=100; schedule: WarmUpCosineAnneal；if True:
    - 用checkpoint技术; 去除AMP技术(pytorch版本，不做梯度scale)
    其他: 每个epoch都保存；DDP技术
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="在一个node上进程的相对序号，local_rank在node之间相互独立",
    )
    parser.add_argument(
        "-exp_name", default="ConvLSTM-model optimize", type=str, help="experiment name"
    )
    parser.add_argument(
        "-description", default=description, type=str, help="description of expr"
    )
    parser.add_argument(
        "-exp_root", default=exp_root, type=str, help="experiment root directory"
    )
    parser.add_argument(
        "-exp_dir",
        default=os.path.join(exp_root, timestamp),
        type=str,
        help="experiment root directory",
    )
    parser.add_argument(
        "-data_root",
        default="../../data/urbanflood22",
        type=str,
        help="data root directory",
    )
    parser.add_argument(
        "--clstm",
        help="use convlstm as base cell",
        default=False,
    )
    parser.add_argument(
        "-use_checkpoint", default=True, type=bool, help="sequence number per iteration"
    )
    parser.add_argument(
        "-seq_num", default=28, type=int, help="sequence number per iteration"
    )
    parser.add_argument(
        "-window_size", default=360, type=int, help="window size per sample"
    )
    parser.add_argument(
        "-train_event", default=True, type=bool, help="train full event or rain process"
    )
    parser.add_argument(
        "-all_seq_train", default=False, type=bool, help="all seq per forward"
    )
    parser.add_argument(
        "-full_window_size",
        default=False,
        type=bool,
        help="whether seq_num == window size per backward",
    )
    parser.add_argument("-blocks", default=3, type=int,
                        help="stage number of network")
    parser.add_argument("-batch_size", default=1, type=int, help="total batch")
    parser.add_argument("-random_seed", default=42,
                        type=int, help="random seed")
    parser.add_argument("-num_workers", default=20,
                        type=int, help="number of workers")
    # parser.add_argument("-gpu_list",
    #                     default=[0, 3],
    #                     type=list,
    #                     help="gpu list")
    parser.add_argument("--device", default="0", type=str, help="gpu list")
    parser.add_argument("-lr", default=1e-2, type=float, help="learning rate")
    parser.add_argument(
        "-schedule_name",
        default="WarmUpCosineAnneal",
        type=str,
        help="schedule_name: ReduceLROnPlateau | WarmUpCosineAnneal",
    )
    parser.add_argument("-warm_up_iter", default=10,
                        type=float, help="warm_up_iter")
    parser.add_argument(
        "-patience", default=10, type=float, help="patience of learning rate decrease"
    )
    parser.add_argument(
        "-factor", default=0.9, type=float, help="decrease rate of learning rate"
    )
    parser.add_argument("-epochs", default=1000, type=int, help="sum of epochs")
    parser.add_argument(
        "-loss_name",
        default="FocalBCE_and_WMSE",
        type=str,
        help="loss function: FocalBCE_and_W2MSE | FocalBCE_and_WMSE | FocalBCE_and_Flood_MSE | WMSE | MSE | Focal_MSE | Focal_MSE_BCE",
    )
    parser.add_argument(
        "-reduction", default="mean", type=str, help="loss function: mean|sum"
    )
    parser.add_argument(
        "-save_loss_dir",
        type=str,
        default="%s/%s/save_train_loss" % (exp_root, timestamp),
        help="save directory of final loss",
    )
    parser.add_argument(
        "-save_res_data_dir",
        type=str,
        default="%s/%s/save_res_data" % (exp_root, timestamp),
        help="save directory of final loss",
    )
    parser.add_argument(
        "-save_model_dir",
        type=str,
        default="%s/%s/save_model" % (exp_root, timestamp),
        help="save directory of model",
    )
    parser.add_argument(
        "-save_dir_flood_maps",
        type=str,
        default="%s/%s/flood_maps" % (exp_root, timestamp),
        help="save directory of pred vs. label flood maps",
    )
    parser.add_argument(
        "-save_fig_dir",
        type=str,
        default="%s/%s/figs/" % (exp_root, timestamp),
        help="save directory of figures",
    )
    parser.add_argument(
        "-flood_thres", type=float, default=0.15 * 1000, help="flood threshold"
    )
    parser.add_argument(
        "-model_params", default="See wandb", help="model params")
    parser.add_argument("-upload", default=True, help="upload to wandb")
    parser.add_argument("-wind_random", default=True,
                        help="random window segment")
    parser.add_argument("-test", default=True, help="test after train")
    parser.add_argument("-amp", default=False, help="mix fp16,32 ")
    parser.add_argument("-timestamp", default=timestamp, help="time stamp")
    parser.add_argument("-resume", type=bool, default=True, help="time stamp")
    parser.add_argument("-cls_thred", type=float, default=0.5, help="time stamp")
    args = parser.parse_args()

    return args
