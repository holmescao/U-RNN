import torch.distributed as dist
from src.lib.utils import select_device
import math

# from torch.cuda.amp import autocast, GradScaler
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
import sys
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
from src.lib.dataset.Dynamic2DFlood import (
    Dynamic2DFlood,
    preprocess_inputs,
    MinMaxScaler,
)
from src.lib.model.networks.net_params_w_cls_16_64_96_k1 import (
    
    convgru_encoder_params,
    convgru_decoder_params,
)

from src.lib.model.networks.model import ED
from src.lib.model.networks.losses import select_loss_function
from config import ArgumentParsers
from pathlib import Path

# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(int(str(device).split(":")[-1]))

# CUDA_VISIBLE_DEVICES = "0"

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


def correction_seq_num(seq_num, window_size, full_window_size=False):
    seq_num = window_size if full_window_size else min(seq_num, window_size)

    return seq_num


def correction_window_size(
    window_size, rain_len, event_len, all_seq_train=False, train_event=False
):
    # 如果不是全序列训练，那么就按设置的来
    # 否则，就改为全序列的参数
    sample_length = event_len if train_event else rain_len
    window_size = sample_length if all_seq_train else min(window_size, sample_length)

    return window_size


def get_start_loc(rain_len, window_size, event_len, train_event=False):
    sample_length = event_len if train_event else rain_len

    if sample_length - window_size <= 0:
        loc = 0
    else:
        # 可随机选取起始位置
        loc = np.random.randint(0, sample_length - window_size, size=1, dtype=int)[0]
        # loc = 0
        # print("loc:", loc)

    return loc


def split_iter_index(start_loc, seq_num, window_size):
    sample_length = window_size - start_loc

    iter_indexes = list(range(start_loc, sample_length, seq_num))
    if sample_length % seq_num > 0:
        iter_indexes[-1] = window_size - seq_num

    return iter_indexes


def WarmUpCosineAnneal(optimizer, warm_up_iter, T_max, lr_max, lr_min):
    # 设置学习率调整规则 - Warm up + Cosine Anneal
    def lambda0(cur_iter):
        # print("########################cur_iter:", cur_iter)
        cur_iter = cur_iter+47 # TODO: 为了恢复而改的
        lr = (
            cur_iter / warm_up_iter
            if cur_iter < warm_up_iter
            else (
                lr_min
                + 0.5
                * (lr_max - lr_min)
                * (
                    1.0
                    + math.cos(
                        (cur_iter - warm_up_iter) / (T_max - warm_up_iter) * math.pi
                    )
                )
            )
            / 0.1
        )
        # print(f"学习率:{lr}")
        return lr

    # LambdaLR
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda0])

    return scheduler


def lr_schedule(optimizer, schedule_name, lr, warm_up_iter, epochs):
    if schedule_name == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.9,
            patience=10,
            cooldown=0,
            verbose=True,
            # min_lr=1e-4,
            min_lr=5e-5,
        )
    elif schedule_name == "WarmUpCosineAnneal":
        # scheduler = WarmUpCosineAnneal(
        #     optimizer, warm_up_iter=10, T_max=1000, lr_max=1e-2, lr_min=1e-4)
        scheduler = WarmUpCosineAnneal(
            optimizer, warm_up_iter=warm_up_iter, T_max=epochs, lr_max=lr, lr_min=1e-4
        )

    return scheduler


def init_torch_seeds(seed=0):
    """用在general.py的init_seeds函数
    用于初始化随机种子并确定训练模式
    Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    """
    # 为CPU设置随机种子，方便下次复现实验结果  to seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)

    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed(seed)
    # benchmark模式会自动寻找最优配置 但由于计算的随机性 每次网络进行前向传播时会有差异
    # 避免这种差异的方法就是将deterministic设置为True(表明每次卷积的高效算法相同)
    # 速度与可重复性之间的权衡  涉及底层卷积算法优化
    # slower, more reproducible  慢 但是具有可重复性 适用于网络的输入数据在每次iteration都变化的话
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
    # if seed == 42:
    #     # slower, more reproducible  慢 但是具有可重复性 适用于网络的输入数据在每次iteration都变化的话
    #     torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    # else:
    #     # faster, less reproducible 快 但是不可重复  适用于网络的输入数据维度或类型上变化不大
    #     torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = True, False


def init_seeds(random_seed):
    # seed
    np.random.seed(random_seed)
    random.seed(random_seed)
    init_torch_seeds(random_seed)


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


def load_model(args, device):
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
            model_names, key=lambda x: int(x.replace("checkpoint_", "").split("_")[0])
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

        # net.load_state_dict(model_info["state_dict"])
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

        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if RANK in {-1, 0}:
            if not os.path.isdir(args.save_model_dir):
                os.makedirs(args.save_model_dir)
            torch.save({"state_dict": net.state_dict()}, checkpoint_path)
            print("saved model!")
        if RANK != -1:
            dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        model_info = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(model_info["state_dict"])

        cur_epoch = 0
    net = net.to(device)
    print(f"cur_epoch:{cur_epoch}")
    # Convert to DDP
    cuda = device.type != "cpu"
    if cuda and RANK != -1:
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK
        )

    return net, optimizer, cur_epoch


def set_optimizer(optimizer, args):
    # 学习率要根据并行GPU的数量进行倍增
    # lr = args.lr * args.batch_size
    # lr = args.batch_size * args.lr if RANK != - \
    #     1 else args.lr
    # lr = args.lr * WORLD_SIZE if RANK != - \
    #     1 else args.lr
    lr = args.lr
    # pg = [p for p in net.parameters() if p.requires_grad]
    # optimizer = optim.Adam(net.parameters(), lr=0.000937555)
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    # 学习率方案
    scheduler = lr_schedule(
        optimizer, args.schedule_name, lr, args.warm_up_iter, args.epochs
    )
    # 训练优化方案
    lossfunction = select_loss_function(args.loss_name, args.reduction)

    return scheduler, lossfunction


def get_window(args, inputs, label):
    # correction/get: window_size, seq_num, loc
    rain_len = inputs["rainfall"].shape[1]  # 获取降雨时长
    event_len = label.shape[1]
    args.window_size = correction_window_size(
        args.window_size, rain_len, event_len, args.all_seq_train, args.train_event
    )
    loc = get_start_loc(rain_len, args.window_size, event_len, args.train_event)
    args.seq_num = correction_seq_num(
        args.seq_num, args.window_size, args.full_window_size
    )

    ind = loc
    cur_iter = 0

    # 分割、随机选取起始位置
    iter_indexes = split_iter_index(loc, args.seq_num, args.window_size)
    if args.wind_random:
        random.shuffle(iter_indexes)

    return ind, cur_iter, iter_indexes


def get_loss_from_gpu(losses, iter_loss, seq_num):
    for k, v in losses.items():
        v = v.item()
        # v = v.item() / seq_num
        # losses[k] = reduce_value(v, average=True)
        # iter_loss[k].append(losses[k])
        iter_loss[k].append(v)

    return iter_loss


def model_forward(
    args, batch, epoch, net, inputs, label, device, optimizer, lossfunction
):
    ind, cur_iter, iter_indexes = get_window(args, inputs, label)

    # clear lists to track next epoch
    iter_loss = {
        "loss": [],
        "loss_reg": [],
        "loss_reg_label": [],
        "loss_reg_pred": [],
        "loss_cls": [],
        "loss_ssim": [],
    }

    # nums = len(iter_indexes)//3  # 每次随机学习三分之一的特征 
    # for ind in iter_indexes[:nums]:
    for ind in iter_indexes:
        # 由于洪水问题，每次都从0时刻开始
        depth_output_t = label[:, 0:1]
        depth_output_t = depth_output_t.to(device, dtype=torch.float32)

        iter_label = label[:, ind : ind + args.seq_num]
        iter_label = iter_label.to(device, dtype=torch.float32)

        # state_prev_t = [None] * args.blocks
        prev_encoder_state = [None] * args.blocks
        prev_decoder_state = [None] * args.blocks
        pred = None
        optimizer.zero_grad()  # 梯度清零

        if ind > 0:
            """
            模拟生成0到t-1时刻的flood，作为t-1时刻的flood输入，
            从而当前窗口从一开始就是有累计误差的
            """

            with torch.no_grad():
                for t in range(0, ind):
                    # input of t-th frame
                    output_t_info = {
                        "output_t": depth_output_t,
                    }
                    input_t = preprocess_inputs(t, inputs, output_t_info, device)

                    # depth_output_t, _, state_t = net(input_t, state_prev_t)
                    depth_output_t, cls_output_t, encoder_state_t, decoder_state_t = net(
                        input_t,  prev_encoder_state, prev_decoder_state)

                    # update
                    prev_encoder_state = encoder_state_t
                    prev_decoder_state = decoder_state_t

        for t in range(ind, ind + args.seq_num):
            # input of t-th frame
            output_t_info = {
                "output_t": depth_output_t,
            }
            input_t = preprocess_inputs(t, inputs, output_t_info, device)
            # depth_output_t, cls_output_t, state_t = net(input_t, state_prev_t)
            depth_output_t, cls_output_t, encoder_state_t, decoder_state_t = net(
                        input_t,  prev_encoder_state, prev_decoder_state)

            # get pred
            if pred is None:
                pred = {"reg": depth_output_t, "cls": cls_output_t}
            else:
                pred["reg"] = torch.cat((pred["reg"], depth_output_t), dim=1)
                pred["cls"] = torch.cat((pred["cls"], cls_output_t), dim=1)

            # update
            prev_encoder_state = encoder_state_t
            prev_decoder_state = decoder_state_t
            cur_iter += 1

        # 计算pred与label的每一步误差
        losses = lossfunction(pred, iter_label, epoch)

        losses["loss"].backward()

        iter_loss = get_loss_from_gpu(losses, iter_loss, args.seq_num)

        # 梯度裁剪，以防止梯度爆炸
        # torch.nn.utils.clip_grad_value_(net.parameters(),
        #                                 clip_value=10.0)

        optimizer.step()  # 更新一次网络参数

        if RANK in {-1, 0}:
            batch.set_postfix(
                {
                    "iter": "%d/%d" % (cur_iter // args.seq_num, len(iter_indexes)),
                    "total": "{:.9f}".format(np.average(iter_loss["loss"])),
                    "reg": "{:.9f}".format(np.average(iter_loss["loss_reg"])),
                    "reg_label": "{:.9f}".format(
                        np.average(iter_loss["loss_reg_label"])
                    ),
                    "reg_pred": "{:.9f}".format(np.average(iter_loss["loss_reg_pred"])),
                    "cls": "{:.9f}".format(np.average(iter_loss["loss_cls"])),
                    "ssim": "{:.9f}".format(np.average(iter_loss["loss_ssim"])),
                    "epoch": "{:02d}".format(epoch),
                }
            )

    return optimizer, iter_loss


def train_one_epoch(
    args, net, optimizer, scheduler, lossfunction, trainLoader, device, epoch
):
    net.train()

    pbar = enumerate(trainLoader)
    if RANK in {-1, 0}:
        pbar = tqdm(pbar, leave=False, total=len(trainLoader))

    # 按batch_size来遍历
    for step, (inputs, label,_) in pbar:
        # # TODO: 只训练一半
        # if step > 1:
        #     break
        # print("rank:%d. %s" % (RANK, event_dir))
        inputs = to_device(inputs, device)
        flood_max = 5 * 1000 # ! 最高不超过5m
        label = MinMaxScaler(label, flood_max, 0)
        optimizer, iter_loss = model_forward(
            args, pbar, epoch, net, inputs, label, device, optimizer, lossfunction
        )
    # 等待所有进程计算完毕
    if RANK != -1:
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

    upload_info = {"lr": optimizer.param_groups[0]["lr"]}
    for k, v in iter_loss.items():
        upload_info[k] = np.average(v)

    scheduler = scheduler_update(args, scheduler, upload_info)

    return optimizer, upload_info


def print_epoch_train_info(epoch, upload_info, epoch_start_time, epoch_end_time):
    if args.upload:
        wandb.log(upload_info)

    train_info = ["%s:%.9f | " % (k, v) for k, v in upload_info.items()]
    log_info = (
        "epoch:%d | " % (epoch + 1)
        + "".join(train_info)
        + "time:%.2f sec" % (epoch_end_time - epoch_start_time)
    )

    logging.info(log_info)

    epoch_len = len(str(args.epochs))

    print_msg = (
        f"[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] "
        + f"total: {upload_info['loss']:.9f} | "
        + f"reg: {upload_info['loss_reg']:.9f} | "
        + f"reg_label: {upload_info['loss_reg_label']:.9f} | "
        + f"reg_pred: {upload_info['loss_reg_pred']:.9f} | "
        + f"cls: {upload_info['loss_cls']:.9f} | "
        + f"ssim: {upload_info['loss_ssim']:.9f} | "
        + "time:%.2f sec" % (epoch_end_time - epoch_start_time)
    )

    print(print_msg)


def scheduler_update(args, scheduler, upload_info):
    if args.schedule_name == "ReduceLROnPlateau":
        scheduler.step(upload_info["loss"])
    elif args.schedule_name == "WarmUpCosineAnneal":
        scheduler.step()

    return scheduler


def train(args, device, trainLoader, train_sampler, testLoader):
    # TODO:重构下，需要封装成Trainer
    """
    main function to run the training
    """
    # Settings
    init_seeds(args.random_seed)

    # Load model
    net, optimizer, cur_epoch = load_model(args, device)
    # Optimizer
    scheduler, lossfunction = set_optimizer(optimizer, args)
    if RANK in {-1, 0}:
        save_best_model = SaveBestModel(patience=20, verbose=True)

    # scaler = GradScaler() # AMP
    """begin training"""
    # wandb.watch(net, log="all")
    for epoch in range(cur_epoch, args.epochs):
        if RANK != -1:
            trainLoader.sampler.set_epoch(epoch)

        epoch_start_time = time.time()
        optimizer, upload_info = train_one_epoch(
            args=args,
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            lossfunction=lossfunction,
            trainLoader=trainLoader,
            device=device,
            epoch=epoch,
        )
        epoch_end_time = time.time()

        if RANK in {-1, 0}:
            if (epoch + 1) % 50 == 0:
                # test
                if args.test:
                    test(args, device, testLoader, epoch, upload=args.upload)

            # print training/validation statistics
            # calculate average loss over an epoch
            print_epoch_train_info(epoch, upload_info, epoch_start_time, epoch_end_time)

            # 保存模型
            save_best_model(
                upload_info["loss"], net, optimizer, epoch, args.save_model_dir
            )


def load_dataset(args):
    # Instance dataset
    trainvalFolder = Dynamic2DFlood(
        data_root=args.data_root,
        split="trainval",
    )
    testFolder = Dynamic2DFlood(
        data_root=args.data_root,
        split="test",
    )

    # split train and val dataset
    # ratio = [10, 0]
    # train_len = len(trainvalFolder) * ratio[0] // 10
    # train_len = 2  # 测试
    # val_len = len(trainvalFolder) - train_len
    # print("train:val = {}:{} samples.".format(train_len, val_len))
    # trainvalFolder, valFolder = torch.utils.data.random_split(
    #     trainvalFolder, [train_len, val_len])

    # num_workers
    nw = 1
    # nw = int(os.cpu_count() / torch.cuda.device_count() / 2)
    if RANK in {-1, 0}:
        print("Using {} dataloader workers every process".format(nw))

    # Assign the training sample index to each rank corresponding process
    train_sampler = (
        None
        if LOCAL_RANK == -1
        else torch.utils.data.distributed.DistributedSampler(
            trainvalFolder, shuffle=True
        )
    )
    test_sampler = (
        None
        if LOCAL_RANK == -1
        else torch.utils.data.distributed.DistributedSampler(testFolder, shuffle=False)
    )

    trainLoader = torch.utils.data.DataLoader(
        trainvalFolder,
        batch_size=1, # debug时用的
        # batch_size=args.batch_size // torch.cuda.device_count(), # 分布式训练必须的
        shuffle=train_sampler is None,
        num_workers=nw,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
    )
    # # valLoader = torch.utils.data.DataLoader(
    # #     valFolder,
    # #     batch_size=args.batch_size,
    # #     shuffle=False,
    # #     num_workers=nw,
    # #     pin_memory=False,
    # #     drop_last=True,
    # # )
    testLoader = torch.utils.data.DataLoader(
        testFolder,
        batch_size=1,
        # batch_size=args.batch_size // torch.cuda.device_count(), # 分布式训练必须的
        shuffle=False,
        num_workers=nw,
        sampler=test_sampler,
        pin_memory=True,
        drop_last=False,
    )

    return trainLoader, train_sampler, testLoader


def load_model_params(args):
    
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params

    # 初始化wandb，包括启动、参数保存
    args.model_params = {
        "encoder_params": encoder_params,
        "decoder_params": decoder_params,
    }
    return args


def RecordsPrepare(args):
    run = None
    log_filename = ""
    if RANK in {-1, 0}:
        expr_dir = os.path.join(args.exp_root, args.timestamp)
        exp_record.params_to_yaml(args, expr_dir)
        exp_record.save_overall_records(
            timestamp=args.timestamp,
            description=args.description,
            done=False,
            save_dir="exp_records",
            file_name=args.exp_name,
        )
        # git: 代码保存
        commit_id = exp_record.git_code(args.timestamp + "\n" + args.description)

        if args.upload:
            run = exp_record.wandb_init(
                job_type="Training",
                id=args.timestamp,
                name=args.timestamp,
                config=args,
                project=args.exp_name,
                notes="git:%s\n%s" % (commit_id, args.description),
                wandb_dir="../../",
            )
            # wandb.run.log_code(".")

        log_filename = exp_record.init_logging(expr_dir)

    return run, log_filename


def update_opts(args, timestamp_save_path):
    with open(timestamp_save_path, "r") as f:
        timestamp = f.readline()
    print("Exp start! now: ", timestamp)

    args.timestamp = args.timestamp.replace("timestamp", timestamp)
    args.save_loss_dir = args.save_loss_dir.replace("timestamp", timestamp)
    args.save_model_dir = args.save_model_dir.replace("timestamp", timestamp)
    args.save_dir_flood_maps = args.save_dir_flood_maps.replace("timestamp", timestamp)
    args.save_fig_dir = args.save_fig_dir.replace("timestamp", timestamp)
    args.exp_dir = args.exp_dir.replace("timestamp", timestamp)

    return args


def main(args, timestamp_save_path):
    # init device
    device = init_device(args.device, args.batch_size)
    # update opts
    args = update_opts(args, timestamp_save_path)
    # load dataset, model
    trainLoader, train_sampler, testLoader = load_dataset(args)
    args = load_model_params(args)
    # 保存实验信息
    run, log_filename = RecordsPrepare(args)

    try:
        # train
        train(
            args,
            device,
            trainLoader,
            train_sampler,
            testLoader,
        )

        # exp_record.save_overall_records(timestamp=timestamp,
        #                                 description=args.description,
        #                                 done=True,
        #                                 save_dir="exp_records",
        #                                 file_name=args.exp_name)
        # test
        if RANK in {-1, 0}:
            if args.test:
                test(
                    args,
                    device,
                    testLoader,
                    args.epochs,
                    upload=args.upload,
                )

        if RANK in {-1, 0}:  # 只需要在主进程创建
            if args.upload:
                run.finish()

    except Exception as e:
        if RANK in {-1, 0}:  # 只需要在主进程创建
            traceback.print_exc(file=open(log_filename, "a+"))
            traceback.print_exc()


if __name__ == "__main__":
    exp_root = "../../exp"
    timestamp_save_path = "%s/timestamp.txt" % exp_root

    if RANK in {-1, 0}:
        now = datetime.datetime.now()
        # timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
        timestamp = "20240131_154653_360175"
        with open(timestamp_save_path, "w") as f:
            f.write(timestamp)

    args = ArgumentParsers(exp_root)
    main(args, timestamp_save_path)
