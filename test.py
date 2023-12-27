import wandb
from config import ArgumentParsers
from visual_results import (
    show_results,
)

import time
import numpy as np
from tqdm import tqdm
import torch
from src.lib.model.networks.net_params_w_cls_16_64_96_k1 import (
# from src.lib.model.networks.net_params_w_cls_16_64_96_k3 import (
    
    convgru_encoder_params,
    convgru_decoder_params,
)
from src.lib.model.networks.model import ED
import os
# import seaborn as sns
from src.lib.dataset.Dynamic2DFlood import (Dynamic2DFlood, preprocess_inputs,
                                            MinMaxScaler, r_MinMaxScaler)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    # encoder = Encoder(args.clstm,
    #                   encoder_params[0],
    #                   encoder_params[1],
    #                   use_checkpoint=args.use_checkpoint)
    # decoder = Decoder(args.clstm,
    #                   decoder_params[0],
    #                   decoder_params[1],
    #                   use_checkpoint=args.use_checkpoint)
    # head = YOLOXHead()  # in_channels为decoder最后一个stage的输出channel
    # net = ED(encoder, decoder, head)
    # net = net.to(device)
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
    # cur_epoch = model_info["epoch"]
    print("loaded model:%s" % (model_path))
    state_dict = {}
    for k, v in model_info["state_dict"].items():
        if k[:7] == "module.":
            state_dict[k[7:]] = v
        else:
            state_dict[k] = v
    net.load_state_dict(state_dict)
    # net.load_state_dict(model_info["state_dict"])
    net = net.to(device)
    # 定义损失函数
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
            H, W = inputs['absolute_DEM'].shape[-2:]
            """
            迭代条件：当降雨为0（t>=Frames）后，判断flood峰值是否达到内涝水准，
            如果是，那么就停止迭代，否则继续迭代
            
            从第t=0开始迭代，不需要中间抽出来计算loss，计算总loss
            """
            loc = 0
            t = loc
            ind_ = max(0, loc - 1)
            depth_output_t = label[:, ind_:ind_ + 1]

            prev_encoder_state = [None] * args.blocks
            prev_decoder_state = [None] * args.blocks
            test_losses = []
            test_cls_losses = []
            pred = None
            cur_step = 0
            test_start_time = time.time()
            # while (t < Frames):
            while (t < Frames) or \
                    (t < event_Frames):
                #  and output_t.max() >= args.flood_thres):
                # input of t-th frame
                output_t_info = {
                    "output_t": depth_output_t,
                }
                input_t = preprocess_inputs(t, inputs, output_t_info, device)

                depth_output_t, cls_output_t, encoder_state_t, decoder_state_t = net(
                        input_t,  prev_encoder_state, prev_decoder_state)
                depth_output_t = torch.clamp(depth_output_t, min=0.0)  # 内涝不能为负

                # # get pred
                output_t = {"reg": depth_output_t, "cls": cls_output_t}
                if pred is None:
                    pred = output_t
                else:
                    pred["reg"] = torch.cat((pred["reg"], output_t["reg"]),
                                            dim=1)
                    pred["cls"] = torch.cat((pred["cls"], output_t["cls"]),
                                            dim=1)

                # update
                prev_encoder_state = encoder_state_t
                prev_decoder_state = decoder_state_t
                cur_step += 1
                t += 1

            test_end_time = time.time()
            test_use_time = test_end_time - test_start_time
            print("test_time: %.2f sec %d step. %.2f sec/step" %
                  (test_use_time, t, test_use_time / t))

            # 保存每一帧的pred-label
            # pred["cls"] = r_MinMaxScaler(pred["cls"], 1, 0) # 不需要
            pred["reg"] = r_MinMaxScaler(pred["reg"], flood_max, 0)

            cls_label = label_reg2cls(label)
            # m = torch.unique(cls_label)
            pred["cls"] = (pred["cls"] >= 0.5).float() # ! 强制置1

            label = r_MinMaxScaler(label, flood_max, 0)
            label_dict = {
                "reg": label,
                "cls": cls_label,
            }
            
            pred_cls_h = pred["cls"].to("cpu").detach().numpy()
            pred_h = pred["reg"].to("cpu").detach().numpy()
            label_cls_h = label_dict["cls"].to("cpu").detach().numpy()
            label_h = label_dict["reg"].to("cpu").detach().numpy()
            # 保存标签和预测结果
            if not os.path.exists(args.save_loss_dir):
                os.makedirs(args.save_loss_dir, exist_ok=True)
            pred_huv_save_path = os.path.join(
                args.save_loss_dir, "pred_%d_%s.npz" % (i, str(args.cls_thred).replace(".",""))
            )
            label_huv_save_path = os.path.join(
                args.save_loss_dir, "label_%d_%s.npz" % (i, str(args.cls_thred).replace(".",""))
            )
            np.savez(pred_huv_save_path, pred_h=pred_h,pred_cls_h=pred_cls_h)
            np.savez(label_huv_save_path, label_h=label_h,label_cls_h=label_cls_h)

    
            event_name = event_dir[0].split("/")[-1]
            save_epoch_sample = os.path.join(save_epoch, f"_sample@{i}_{event_name}")

            if not os.path.exists(save_epoch_sample):
                os.makedirs(save_epoch_sample, exist_ok=True)
            
            rainfall = inputs["rainfall"][0,:,0,0,0]
            # for task in ["reg", ]:
            for task in [
                "cls",
                "reg",
            ]:
                unit = 1 if task == "cls" else 1000
                show_results(pred,
                            label_dict,
                            rainfall,
                            task,
                            test_losses=test_losses,
                            save_fig_dir=save_epoch_sample,
                            unit=unit,
                            upload=upload)


def load_dataset(args):
    testFolder = Dynamic2DFlood(data_root=args.data_root, split="test")

    testLoader = torch.utils.data.DataLoader(
        testFolder,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
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


if __name__ == "__main__":

    print("="*10)
    # print("Exp: loss_name@%s, reduction@%s" % (loss_name, reduction))
    # 设置参数
    timestamp = "20231216_235718_449685"
    args = ArgumentParsers(exp_root='../../exp', timestamp=timestamp)
    # load dataset、model
    args = load_model_params(args)
    testLoader = load_dataset(args)
    # test
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test(
        args, device, testLoader,999998,
    )
