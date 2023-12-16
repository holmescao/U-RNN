import logging
import wandb
from sklearn.metrics import r2_score
import pandas as pd
import torch
from tqdm import tqdm
from celluloid import Camera
import matplotlib.cm as cm
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def vis_train_loss(all_data, start_epoch, save_fig_dir):

    for key, val in all_data.items():
        xx = range(start_epoch, len(val))
        plt.plot(xx, val[start_epoch:], label=key)

    plt.legend()
    plt.xlabel("Training Epoch")
    plt.ylabel("Loss")

    if not os.path.exists(save_fig_dir):
        os.makedirs(save_fig_dir, exist_ok=True)

    reduction = key.split("-")[1]
    save_fig_path = os.path.join(save_fig_dir, "vs_loss-%s.png" % reduction)
    plt.savefig(save_fig_path, dpi=150, bbox_inches="tight")
    plt.close()


def vis_event_loss(all_data, start_epoch, save_fig_dir):
    """
    对比不同方法在同一场内涝事件下的误差
    """
    for key, val in all_data.items():
        xx = range(start_epoch, len(val))
        plt.plot(xx, val[start_epoch:], label=key)

    plt.legend()
    plt.xlabel("Time step (min)")
    plt.ylabel("Loss")

    save_fig_path = os.path.join(save_fig_dir,
                                 "TF@%d_vs_event_step_loss.png" % (len(xx)))

    plt.savefig(save_fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    for key, val in all_data.items():
        xx = range(start_epoch, len(val))
        plt.plot(xx, np.cumsum(val[start_epoch:]), label=key)

    plt.legend()
    plt.xlabel("Time step (min)")
    plt.ylabel("Cumulative Loss")

    save_fig_path = os.path.join(save_fig_dir,
                                 "TF@%d_vs_event_cumulative_loss.png" % (len(xx)))

    plt.savefig(save_fig_path, dpi=150, bbox_inches="tight")
    plt.close()


def vis_total_volumn(event_data, title, save_fig_dir):
    pred = event_data[0]
    label = event_data[1]
    rainfall = event_data[2]

    # ind = np.unravel_index(np.argmax(label), label.shape)

    # pred_max_ind = pred[0, :, 0, ind[-2], ind[-1]]
    # label_max_ind = label[0, :, 0, ind[-2], ind[-1]]

    pred = torch.sum(pred, dim=4)
    pred = torch.sum(pred, dim=3)
    label = torch.sum(label, dim=4)
    label = torch.sum(label, dim=3)

    pred = pred.reshape(-1)
    label = label.reshape(-1)

    # plt.plot(pred, c='r', label="pred")
    # plt.plot(label, c='g', label="label", ls="--")
    # plt.legend()

    xlabel_title = "Time step (min)"
    # ylabel_title = title
    # plt.xlabel(xlabel_title)
    # plt.ylabel(ylabel_title)
    # plt.tight_layout()

    # Create a figure and a single subplot
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot the rainfall timeseries as bars on the first axis
    color = 'tab:blue'
    # ax1.plot(range(len(cumulative_rainfall)), cumulative_rainfall, color='tab:green', label='Cumulative Rainfall', linestyle='--')

   
    # ax1.legend(loc='upper left')
    ax1.plot(range(len(rainfall)) , rainfall,color=color, alpha=0.6, label='Rainfall')
    ax1.bar(range(len(rainfall)) , rainfall,color=color, alpha=0.6, label='Rainfall')
    ax1.set_xlabel(xlabel_title)
    ax1.set_ylabel('Rainfall (mm/min)', color=color)
    ax1.set_ylim([8, 0])
    # ax1.set_ylim([rainfall.max()*3, 0])
    ax1.tick_params(axis='y', labelcolor=color)
    # ax1.set_title('Rainfall and Flood Timeseries at Spatial Location of Max Value')
    # ax1.legend(loc='upper right')

    # Create a second axis for the flood timeseries
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.plot(pred, c='r', label="pred")
    ax2.plot(label, c='g', label="label", ls="--")
    
    # ax2.plot(timeseries_at_max_location, color=color, label='Flood at Max Value Location')
    # ax2.plot(range(len(runoff)) , runoff,color="red", alpha=0.6, label=f'runoff')
    ax2.set_ylabel('Flood Depth (m)', color=color)
    ax2.set_ylim([0, label.max()*3])
    ax2.tick_params(axis='y', labelcolor=color)
    # ax2.legend(loc='upper left')
    plt.legend(loc='upper right')
    # Show the figure
    fig.tight_layout()
    
    if not os.path.exists(save_fig_dir):
        os.makedirs(save_fig_dir, exist_ok=True)

    save_fig_path = os.path.join(
        save_fig_dir, "%s.png" % (title))

    plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    return pred, label, xlabel_title


def vis_max_depth(event_data, title, save_fig_dir):
    pred = event_data[0].numpy()
    label = event_data[1].numpy()
    rainfall = event_data[2].numpy()

    ind = np.unravel_index(np.argmax(label), label.shape)

    pred_max_ind = pred[0, :, 0, ind[-2], ind[-1]]
    label_max_ind = label[0, :, 0, ind[-2], ind[-1]]

    # plt.plot(pred_max_ind, c='r', label="pred")
    # plt.plot(label_max_ind, c='g', label="label", ls="--")
    # plt.legend()

    # plt.xlabel("Time step (min)")
    # plt.ylabel(title)
    # plt.tight_layout()

    # Create a figure and a single subplot
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot the rainfall timeseries as bars on the first axis
    color = 'tab:blue'
    # ax1.plot(range(len(cumulative_rainfall)), cumulative_rainfall, color='tab:green', label='Cumulative Rainfall', linestyle='--')

   
    # ax1.legend(loc='upper left')
    ax1.plot(range(len(rainfall)) , rainfall,color=color, alpha=0.6, label='Rainfall')
    ax1.bar(range(len(rainfall)) , rainfall,color=color, alpha=0.6, label='Rainfall')
    ax1.set_xlabel("Time step (min)")
    ax1.set_ylabel('Rainfall (mm/min)', color=color)
    ax1.set_ylim([8, 0])
    # ax1.set_ylim([rainfall.max()*3, 0])
    ax1.tick_params(axis='y', labelcolor=color)
    # ax1.set_title('Rainfall and Flood Timeseries at Spatial Location of Max Value')
    # ax1.legend(loc='upper right')

    # Create a second axis for the flood timeseries
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.plot(pred_max_ind, c='r', label="pred")
    ax2.plot(label_max_ind, c='g', label="label", ls="--")
    
    # ax2.plot(timeseries_at_max_location, color=color, label='Flood at Max Value Location')
    # ax2.plot(range(len(runoff)) , runoff,color="red", alpha=0.6, label=f'runoff')
    ax2.set_ylabel('Flood Depth (m)', color=color)
    ax2.set_ylim([0, label_max_ind.max()*3])
    ax2.tick_params(axis='y', labelcolor=color)
    # ax2.legend(loc='upper left')
    plt.legend(loc='upper right')
    # Show the figure
    fig.tight_layout()
    
    save_fig_path = os.path.join(
        save_fig_dir, "TF@%d_vs_maxdepthpoint_%s.png" % (len(pred_max_ind), title))

    plt.savefig(save_fig_path, dpi=150, bbox_inches="tight")
    plt.close()


def vis_event_dynamic(event_data, label_name, save_fig_dir):
    """
    绘制时空动态对比图: pred vs label
    """
    pred = event_data[0]
    label = event_data[1]

    frames = pred.shape[1]
    height = pred.shape[-2]
    width = pred.shape[-1]

    v_min = 0
    v_max = max(pred.max(), label.max())

    num = 30
    pal = [cm.jet(i / num) for i in range(num)]
    pal.insert(0, (1, 1, 1, 1))
    pal = sns.color_palette(palette=pal)

    bar = tqdm(range(0, frames, 30))
    bar.set_description(desc="Frame")
    for f in bar:
        fig = plt.figure(figsize=(10, int(5 * (height / width))))
        ax = plt.axes()
        ax.set_ylim(0, height)
        ax.set_xlim(0, width)

        pred_f = pred[0, f, 0]
        pred_f = torch.clamp(pred_f, min=0)
        label_f = label[0, f, 0]
        data = torch.cat((pred_f, label_f), dim=1)
        h = sns.heatmap(
            data,
            cmap=pal,
            cbar=False,
            vmin=v_min,
            vmax=v_max,
        )

        # 自定义colorbar
        cb = h.figure.colorbar(h.collections[0])  # 显示colorbar
        cb.ax.tick_params(labelsize=25)  # 设置colorbar刻度字体大小
        font = {
            "family": "serif",
            # "color": "darkred",
            "weight": "normal",
            "size": 30,
        }
        cb.set_label(label_name, fontdict=font)  # 设置colorbar的label

        # plt.title("event_dynamic")
        # plt.title("event_dynamic-%s-%s"
        #           % (loss_name, reduction))
        plt.grid()
        plt.tight_layout()

        save_fig_dir_ = os.path.join(save_fig_dir, "dynamic")
        if not os.path.exists(save_fig_dir_):
            os.makedirs(save_fig_dir_, exist_ok=True)
        save_fig_path = os.path.join(save_fig_dir_,
                                     "TF@%d_F@%d_event_dynamic-%s.png"
                                     % (frames, f, label_name))

        plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
        plt.close()


def vis_event_dynamic_gif(event_data, title, save_fig_dir):
    """
    绘制时空动态对比图: pred vs label
    """
    pred = event_data[0]
    label = event_data[1]

    frames = pred.shape[1]
    height = pred.shape[-2]
    width = pred.shape[-1]

    fig = plt.figure(figsize=(10, int(5 * (height / width))), dpi=300)
    ax = plt.axes()
    ax.set_ylim(0, height)
    ax.set_xlim(0, width)

    v_min = 0
    v_max = max(pred.max(), label.max())

    num = 30
    pal = [cm.jet(i / num) for i in range(num)]
    pal.insert(0, (1, 1, 1, 1))
    pal = sns.color_palette(palette=pal)
    camera = Camera(fig)
    bar = tqdm(range(0, frames, 10))
    bar.set_description(desc="Frame")
    for f in bar:
        pred_f = pred[0, f, 0]
        pred_f = torch.clamp(pred_f, min=0)
        label_f = label[0, f, 0]
        data = torch.cat((pred_f, label_f), dim=1)
        h = sns.heatmap(
            data,
            cmap=pal,
            cbar=False,
            vmin=v_min,
            vmax=v_max,
        )
        camera.snap()

    # 自定义colorbar
    cb = h.figure.colorbar(h.collections[0])  # 显示colorbar
    cb.ax.tick_params(labelsize=25)  # 设置colorbar刻度字体大小
    font = {
        "family": "serif",
        # "color": "darkred",
        "weight": "normal",
        "size": 30,
    }
    cb.set_label(title, fontdict=font)  # 设置colorbar的label

    # plt.title("event_dynamic")
    plt.grid()
    plt.tight_layout()

    if not os.path.exists(save_fig_dir):
        os.makedirs(save_fig_dir, exist_ok=True)
    save_fig_path = os.path.join(
        save_fig_dir, "event_dynamic_%s.gif" % (title))

    animation = camera.animate()
    animation.save(
        save_fig_path,
        writer="imagemagick",
        fps=10,
    )
    plt.clf()
    plt.close()


def get_train_loss(exp_dir, exp_all_loss):
    exp_path = os.path.join(exp_dir,
                            "reduction@%s/lossfunc@%s/save_loss/" %
                            (reduction, loss_name))

    file_name = "lossfunc@%s_reduction@%s.npy" % (loss_name, reduction)
    loss = np.load(os.path.join(exp_path, file_name))

    exp_all_loss[loss_name+"-"+reduction] = loss

    return exp_all_loss


def get_event_loss(exp_dir, batch_id, mode, reduction, loss_name):
    file_name = "batch%d_%s_lossfunc@%s_reduction@%s.npy" % (
        batch_id, mode, loss_name, reduction)
    exp_path = os.path.join(exp_dir,
                            "reduction@%s/lossfunc@%s/save_loss/" %
                            (reduction, loss_name))

    loss = np.load(os.path.join(exp_path, file_name))
    return loss


def get_event_maps(exp_dir, batch_id,):
    file_name = "batch%d_flood_maps_predvslabel.npy" % (
        batch_id)
    exp_path = os.path.join(exp_dir,
                            "reduction@%s/lossfunc@%s/flood_maps/" %
                            (reduction, loss_name))

    event_data = np.load(os.path.join(exp_path, file_name), allow_pickle=True)
    pred = event_data.item()["pred"].cpu()
    label = event_data.item()["label"].cpu()
    return (pred, label)


def vis_scatter(event_data, save_fig_dir):
    pred = event_data[0].reshape(-1).cpu().numpy()
    label = event_data[1].reshape(-1).cpu().numpy()

    data = np.array([label, pred]).reshape(-1, 2)
    df_scatter = pd.DataFrame(data, columns=["Label", "Prediction"])

    sns.regplot(x="Label", y="Prediction", data=df_scatter)

    # save_fig_dir_ = os.path.join(save_fig_dir, "R2")
    # if not os.path.exists(save_fig_dir_):
    #     os.makedirs(save_fig_dir_, exist_ok=True)
    save_fig_path = os.path.join(save_fig_dir, "R2.png")
    plt.savefig(
        save_fig_path,
        bbox_inches="tight",
        dpi=150,
    )
    plt.close()


def show_results(pred, label, rainfall, task, test_losses,
                 save_fig_dir,
                 unit=1000, upload=False):

    
    event_data = (pred[task][:, :].cpu()/unit,
                  label[task][:, :pred[task].shape[1]].cpu()/unit,
                  rainfall.cpu())

    r2 = r2_score(event_data[1].reshape(-1).numpy(),
                  event_data[0].reshape(-1).numpy()
                  )
    print("r2:%.6f" % (r2))
    logging.info("%s r2:%.6f" % (task, r2))

    if task == "reg":
        title = "reg:Water Depth (m) - R2=%.4f" % r2
        """获取积水最深的点的内涝曲线"""
        vis_max_depth(event_data, title, save_fig_dir)

    elif task == "cls":
        title = "cls:Flood-Unflood"

    """获取内涝事件的时空变化图"""
    # if task == "reg":
    #     vis_event_dynamic(event_data, title, save_fig_dir)
    # vis_event_dynamic_gif(event_data, title, save_fig_dir)

    # """获取降雨-内涝事件的损失"""
    # exp_all_event_loss = {"event_reg_error": test_losses}
    # vis_event_loss(exp_all_event_loss, 0, save_fig_dir)

    # 散点图，图上有R2
    # r2 = r2_score(event_data[1].reshape(-1).numpy(),
    #                 event_data[0].reshape(-1).numpy()
    #                 )
    # print("r2:%.6f" % (r2))
    # vis_scatter(event_data, save_fig_dir)

    """区域内的总水深曲线对比图"""
    predsum, labelsum, xlabel_title = \
    vis_total_volumn(event_data, title, save_fig_dir)

    """upload to wandb"""
    if upload:
        # xs = [i for i in range(len(pred))]
        # ys = [predsum, labelsum]
        # wandb.log({"wandb.Table": wandb.plot.line_series(
        #     xs=xs, ys=ys,
        #     keys=["pred", "label"],
        #     title=title,
        #     xname=xlabel_title,
        # )})

        wandb.log({"R2-%s" % task: r2})


if __name__ == "__main__":
    exp_dir = "exp/2022-12-09/m_to_mm_drop_drainge_and_flow"
    # exp_dir = "exp/2022-12-08/correction_unit_win"
    # exp_dir = "exp/2022-12-08/overfit_test"
    save_root = "results/"
    save_dir = os.path.join(save_root, exp_dir)
    save_fig_dir = os.path.join(save_dir, "figs")
    if not os.path.exists(save_fig_dir):
        os.makedirs(save_fig_dir, exist_ok=True)

    batch_id, mode = 0, "test"
    start_epoch = 0

    for reduction in ["sum", ]:
        # for reduction in ["sum", "mean"]:
        exp_all_train_loss, exp_all_event_loss = {}, {}

        for loss_name in ["Focal_MSE"]:
            # for loss_name in ["MSE", "Focal_MSE", "Focal_MSE_BCE", "WMSE"]:
            # if loss_name == "WMSE":
            #     # exp_dir = "exp/2022-12-08/m_to_mm_WMSE"
            #     continue
            # else:
            #     exp_dir = "exp/2022-12-08/overfit_test"
            # exp_all_train_loss = get_train_loss(exp_dir, exp_all_train_loss)

            # """获取降雨-内涝事件的损失"""
            # event_loss = get_event_loss(
            #     exp_dir, batch_id, mode, reduction, loss_name)
            # exp_all_event_loss[loss_name+"-"+reduction] = event_loss

            """获取内涝事件的时空变化图"""
            event_data = get_event_maps(exp_dir, batch_id)
            vis_event_dynamic(event_data, save_fig_dir)
            # vis_event_dynamic(event_data, save_fig_dir, reduction, loss_name)

        # save_fig_dir = "./"
        # vis_train_loss(exp_all_train_loss, start_epoch, save_fig_dir)
        # vis_event_loss(exp_all_event_loss, start_epoch, save_fig_dir)
