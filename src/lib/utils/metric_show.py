import matplotlib.font_manager as fm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.animation as animation
import seaborn as sns
import os
from matplotlib import rcParams

sns.set_palette("Set2")

import numpy as np


def cal_max_error_position(pred, label):
    # 计算绝对误差
    absolute_error = np.abs(pred - label)

    # 计算每个时刻的误差最大值
    max_error_per_timestep = np.max(absolute_error, axis=(1, 2))

    # 找到误差最大值所在的时刻
    max_error_timestep = np.argmax(max_error_per_timestep)

    return max_error_timestep


def cal_Error(pred, label):
    return pred - label


def save_curve(key, values, save_dir):
    sns.set_style("whitegrid")  # Set the style to whitegrid
    sns.set_context("paper")  # Set the context to paper
    sns.set(style="ticks", font_scale=1.2)

    # Font settings
    fs = 25
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Times New Roman"] + rcParams["font.serif"]
    rcParams["font.size"] = fs

    fig, ax = plt.subplots(figsize=(8, 6))
    # ax.plot(values1, label='Baseline')
    ax.plot(
        values, label="Current Best", lw=2, linestyle="--", color=sns.color_palette()[1]
    )
    ax.set_xlabel("Epoch", fontweight="bold", fontsize=fs)
    ax.set_ylabel("Value", fontweight="bold", fontsize=fs)
    ax.set_title(key, fontweight="bold", fontsize=fs + 2)
    ax.tick_params(axis="both", labelsize=fs - 2)
    ax.legend(loc="upper right", fontsize=fs)

    sns.despine()
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"vs_{key}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_comparison_curve(key, values1, values2, save_dir):
    sns.set_style("whitegrid")  # Set the style to whitegrid
    sns.set_context("paper")  # Set the context to paper
    sns.set(style="ticks", font_scale=1.2)

    # Font settings
    fs = 25
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Times New Roman"] + rcParams["font.serif"]
    rcParams["font.size"] = fs

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(values1, label="Baseline", lw=2, color=sns.color_palette()[0])
    ax.plot(
        values2,
        label="Current Best",
        lw=2,
        linestyle="--",
        color=sns.color_palette()[1],
    )
    ax.set_xlabel("Epoch", fontweight="bold", fontsize=fs)
    ax.set_ylabel("Value", fontweight="bold", fontsize=fs)
    ax.set_title(key, fontweight="bold", fontsize=fs + 2)
    ax.tick_params(axis="both", labelsize=fs - 2)
    ax.legend(loc="upper right", fontsize=fs)

    sns.despine()
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"vs_{key}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


def extract_log_info(baseline_logs):
    baseline_info = {}
    for line in baseline_logs:
        line_info = extract_line_values(line)

        if baseline_info != {}:
            for k, v in line_info.items():
                baseline_info[k].append(v)
        else:
            for k, v in line_info.items():
                baseline_info[k] = [v]

    return baseline_info


def extract_line_values(log_line):
    data = {}
    fields = log_line.split("|")
    for field in fields:
        key_value = field.strip().split(":")
        if len(key_value) == 2:
            key = key_value[0].strip()
            value = key_value[1].strip().split()[0]
            data[key] = float(value)
    return data


def load_arrays_from_npz(sample_id, exp_dir, key="huv", unit=1000):
    pred_huv_save_path = os.path.join(exp_dir, f"pred_{sample_id}.npz")
    label_huv_save_path = os.path.join(exp_dir, f"label_{sample_id}.npz")
    # 加载.npz文件
    pred_huv = np.load(pred_huv_save_path)
    label_huv = np.load(label_huv_save_path)

    # 读取数组
    pred = []
    label = []
    for k in key:
        pred.append(np.squeeze(pred_huv["pred_%s" % k]) / unit)
        label.append(np.squeeze(label_huv["label_%s" % k]) / unit)

    return pred, label


def vis_error(pred, label, error, key="h", unit="mm", frame=100, cmap=None, save_dir=None):
    # 计算数据的最大值和最小值
    vmin = np.min([np.min(pred), np.min(label)])
    vmax = np.max([np.max(pred), np.max(label)])
    # vmax = np.max([np.abs(vmin), np.abs(vmax)])

    # error的最大和最小值
    # error = pred - label
    ab_error = np.abs(error)

    max_ab_error = np.max(ab_error)
    min_ab_error = np.min(ab_error)
    min_error = np.min(error)
    max_error = np.max(error)

    print("ab_error: [%.2f,%.2f]" % (min_ab_error, max_ab_error))
    print("error: [%.2f,%.2f]" % (min_error, max_error))

    # emax = np.max([np.abs(min_error), np.abs(max_error)])

    # 设置图形布局
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    fig.subplots_adjust(wspace=0.3)  # 调整子图间的间距

    # 创建一个文本注释对象
    text_annotation = axs[0].text(
        0.05, 1.2, "", transform=axs[0].transAxes, fontsize=15, color="blue"
    )

    # 创建空白的热力图对象
    cmap.set_bad(color="white")  # 将数值为0的颜色设置为白色
    pred_f = pred.copy()
    label_f = label.copy()
    error_f = error.copy()
    ab_error_f = ab_error.copy()
    pred_f[pred_f == 0] = np.nan
    label_f[label_f == 0] = np.nan
    error_f[error_f == 0] = np.nan
    ab_error_f[ab_error_f == 0] = np.nan
    im1 = axs[0].imshow(pred_f, cmap=cmap, vmin=vmin, vmax=vmax)
    im2 = axs[1].imshow(label_f, cmap=cmap, vmin=vmin, vmax=vmax)
    im3 = axs[2].imshow(error_f, cmap=cmap, vmin=min_error, vmax=max_error)
    im4 = axs[3].imshow(ab_error_f, cmap=cmap, vmin=min_ab_error, vmax=max_ab_error)
    axs[0].set_title("Pred_%s" % key)
    axs[1].set_title("Label")
    axs[2].set_title("Error_%s" % key)
    axs[3].set_title("AE_%s" % key)

    cbar = fig.colorbar(im2, ax=[axs[0], axs[1]], location="right", shrink=0.7)
    cbar.set_label(unit)
    cbar_ax = cbar.ax

    cbar_2 = fig.colorbar(im3, ax=[axs[2]], location="right", shrink=0.7)
    cbar_2.set_label(unit)
    cbar_ax = cbar_2.ax

    cbar_3 = fig.colorbar(im4, ax=[axs[3]], location="right", shrink=0.7)
    cbar_3.set_label(unit)
    cbar_ax = cbar_3.ax

    text_annotation.set_text("Frame: %d" % (frame))
    # 将文本注释对象添加到图形中
    fig.artists.append(text_annotation)
    # save
    save_path = os.path.join(save_dir, "vs_%s_frame@%d_error.png" % (key, frame))
    plt.savefig(save_path, dpi=300)
    plt.close()


def create_animation(
    pred, label, key="h", interval=50, save_dir=None, share_colorbar=True
):
    # 计算数据的最大值和最小值
    vmin = np.min([np.min(pred), np.min(label)])
    vmax = np.max([np.max(pred), np.max(label)])
    # error的最大和最小值
    error = pred - label
    min_error = np.min(error)
    max_error = np.max(error)

    # 设置图形布局
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.subplots_adjust(wspace=0.2)  # 调整子图间的间距

    # 创建一个文本注释对象
    text_annotation = axs[0].text(
        0.05, 1.2, "", transform=axs[0].transAxes, fontsize=15, color="blue"
    )
    # 创建空白的热力图对象
    cmap = plt.cm.jet
    cmap.set_bad(color="white")  # 将数值为0的颜色设置为白色
    pred_f = pred[0].copy()
    label_f = label[0].copy()
    pred_f[pred_f == 0] = np.nan
    label_f[label_f == 0] = np.nan
    im1 = axs[0].imshow(pred_f, cmap=cmap, vmin=vmin, vmax=vmax)
    im2 = axs[1].imshow(label_f, cmap=cmap, vmin=vmin, vmax=vmax)
    # im1 = axs[0].imshow(pred[0], cmap=cmap, vmin=vmin, vmax=vmax)
    # im2 = axs[1].imshow(label[0], cmap=cmap, vmin=vmin, vmax=vmax)
    im3 = axs[2].imshow(pred[0] - label[0], cmap="bwr", vmin=min_error, vmax=max_error)

    cbar = fig.colorbar(im2, ax=[axs[0], axs[1]], location="right", shrink=0.7)
    cbar.set_label("m")
    cbar_ax = cbar.ax

    cbar_2 = fig.colorbar(im3, ax=[axs[2]], location="right", shrink=0.7)
    cbar_2.set_label("m")
    cbar_ax = cbar_2.ax

    intc = 5
    # 设置动态图更新函数

    def update(frame):
        axs[0].cla()  # 清空子图1
        axs[1].cla()  # 清空子图2
        axs[2].cla()  # 清空子图3

        pred_f = pred[intc * frame].copy()
        label_f = label[intc * frame].copy()
        pred_f[pred_f == 0] = np.nan
        label_f[label_f == 0] = np.nan
        # 绘制子图1：pred的热力图
        axs[0].imshow(pred_f, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[0].set_title("Pred")
        # axs[0].set_title("Pred_%s" % key)

        # 绘制子图2：label的热力图
        axs[1].imshow(label_f, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[1].set_title("Label")

        # 计算pred与label对应位置的误差
        error = pred[intc * frame] - label[intc * frame]

        # 绘制子图3：绝对误差图
        axs[2].imshow(error, cmap="bwr", vmin=min_error, vmax=max_error)
        axs[2].set_title("Error_%s" % key)

        # 更新文本注释的数值
        text_annotation.set_text("Frame: %d" % (intc * frame))

    # 设置动画
    ani = animation.FuncAnimation(
        fig, update, frames=pred.shape[0] // intc, interval=interval, blit=False
    )

    # 将文本注释对象添加到图形中
    fig.artists.append(text_annotation)

    save_path = os.path.join(save_dir, "vs_%s_dynamic_error.gif" % key)
    if save_path:
        # 保存动画为GIF文件
        ani.save(save_path, writer="pillow", dpi=300)

    # # 显示动画
    # plt.show()
    plt.close()


def calculate_r2(pred, label):
    # 计算总平方和
    total_sum_of_squares = np.sum((label - np.mean(label)) ** 2, axis=0)

    # 计算残差平方和
    residual_sum_of_squares = np.sum((label - pred) ** 2, axis=0)

    # 计算决定系数
    r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)

    return r2


def calculate_mse(pred, label):
    # 计算均方误差
    mse = np.mean((pred - label) ** 2, axis=0)

    return mse


def calculate_mae(pred, label):
    # 计算平均绝对误差
    mae = np.mean(np.abs(pred - label), axis=0)

    return mae


def visualize_2d_metric(data, vmax, vmin, metric_name, key, save_dir):
    # 绘制均方误差图像
    plt.imshow(data, cmap="bwr", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(metric_name + "-" + key)

    save_path = os.path.join(save_dir, "vs_%s_%s_.png" % (key, metric_name))
    plt.savefig(save_path, dpi=300)
    plt.close()


def vs_visualize_2d_metric(data1, data2, vmax, vmin, metric_name, key1, key2, save_dir):
    # 创建图形布局
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(wspace=0.2)  # 调整子图间的间距

    # 绘制第一个子图
    im1 = axs[0].imshow(data1, cmap="bwr", vmin=vmin, vmax=vmax)
    axs[0].set_title(key1)

    # 绘制第二个子图
    if data2 is not None:
        im2 = axs[1].imshow(data2, cmap="bwr", vmin=vmin, vmax=vmax)
        axs[1].set_title(key2)

    # 添加共享的颜色条
    cbar = fig.colorbar(im1, ax=axs, location="right", shrink=0.7)
    cbar.set_label(metric_name)

    # # 设置图形标题
    # plt.suptitle(metric_name1 + " vs " + metric_name2 + " - " + key)

    # 保存图像
    save_path = os.path.join(save_dir, "vs_%s_%s_%s.png" % (key1, key2, metric_name))
    plt.savefig(save_path, dpi=300)
    plt.close()


def vs_visualize_metric_space(data_list, name, key_list, vmax, vmin, save_dir):
    n = len(data_list)
    # 创建图形布局
    fig, axs = plt.subplots(1, n, figsize=(6 * n, 6))
    fig.subplots_adjust(wspace=0.2)  # 调整子图间的间距

    for i in range(n):
        # 绘制第一个子图
        im1 = axs[i].imshow(data_list[i], cmap="bwr", vmin=vmin, vmax=vmax)
        axs[i].set_title(key_list[i])

    # 添加共享的颜色条
    cbar = fig.colorbar(im1, ax=axs, location="right", shrink=0.7)
    cbar.set_label(name)

    # # 设置图形标题
    # plt.suptitle(metric_name1 + " vs " + metric_name2 + " - " + key)

    # 保存图像
    save_path = os.path.join(save_dir, "vs_%s.png" % (name))
    plt.savefig(save_path, dpi=100)
    plt.close()


def density_scatter_plot(x, y, key, save_dir):
    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy, bw_method=0.2)(xy)  # 设置bw_method参数控制核函数带宽

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    # Determine the axis limits
    max_val = np.max([np.max(x), np.max(y)])
    min_val = np.min([np.min(x), np.min(y)])

    # Create the scatter plot
    fig, ax = plt.subplots()
    sc = ax.scatter(x, y, c=z, s=10, cmap="jet")
    ax.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--")

    # Fit a linear regression model
    reg = LinearRegression()
    reg.fit(x.reshape(-1, 1), y)
    x_range = np.linspace(min_val, max_val, 100)
    y_pred = reg.predict(x_range.reshape(-1, 1))
    ax.plot(x_range, y_pred, color="black")

    # Calculate R2 score
    r2 = r2_score(y, reg.predict(x.reshape(-1, 1)))

    ax.set_aspect("equal")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel("Pred")
    ax.set_ylabel("Label")
    ax.set_title("%s (R2 = %.2f)" % (key, r2))

    # Create a colorbar using the scatter plot's color map
    cbar = plt.colorbar(sc)
    # cbar = plt.colorbar(sc, ticks=np.linspace(0, 0.5, 6), extend='both')
    cbar.set_label("Density")

    # Add grid lines
    plt.grid(True, linestyle="--", alpha=0.4)

    save_path = os.path.join(save_dir, "%s_density_scatter_plot.png" % (key,))
    plt.savefig(save_path, dpi=300)
    # plt.show()


def plot_error_histogram(pred, label, key, xlim=None, ylim=None, save_dir=None):
    # 计算误差
    error = pred - label

    # 绘制直方图
    fig, ax = plt.subplots()
    _, bins, _ = ax.hist(error, bins=300, edgecolor="black")

    ax.set_xlabel("Error")
    ax.set_ylabel("Frequency (log scale)")
    ax.set_yscale("log")
    ax.set_title("Frequency Distribution Histogram - %s" % key)

    # 设置横坐标刻度范围
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    save_path = os.path.join(save_dir, "hist_%s.png" % key)
    plt.savefig(save_path, dpi=300)


def vs_plot_error_histogram(
    pred1, pred2, label, key, xlim=None, ylim=None, save_dir=None
):
    # 计算误差
    error1 = pred1 - label
    error2 = pred2 - label

    # 绘制直方图
    fig, ax = plt.subplots()
    _, bins, _ = ax.hist(
        [error1, error2],
        bins=300,
        edgecolor="none",
        color=["blue", "red"],
        label=["Baseline", "Best"],
    )

    ax.set_xlabel("Error (m)")
    ax.set_ylabel("Frequency (log scale)")
    ax.set_yscale("log")
    ax.set_title("vs. Frequency Distribution Histogram-%s" % (key))
    ax.legend()

    # 设置横坐标刻度范围
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    save_path = os.path.join(save_dir, "vs_hist-%s.png" % (key))
    plt.savefig(save_path, dpi=300)


def calculate_grouped_mae(pred, label, vmin, vmax, intc):
    # 计算分组数量
    groups = int((vmax - vmin) // intc) + 1

    # 计算每个值的分组
    grouped_indices = np.floor_divide(label, intc).astype(int)

    # 计算分组MAE
    grouped_mae = np.zeros(groups)
    count = np.bincount(grouped_indices, minlength=groups)

    np.add.at(grouped_mae, grouped_indices, np.abs(pred - label))

    # 计算平均分组MAE
    grouped_mae /= count
    grouped_mae = np.nan_to_num(grouped_mae)

    return grouped_mae


# def calculate_grouped_mae(pred, label, vmin, vmax, intc):
#     # 计算分组数量
#     groups = int((vmax - vmin) // intc)

#     # 计算每个值的分组
#     grouped_indices = np.floor_divide(label, intc).astype(int)

#     # 创建分组列表
#     grouped_pred = [np.array([]) for _ in range(groups)]
#     for i, idx in enumerate(grouped_indices):
#         grouped_pred[idx] = np.append(grouped_pred[idx], pred[i])

#     # 计算分组MAE
#     grouped_mae = np.array([np.mean(
#         np.abs(grouped_pred[i] - label[grouped_indices == i])) for i in range(groups)])
#     grouped_mae = np.nan_to_num(grouped_mae)

#     return grouped_mae


def visualize_grouped_mae(grouped_mae, intc, key, save_dir):
    groups = len(grouped_mae)
    x_ticks = np.arange(0, (groups * intc), intc)

    bar_width = intc * 0.8  # 柱状图宽度调整为间隔的80%
    plt.figure(figsize=(20, 6))
    # plt.bar(x_ticks, grouped_mae,width=20)
    plt.bar(x_ticks, grouped_mae, width=bar_width)
    # plt.bar(range(groups), grouped_mae)
    plt.xlabel("Group", fontsize=15)
    plt.ylabel("MAE", fontsize=15)
    plt.title("Grouped MAE", fontsize=15)
    plt.xticks(x_ticks, rotation=90, fontsize=15)  # 将刻度值旋转90度
    plt.yscale("log")  # 设置纵坐标为对数比例

    # 调整坐标轴标签的字体大小
    plt.tick_params(axis="both", which="major", labelsize=12)
    # plt.show()
    save_path = os.path.join(save_dir, "group_bar_%s.png" % key)
    plt.savefig(save_path, dpi=300)


def vs_visualize_grouped_mae(grouped_mae1, grouped_mae2, intc, key, save_dir):
    groups = len(grouped_mae1)
    x_ticks = np.arange(0, (groups * intc), intc)

    bar_width = intc * 0.35  # 柱状图宽度调整为间隔的35%
    plt.figure(figsize=(12, 6))
    plt.bar(x_ticks, grouped_mae1, width=bar_width, color="blue", label="Baseline")
    plt.bar(
        x_ticks, grouped_mae2, width=bar_width, color="red", alpha=0.7, label="Best"
    )
    plt.xlabel("Group: water depth (mm)", fontsize=15)
    plt.ylabel("MAE", fontsize=15)
    plt.title("vs Grouped MAE-%s" % key, fontsize=15)
    plt.xticks(x_ticks, rotation=90, fontsize=12)
    plt.yscale("log")

    # 调整坐标轴标签的字体大小
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.legend()

    save_path = os.path.join(save_dir, "vs_group_bar-%s.png" % key)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


def vis_group_error(pred, label, vmin, vmax, intc, key, save_dir):
    # 计算分组MAE
    grouped_mae = calculate_grouped_mae(pred, label, vmin, vmax, intc)
    # 可视化
    visualize_grouped_mae(grouped_mae, intc, key, save_dir)


def vs_vis_group_error(pred1, pred2, label, key, vmin, vmax, intc, save_dir):
    # 计算分组MAE
    grouped_mae1 = calculate_grouped_mae(pred1, label, vmin, vmax, intc)
    grouped_mae2 = calculate_grouped_mae(pred2, label, vmin, vmax, intc)
    # 可视化
    vs_visualize_grouped_mae(grouped_mae1, grouped_mae2, intc, key, save_dir)


def vis_vs_mean_std(
    data, mean, std, name, save_dir, data_type=["pred", "label"], fontsize=20
):
    plt.figure(figsize=(8, 12))
    # Set fontsize for better readability
    plt.rcParams["font.size"] = fontsize

    # Define data types

    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot mean and fill between with std for each data type
    for i in range(len(data)):
        time = np.arange(data[i].shape[0])  # Create time array
        ax.plot(time, data[i], label="%s" % data_type[i])
        if mean is not None:
            ax.fill_between(
                time,
                mean[i] - std[i],
                mean[i] + std[i],
                alpha=0.2,
                label="%s_std" % data_type[i],
            )

    # Set x and y labels
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("m")

    # Set title with MSE values
    title = name
    ax.set_title(title)

    # Add legend
    ax.legend(fontsize=fontsize)
    # ax.legend(fontsize=fontsize, bbox_to_anchor=(1.6, 1))

    # Save the plot
    save_path = os.path.join(save_dir, "vs_res_%s.jpg" % (name))
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# 递归函数判断字典中是否存在某个键
def check_key_exists(key, dictionary):
    if key in dictionary:
        return True
    for value in dictionary.values():
        if isinstance(value, dict):
            if check_key_exists(key, value):
                return True
    return False
