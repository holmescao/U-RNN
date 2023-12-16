import numpy as np
from skimage.metrics import (
    peak_signal_noise_ratio,
    mean_squared_error,
    structural_similarity,
)


def euclidean_distance(point1, point2):
    """
    计算两点之间的欧式距离

    参数：
    point1: 第一个点的坐标，格式为 (x1, y1, z1, ...)
    point2: 第二个点的坐标，格式为 (x2, y2, z2, ...)

    返回值：
    两点之间的欧式距离
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    distance = np.linalg.norm(point1 - point2)
    return distance


def cal_peak_signal_noise_ratio(label, pred, data_range, axis):
    if axis == (1, 2):
        # 计算每一时刻的psnr
        psnr = []
        for i in range(label.shape[0]):
            psnr.append(
                peak_signal_noise_ratio(
                    label[i], pred[i], data_range=data_range)
            )
        psnr = np.array(psnr)
    elif axis == None:
        # 计算整体psnr
        psnr = peak_signal_noise_ratio(label, pred, data_range=data_range)
    elif axis == 0:
        psnr = None

    return psnr


def cal_structural_similarity(label, pred, data_range, axis):
    if axis == (1, 2):
        # 计算每一时刻的ss
        ss = []
        for i in range(label.shape[0]):
            ss.append(structural_similarity(
                label[i], pred[i], data_range=data_range))
        ss = np.array(ss)
    elif axis == None:
        # 计算整体ss
        ss = structural_similarity(label, pred, data_range=data_range)
    elif axis == 0:
        ss = None

    return ss


def cal_max_time_AE(pred, label, axis):
    if axis == 0:
        pred_offset_idx = np.argmax(pred, axis)
        label_offset_idx = np.argmax(label, axis)
        # 最值点的时间偏移
        MaxTAE = pred_offset_idx - label_offset_idx
    else:
        MaxTAE = None

    return MaxTAE


def cal_max_position_AE(pred, label, axis):
    if axis is None:
        pred_max_idx = np.unravel_index(np.argmax(pred), pred.shape)[1:]
        label_max_idx = np.unravel_index(np.argmax(label), label.shape)[1:]
        MaxPAE = euclidean_distance(pred_max_idx, label_max_idx)
    else:
        MaxPAE = None
    return MaxPAE


def cal_max_value_AE(pred, label, axis):
    return np.max(pred, axis) - np.max(label, axis)


def cal_max_AE(pred, label, axis):
    error = pred - label
    ae = np.abs(error)
    return np.max(ae, axis)


def cal_MAE(pred, label, axis):
    return np.mean(np.abs(pred - label), axis)


def cal_RMSE(pred, label, axis):
    return np.sqrt(np.mean((pred - label) ** 2, axis))



def calculate_error(pred, label, axis=(0, 1, 2)):
    """
    计算预测结果和真实标签之间的误差指标

    参数：
    pred: 预测结果矩阵，shape为(时序长度, 空间长度, 空间长度)
    label: 真实标签矩阵，shape为(时序长度, 空间长度, 空间长度)
    axis: 指定要计算误差的维度，可选值为(0, 1, 2)，默认计算所有维度
        # axis=0: 得到空间的结果(500, 500)
        # axis=(1,2)：得到时间的结果(360,)
        # axis=None：得到整体结果，1个值(1,)

    返回值：
    一个包含各个误差指标的字典，包括峰值误差、最值点的位置偏移、MAE、MSE、SSIM、PSNR、误差的最大值
    """
    data_range = np.max(label) - np.min(label)

    errors = {}

    # 峰值误差
    errors["MaxVAE"] = cal_max_value_AE(pred, label, axis)

    # 每个位置的峰值时间偏移
    errors["MaxTAE"] = cal_max_time_AE(pred, label, axis)

    # 最值点的位置偏移
    errors["MaxPAE"] = cal_max_position_AE(pred, label, axis)

    # MAE
    errors["MAE"] = cal_MAE(pred, label, axis)

    # RMSE
    errors["RMSE"] = cal_RMSE(pred, label, axis)

    # SSIM
    errors["SSIM"] = cal_structural_similarity(label, pred, data_range, axis)

    # PSNR
    errors["PSNR"] = cal_peak_signal_noise_ratio(label, pred, data_range, axis)

    # # 误差的最大值
    # errors["MaxAE"] = cal_max_AE(pred, label, axis)

    return errors
