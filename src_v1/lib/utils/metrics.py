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


def cal_max_time_AE(pred, label, axis,threshold=0.15):
    
    if axis == 0:
        # 在时间维度上找到每个位置的最大值
        # pred_max = np.max(pred, axis=0)
        label_max = np.max(label, axis=0)

        # 判断最大值是否大于阈值
        valid_mask = (label_max >= threshold)
        pred_offset_idx = np.argmax(pred, axis)
        label_offset_idx = np.argmax(label, axis)
        # 最值点的时间偏移
        MaxTAE = np.abs(pred_offset_idx - label_offset_idx)
        
        MaxTAE[~valid_mask] = 0
    else:
        MaxTAE = None

    return MaxTAE


def cal_max_position_AE(pred, label,axis,threshold=0.15):
    """
    计算pred和label在每个时刻的峰值位置的欧式距离。
    :param pred: 预测矩阵，形状为(T, H, W)
    :param label: 标签矩阵，形状为(T, H, W)
    :return: 每个时刻峰值位置的欧式距离列表
    """
    distances = []
    for t in range(label.shape[0]):
        # 找到每个矩阵在时刻t的峰值位置
        pred_peak_pos = np.unravel_index(np.argmax(pred[t]), pred[t].shape)
        label_peak_pos = np.unravel_index(np.argmax(label[t]), label[t].shape)
        
        label_max = np.max(label[t])

        if label_max >= threshold:
            # 找到每个矩阵在时刻t的峰值位置
            pred_peak_pos = np.unravel_index(np.argmax(pred[t]), pred[t].shape)
            label_peak_pos = np.unravel_index(np.argmax(label[t]), label[t].shape)

            # 计算欧式距离
            distance = np.linalg.norm(np.array(pred_peak_pos) - np.array(label_peak_pos))
        else:
            distance = 0
        distances.append(distance)

    distances = np.array(distances)
    
    if not isinstance(axis,tuple):
        distances = None
        
    return distances

def cal_max_value_AE(pred, label, axis):
    return np.abs(np.max(pred, axis) - np.max(label, axis))


def calculate_duration_error(pred, label, axis,threshold=0.15, min_duration=30, time_interval=1):
    """
    计算预测和实际数据中大于指定阈值的持续时间误差，并计算平均误差。
    :param pred: 预测矩阵，形状为(T, H, W)
    :param label: 实际矩阵，形状为(T, H, W)
    :param threshold: 内涝判定阈值（默认0.15米）
    :param min_duration: 最小持续时间（分钟）（默认30分钟）
    :param time_interval: 时间间隔（分钟）（默认1分钟）
    :return: 所有位置的平均持续时间误差
    """
    def calculate_duration(matrix):
        """
        计算大于阈值的持续时间
        :param matrix: 输入矩阵，形状为(T, H, W)
        :return: 每个空间位置的持续时间，形状为(H, W)
        """
        # 生成大于阈值的布尔矩阵
        greater_than_threshold = (matrix >= 0.15).astype(int)

        # 计算累积和
        cumsum = np.cumsum(greater_than_threshold, axis=0)

        # 重置累积和的值
        reset_points = np.roll(greater_than_threshold, shift=1, axis=0)
        reset_points[0, :, :] = 0  # 确保第一个时间点不被重置
        cumsum_reset = np.cumsum(greater_than_threshold * (greater_than_threshold - reset_points < 0), axis=0)

        # 计算持续时间
        duration = cumsum - cumsum_reset

        # 获取最后一个时间点的持续时间作为每个空间位置的持续时长
        duration_at_last_time_point = duration[-1, :, :]

        return duration_at_last_time_point

    if axis==0:
        pred_duration = calculate_duration(pred)
        label_duration = calculate_duration(label)

        # 计算持续时间差
        duration_error = np.abs(pred_duration - label_duration) * time_interval

        # 计算平均误差
        # 沿着时间维度判断是否存在大于阈值的数值
        flood_locations = np.any(label >= threshold, axis=0)
        average_error = duration_error
        # average_error = duration_error[flood_locations]
        # average_error = np.mean(duration_error)
    else:
        average_error = None
        
    return average_error


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
    
    errors["FDAE"] = calculate_duration_error(pred, label, axis)

    return errors
