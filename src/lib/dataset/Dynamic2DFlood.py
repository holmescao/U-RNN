from scipy.ndimage import gaussian_filter1d
import numpy as np
import os
import torch
import torch.utils.data as data
import random


def split_and_merge_datasets(events_list):
    # 设置随机种子以确保可重复性
    random.seed(0)

    # 根据名字前缀分割实测降雨和设计降雨
    real_rainfall = [event for event in events_list if event.startswith("G")]
    design_rainfall = [event for event in events_list if event.startswith("r")]

    # 随机选择实测降雨的21场和设计降雨的12场作为训练集
    train_real = random.sample(real_rainfall, 21)
    train_design = random.sample(design_rainfall, 12)

    # 剩余的场次作为测试集
    test_real = [rainfall for rainfall in real_rainfall if rainfall not in train_real]
    test_design = [rainfall for rainfall in design_rainfall if rainfall not in train_design]

    # 合并训练集和测试集
    train_set = train_real + train_design
    test_set = test_real + test_design

    return train_set, test_set


class Dynamic2DFlood(data.Dataset):
    def __init__(
        self,
        data_root,
        split,
    ):
        super(Dynamic2DFlood, self).__init__()

        self.data_root = data_root
        self.data_dir = os.path.join(data_root, "train")
        # self.data_dir = os.path.join(data_root,
        #                              "train" if "train" in split else "test")
        # self.data_dir = os.path.join(data_root,
        #                              "train_small" if "train" in split else "train_small")
        
        self.geo_root = os.path.join(self.data_dir, "geodata")
        self.flood_root = os.path.join(self.data_dir, "flood")
        self.locations = sorted(os.listdir(self.flood_root), key=lambda x: int(''.join(filter(str.isdigit, x))))

        self.locations_dir = [os.path.join(self.flood_root, loc) for loc in self.locations]
        self.event_names = sorted(os.listdir(self.locations_dir[0]))

        if "train" in split:
            train_data = []
            script_dir = os.path.dirname(os.path.abspath(__file__))
            with open(f'{script_dir}/train.txt', 'r') as train_file:
                for line in train_file:
                    train_data.append(line.strip())
            self.event_names = train_data
        elif "test" in split:
            test_data = []
            script_dir = os.path.dirname(os.path.abspath(__file__))
            with open(f'{script_dir}/test.txt', 'r') as train_file:
                for line in train_file:
                    test_data.append(line.strip())
            self.event_names = test_data
        
        print(self.event_names)

        self.num_samples = len(self.event_names) * len(self.locations)

        print(f"Loaded Dynamic2DFlood {split} {self.num_samples} samples (locations:{len(self.locations)}, event nums:{len(self.event_names)})")

    def _load_event(self, index):
        event_data = dict()

        event_id = index // len(self.locations)
        loc_id = index % len(self.locations)
        
        # flood, rainfall
        event_dir = os.path.join(self.flood_root, 
                                 self.locations[loc_id],
                                 self.event_names[event_id])
        
        for attr_file in os.listdir(event_dir):
            if ".jpg" in attr_file:
                continue
            attr_name = attr_file.split(".")[0]
            attr_file_path = os.path.join(event_dir, attr_file)
            attr_data = np.load(attr_file_path, allow_pickle=True)
            event_data[attr_name] = attr_data
        # dem, impervious, manhole
        geo_dir = os.path.join(self.geo_root,self.locations[loc_id])
        for attr_file in os.listdir(geo_dir):
            attr_name = attr_file.split(".")[0]
            attr_file_path = os.path.join(geo_dir, attr_file)
            if os.path.isdir(attr_file_path):
                continue
            attr_data = np.load(attr_file_path, allow_pickle=True)
            event_data[attr_name] = attr_data

        # print(event_dir)
        return event_data, event_dir

    def _prepare_input(self, event_data,event_dir, duration=360):
        """
        按以下顺序（左->右，上->下）拼接input
        地理因素: absolute_DEM, flow_direction, flow_speed
        排水因素: impervious, manhole
        降雨因素: rainfall


        数据维度：(Frames, C, H, W)
        """
        # 提取输入数据
        absolute_DEM = torch.from_numpy(event_data["absolute_DEM"]).float()
        impervious = torch.from_numpy(event_data["impervious"]).float()
        manhole = torch.from_numpy(event_data["manhole"]).float()
        
        # # Add: 降雨数据平滑处理
        window_size=30
        # smoothed_rainfall = np.convolve(event_data["rainfall"], np.ones(window_size)/window_size, mode='same')
        # rainfall = torch.from_numpy(smoothed_rainfall).float()
        
        # rainfall = gaussian_filter1d(event_data["rainfall"], sigma=2)
        # rainfall = torch.from_numpy(rainfall).float()
        rainfall = torch.from_numpy(event_data["rainfall"]).float()
        
        """补0操作"""
        # fill_zeors_nums = 0
        # rainfall = pad_zeros_to_tensor(rainfall, fill_zeors_nums)
        
        if len(rainfall) != duration:
            # Extend the tensor to length 360
            padding_length = duration - len(rainfall)
            rainfall = torch.nn.functional.pad(rainfall, (0, padding_length), "constant", 0)            

        # 累计降雨
        cumsum_rainfall = torch.cumsum(rainfall,dim=0)
        
        # 考虑排水
        # rainfall -= 50/60 # mm/min
        # rainfall[rainfall<0] = 0

        # import matplotlib.pyplot as plt

        # event_name = event_dir.split("/")[-1]
        # plt.plot(rainfall.numpy())
        # plt.savefig(f"figs/smth_drainage/{event_name}.jpg",dpi=100)
        # plt.close()

        # # 处理数据维度
        absolute_DEM = absolute_DEM[None, None, :, :]
        impervious = impervious[None, None, :, :]
        manhole = manhole[None, None, :, :]
        rainfall = rainfall[:, None, None, None]
        cumsum_rainfall = cumsum_rainfall[:, None, None, None]
        
        absolute_DEM = absolute_DEM * 1000  # m -> mm
        max_DEM = absolute_DEM.max()
        min_DEM = absolute_DEM.min()

        # # !为了保证整除，需要对height从385变为384
        # absolute_DEM = absolute_DEM[None, None, 1:, :]
        # impervious = impervious[None, None, 1:, :]
        # manhole = manhole[None, None, 1:, :]
        # rainfall = rainfall[:, None, None, None]

        # # ! expr: 只关注内涝点周围300x300区域
        # absolute_DEM = absolute_DEM[:, :, 20:20 + 300, 200:200 + 300]
        # impervious = impervious[:, :, 20:20 + 300, 200:200 + 300]
        # manhole = manhole[:, :, 20:20 + 300, 200:200 + 300]
        

        # # ! expr: 只关注内涝点周围64x64区域
        # absolute_DEM = absolute_DEM[:, :, 340:340+64, 320:320+64]
        # impervious = impervious[:, :, 340:340+64, 320:320+64]
        # manhole = manhole[:, :, 340:340+64, 320:320+64]
        

        return {
            "absolute_DEM": absolute_DEM, # mm
            "max_DEM":max_DEM,  # mm
            "min_DEM":min_DEM,  # mm
            "impervious": impervious,  # %
            "manhole": manhole,  # %
            "rainfall": rainfall,  # mm/min
            "cumsum_rainfall": cumsum_rainfall,  # mm/min
        }

    def _prepare_target(self, event_data, duration=360):
        """
        内涝因素: rainfall

        数据维度：(Frames, C, H, W)
        """
        # 提取输入数据
        flood = torch.from_numpy(event_data["flood"]).float()
        
        # fill_zeors_nums = 0
        # flood = pad_zeros_to_tensor(flood, fill_zeors_nums)
        
        flood = flood[:duration, :, :, :] * 1000  # m -> mm

        # # !为了保证整除，需要对height从385变为384
        # flood = flood[:, :, 1:, :] * 1000  # m -> mm

        # 只关注300x300m区域
        # flood = flood[:, :, 200:200 + 300, 200:200 + 300]
        # # ! expr: 只关注内涝点周围300x300区域
        # flood = flood[:, :, 20:20 + 300, 200:200 + 300]
        # # # ! expr: 只关注内涝点周围64x64区域
        # flood = flood[:, :, 340:340+64, 320:320+64]

        return flood

    def _set_others(self, inputVars):
        """假设参数"""
        # 降雨损失（全域） mm/min
        rain_loss_val = 0.0033
        # 入渗（透水区域） mm/min
        infiltration = 0.488
        # 下水道排水（排水口） mm/min
        sewer_drainage = 0.2
        # sewer_drainage = 23.7  # TODO: 需要再考虑
        # 78min退水1850mm -> 23.7 mm/min
        """构造降雨总损失矩阵"""
        impervious, manhole = inputVars["impervious"], inputVars["manhole"]
        impervious_area = impervious[0, 0]
        manhole_area = manhole[0, 0]

        # 降雨损失
        rain_loss = torch.ones(1) * rain_loss_val

        # 排水矩阵
        drainage = torch.zeros_like((manhole_area))
        # 入渗损失
        drainage += (1 - impervious_area) * infiltration
        # 下水道排水损失
        drainage += manhole_area * sewer_drainage

        # rain_loss = rain_loss[None, None, :, :]
        drainage = drainage[None, None, :, :]

        # 内涝初始化
        # flood = torch.zeros_like((drainage))

        inputVars["rain_loss"] = rain_loss
        inputVars["drainage"] = drainage
        # inputVars["flood"] = flood

        return inputVars

    def __getitem__(self, index):
        # 加载一场事件
        event_data, event_dir = self._load_event(index)

        # S,C,H,W
        # TODO：考虑不同长度的序列输入
        """input
        构造输入，即一些地理因素
        """
        inputVars = self._prepare_input(event_data, event_dir)
        """target
        构造预测目标，即洪水变量
        """
        targetVars = self._prepare_target(event_data)
        # """others
        # 每个时刻的降雨损失（恒定），根据雨水损失、入渗、排水决定
        # """
        # inputVars = self._set_others(inputVars)

        return [inputVars, targetVars,event_dir]

    def __len__(self):
        return self.num_samples


def preprocess_inputs(t, inputs, device,nums=30):
    """
    归一化
    """
    # 特征变量
    absolute_DEM = inputs["absolute_DEM"]  # (B, 1, 1, H, W)
    impervious = inputs["impervious"]  # (B, 1, 1, H, W)
    manhole = inputs["manhole"]  # (B, 1, 1, H, W)
    rainfall = inputs["rainfall"]
    cumsum_rainfall = inputs["cumsum_rainfall"]
    # print(rainfall.max(),rainfall.min())
    
    # TODO: 思考归一化
    norm_absolute_DEM = MinMaxScaler(
        absolute_DEM, inputs["max_DEM"][0], inputs["min_DEM"][0])
    # norm_impervious = MinMaxScaler(impervious, impervious.max(), impervious.min())
    # norm_manhole = MinMaxScaler(manhole, manhole.max(), manhole.min())
    norm_impervious = MinMaxScaler(impervious, 0.95, 0.05)
    norm_manhole = MinMaxScaler(manhole, 1, 0)
    
    
    # # 积水深度
    # norm_flood = output_t_info["output_t"]  # 预测的就是归一化后的
    # flood = r_MinMaxScaler(norm_flood, 5000, 0)
    
    # 提取降雨
    H, W = absolute_DEM.shape[-2:]
    rainfall = get_past_rainfall(rainfall, t, 1, H, W)
    # rainfall = get_past_rainfall(rainfall, t, nums, H, W)
    cumsum_rainfall = get_past_rainfall(cumsum_rainfall, t, nums, H, W)
    
    # 累计降雨每个时间切片都从0开始
    # cumsum_rainfall = cumsum_rainfall - cumsum_rainfall[:, :1]
    
    norm_rainfall = MinMaxScaler(rainfall, 6, 0)  # 500年一遇最大强度为6
    norm_cumsum_rainfall = MinMaxScaler(cumsum_rainfall, 250, 0)  # 500年一遇最大总降雨量为220多
    
    # 径流
    # runoff = flood + rainfall
    # runoff[norm_absolute_DEM==1] = 0 # 建筑的位置不计算
    # norm_runoff = MinMaxScaler(runoff, 5000, 0)
    
    # 对齐变量维度
    # 拼接所有变量
    processd_inputs = torch.cat(
        [
            norm_rainfall,
            norm_cumsum_rainfall,
            norm_absolute_DEM,
            norm_impervious,
            norm_manhole,
            # norm_flood,
            # norm_runoff,
        ],
        dim=2,
    )
    processd_inputs = processd_inputs.to(device=device, dtype=torch.float32)
    return processd_inputs


def get_past_rainfall(rainfall, t, nums, H, W):
    B, S, C, _, _ = rainfall.shape  # Assuming C is the number of channels
    # Ensure the time range for extraction is within valid limits
    start_idx = max(0, t - nums + 1)
    end_idx = min(t + 1, S)

    # Initialize a new tensor to store the extracted rainfall data
    extracted_rainfall = torch.zeros((B, 1, nums, H, W), device=rainfall.device)

    # Calculate the actual number of time steps extracted
    actual_num_steps = end_idx - start_idx

    # Extract rainfall data and place it in the correct position
    # Reshape the extracted data to match the desired dimensions
    extracted_data = rainfall[:, start_idx:end_idx, 0, ...]
    extracted_data = extracted_data.unsqueeze(1)  # Add an extra dimension for the channel
    extracted_data = extracted_data.expand(-1, 1, -1, H, W)

    extracted_rainfall[:, :, nums - actual_num_steps:, ...] = extracted_data

    # Ensure the shape of the extracted data is correct
    assert extracted_rainfall.shape == (B, 1, nums, H, W)

    return extracted_rainfall





def preprocess_inputs_v1(t, inputs, output_t_info, device):
    """
    归一化
    """
    # 特征变量
    absolute_DEM = inputs["absolute_DEM"]  # (B, 1, 1, H, W)
    impervious = inputs["impervious"]  # (B, 1, 1, H, W)
    manhole = inputs["manhole"]  # (B, 1, 1, H, W)
    rainfall = inputs["rainfall"]

    # TODO: 思考归一化
    norm_absolute_DEM = MinMaxScaler(
        absolute_DEM, absolute_DEM.max(), absolute_DEM.min()
    )
    # norm_impervious = MinMaxScaler(impervious, impervious.max(), impervious.min())
    # norm_manhole = MinMaxScaler(manhole, manhole.max(), manhole.min())
    norm_impervious = MinMaxScaler(impervious, 0.95, 0.05)
    norm_manhole = MinMaxScaler(manhole, 1, 0)
    norm_rainfall_event = MinMaxScaler(rainfall, 6, 0)  # 500年一遇最大强度为6
    norm_flood = output_t_info["output_t"]  # 预测的就是归一化后的

    # 处理降雨
    # print("norm_rainfall_event.shape[1]:",norm_rainfall_event.shape[1])
    if t >= norm_rainfall_event.shape[1]:
        B = norm_rainfall_event.shape[0]
        norm_rainfall = torch.zeros((B, 1, 1, 1, 1)).to(device)
    else:
        norm_rainfall = norm_rainfall_event[:, t : t + 1]  # (B, S, 1, 1, 1)
    H, W = absolute_DEM.shape[-2:]
    norm_rainfall = norm_rainfall.repeat(1, 1, 1, H, W)

    # # 水量转移
    # # 逆归一化洪水，以计算水量转移矩阵
    # flood = r_MinMaxScaler(output_t_info["output_t"], output_t_info["max"],
    #                        output_t_info["min"])
    # flow_volume = cal_flow_volume(absolute_DEM, flood)
    # flow_volume = flow_volume.to(device)
    # norm_flow_volume = MinMaxScaler(flow_volume, flow_volume.max(),
    #                                 flow_volume.min())

    # 对齐变量维度
    # 拼接所有变量
    processd_inputs = torch.cat(
        [
            # norm_absolute_DEM,
            # norm_impervious,
            # norm_manhole,
            norm_rainfall,
            # norm_flood,
        ],
        dim=2,
    )
    processd_inputs = processd_inputs.to(device=device, dtype=torch.float32)
    return processd_inputs


def MinMaxScaler(data, max=1, min=0):
    return (data - min) / (max - min)


def r_MinMaxScaler(data, max=1, min=0):
    return data * (max - min) + min


def pad_zeros_to_tensor(tensor, num_zeros):
    # 获取原始张量的形状
    # 获取原始张量的形状
    original_shape = tensor.shape
    
    # 创建一个包含指定数量零值的零张量
    zeros = torch.zeros((num_zeros,) + original_shape[1:], dtype=tensor.dtype)
    
    # 使用torch.cat将零张量与原始张量连接在一起
    padded_tensor = torch.cat([zeros, tensor], dim=0)
    padded_tensor = padded_tensor[:-num_zeros]
    
    return padded_tensor