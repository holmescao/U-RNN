# U-RNN 数据集知识文档

> 用途：供开发者快速了解各数据集格式、模型输入输出规格、参数标定依据。
> 最后更新：2026-03-15

---

## 1. 数据集总览

| 数据集 | 来源 | 空间分辨率 | 时间分辨率 | 空间网格 | 时间步数 | 训练/测试 | 降雨类型 |
|---|---|---|---|---|---|---|---|
| **UrbanFlood24 full** | 本项目 | 2 m | 1 min | 500×500 | 360 | 40/15 events | 空间均匀 |
| **UrbanFlood24 lite** | 下采样版 | 8 m | 10 min | 128×128 | 36 | 8/4 events | 空间均匀 |
| **LarFNO Futian** | LarFNO 项目 | 20 m | 5 min | 400×560 | 72 | 8/4 events | 空间异质 |
| **LarFNO UKEA** | LarFNO 项目 | 8 m | 5 min（聚合） | 52×120 | 36 | 8/12 events | 空间异质 |

---

## 2. 各数据集详细规格

### 2.1 UrbanFlood24 full（原始高分辨率）

```
路径（AutoDL）：/root/autodl-tmp/data/urbanflood24/
配置文件：configs/full.yaml
```

**目录结构：**
```
urbanflood24/
  train/
    flood/location1/ ... location3/
      <event>/
        flood.npy      (360, 500, 500)  metres — MIKE+ 水深
        rainfall.npy   (360,)           mm/min — 空间均匀降雨
    geodata/location1/ ... location3/
      absolute_DEM.npy  (500, 500)  metres
      impervious.npy    (500, 500)  [0,1]
      manhole.npy       (500, 500)  {0,1}
  test/  （同结构）
```

**事件命名规则：**
- `G<ID>_intensity_<N>` — 历史场次，空间均匀降雨
- `r<N>y_p<p>_d3h` — 设计降雨，返回期 N 年，峰值比例 p，3h 持续

**模型配置参数：**
```yaml
historical_nums: 30    # 30×1min = 30min 回看
rain_max:         6.0  # mm/min
cumsum_rain_max: 250.0 # mm
flood_max:      5000   # mm
duration:        360
window_size:     360
seq_num:          28
```

**已验证：**
- ✅ 3 locations × 1 event 推理（checkpoint_939，360步，~19s）
- ✅ 数据加载正常，无 shape mismatch

---

### 2.2 UrbanFlood24 lite（轻量下采样版）

```
路径（AutoDL）：/root/autodl-tmp/data/urbanflood24_lite/  （需运行 downsample_dataset.py 生成）
配置文件：configs/lite.yaml（原有）/ configs/location1_lite.yaml（8+4 设计降雨子集）
```

**下采样规则（`tools/downsample_dataset.py`）：**
- 空间：500×500 → 125×125（每4格取1）→ 填充至 128×128
- 时间：360×1min → 36×10min（rainfall: sum; flood: subsample）
- 静态：DEM 填充 100m，其余填充 0

**事件子集（`location1_lite.yaml`）：**
- 训练 8 场（`location1_train.txt`）：r 开头设计降雨
- 测试 4 场（`location1_test.txt`）：r 开头设计降雨

**模型配置参数：**
```yaml
input_height: 128
input_width:  128
historical_nums: 3     # 3×10min = 30min 回看
rain_max:       60.0   # mm/10min
cumsum_rain_max: 250.0 # mm
duration: 36
window_size: 36
seq_num: 12
```

---

### 2.3 LarFNO Futian（region1_20m）

```
来源：D:\BaiduSyncdisk\code_repository\urbanflood\larFNO\project\benchmark\urbanflood\
配置文件：configs/futian_scratch.yaml
转换工具：tools/convert_larfno_data.py --dataset futian
```

**原始格式（LarFNO）：**
```
benchmark/urbanflood/
  flood/region1_20m/<event>/
    h.npy         (72, 400, 560)  metres — MIKE+ 水深
    rainfall.npy  (72, 400, 560)  mm/5min — 空间异质降雨
  geodata/region1_20m/
    dem.npy       (400, 560)      metres
```

**转换后格式（U-RNN）：**
```
larfno_futian/
  train/
    flood/region1/<event>/
      flood.npy      (72, 400, 560)  metres  ← h.npy 重命名
      rainfall.npy   (72, 400, 560)  mm/5min ← 不变
    geodata/region1/
      absolute_DEM.npy  (400, 560)  metres
      impervious.npy    (400, 560)  zeros （LarFNO 无此信息）
      manhole.npy       (400, 560)  zeros
  test/  （同结构）
```

**数据特性：**
- 80 个事件（event1–event80），event1–64 为训练区，event65–80 为测试区
- 空间分辨率 20 m，时间分辨率 5 min，模拟时长 6h（72 步）
- 最大降雨强度：~4.0 mm/5min，最大累积：~62 mm
- 最大水深：~2700 mm（2.7 m）
- 仅有 DEM，**无不透水面和排水管网数据**

> **关于 impervious.npy / manhole.npy 为全零：**
> U-RNN 的 `preprocess_inputs` 强制拼接 3 个静态通道 [DEM, impervious, manhole]，
> 三者缺一不可（总输入通道数 C = nums×2+3）。LarFNO 数据集未提供这两项，因此
> `convert_larfno_data.py` 以全零填充作为占位符。物理含义：impervious=0 表示
> "完全透水"，manhole=0 表示"无排水管网"——是合理的保守缺省假设，不影响训练。

**子集选取（8+4）：**
- 训练：event1, event11, event22, event31, event40, event50, event56, event63
- 测试：event65, event70, event74, event78

**模型配置参数：**
```yaml
input_height: 400
input_width:  560
historical_nums: 6     # 6×5min = 30min 回看
rain_max:        5.0   # mm/5min（实测最大 ~4.0，留余量）
cumsum_rain_max: 100.0 # mm（实测最大 ~62mm）
flood_max:      5000   # mm
duration: 72
window_size: 72
seq_num: 24            # 72/24 = 3 个梯度窗口
```

---

### 2.4 LarFNO UKEA（ukea_8m_5min，5min 聚合）

```
来源：D:\BaiduSyncdisk\code_repository\urbanflood\larFNO\project\benchmark\urbanflood\
配置文件：configs/ukea_scratch.yaml
预处理工具：tools/preprocess_ukea_rainfall.py（一次性聚合 1min→5min）
转换工具：tools/convert_larfno_data.py --dataset ukea
```

**原始格式（LarFNO）：**
```
benchmark/urbanflood/
  flood/ukea_8m_5min/<event>/
    h.npy         (36, 50, 120)   metres — MIKE+ 水深（5-min 分辨率，3h = 36步）
    rainfall.npy  (360, 50, 120)  mm/1min — 空间异质降雨（原始 1-min）
  geodata/ukea_8m_5min/
    dem.npy       (50, 120)       metres（min=21.1m, max=40.0m）
```

**重要说明：**
- `h.npy` (36, 50, 120) 为 5-min 分辨率（36步 × 5min = 3h = d3h 设计降雨持续时长）
- `rainfall.npy` (360, 50, 120) 为 1-min 分辨率；前 180 步有降雨（3h），后 180 步全零
- **预处理步骤（一次性）**：`tools/preprocess_ukea_rainfall.py` 将降雨裁剪至前 180 步，
  再按每 5 步求和聚合 → (36, 50, 120) mm/5min，与 h.npy 时间轴对齐

**数据准备两步流程：**
```bash
# Step 1：聚合降雨（在源数据上操作，一次性）
python tools/preprocess_ukea_rainfall.py \
    --src_dir /path/to/larFNO/benchmark/urbanflood \
    --dst_dir /tmp/ukea_preprocessed

# Step 2：结构转换为 U-RNN 格式
python tools/convert_larfno_data.py --dataset ukea \
    --larfno_root /tmp/ukea_preprocessed \
    --dst_root /root/autodl-tmp/data/ukea_8m_5min
  train/
    flood/ukea/<event>/
      flood.npy      (36, 52, 120)  metres   ← h.npy 填充至 52 行
      rainfall.npy   (36, 52, 120)  mm/5min  ← 聚合后填充
    geodata/ukea/
      absolute_DEM.npy  (52, 120)  metres  ← 填充区域用最大值（40m）
      impervious.npy    (52, 120)  zeros
      manhole.npy       (52, 120)  zeros
  test/  （同结构）
```

**空间填充说明：**
- H: 50 → 52（底部添 2 行零，DEM 添 2 行最大值 40.0m）
- W: 120 → 120（不变）
- 原因：AvgPool2d 要求 H 和 W 均为 4 的倍数（50/4=12.5 → 不满足 → 52/4=13 ✓）

**数据特性：**
- 20 个事件：r100y/r200y/r300y/r500y × 多组 p 值
- 最大 5-min 降雨强度：32.11 mm/5min，最大累积：~276.5 mm
- 最大水深：~2700 mm（2.7 m）
- 仅有 DEM，**无不透水面和排水管网数据**

> **关于 impervious.npy / manhole.npy 为全零：**（同 Futian，见 2.3 节说明）

**子集选取（8+12）：**
- 训练：r100y_p0.1, r100y_p0.6, r200y_p0.3, r200y_p0.7, r300y_p0.4, r300y_p0.9, r500y_p0.2, r500y_p0.6（均 _d3h_1）
- 测试：剩余 12 个事件

**模型配置参数：**
```yaml
input_height: 52
input_width:  120
historical_nums: 6     # 6×5min = 30min 回看
rain_max:       40.0   # mm/5min（实测最大 32.11mm，留 25% 余量）
cumsum_rain_max: 300.0 # mm（实测最大 ~276.5mm）
flood_max:      5000   # mm
duration: 36
window_size: 36
seq_num: 12            # 36/12 = 3 个梯度窗口
```

---

## 3. 降雨格式兼容性

U-RNN 的 `Dynamic2DFlood` 和 `preprocess_inputs` 支持两种降雨格式：

| 格式 | numpy shape | 存储 shape（DataLoader 后） | 使用场景 |
|---|---|---|---|
| **标量降雨** | `(T,)` | `(B, T, 1, 1, 1)` | UrbanFlood24 G* 事件 |
| **空间降雨** | `(T, H, W)` | `(B, T, 1, H, W)` | LarFNO Futian/UKEA, UrbanFlood24 r* 事件 |

`get_past_rainfall` 函数通过检测 `h==1 and w==1` 自动判断格式：
- 标量：broadcast 到 `(B, 1, nums, H, W)`
- 空间：直接使用 `(B, 1, nums, H, W)`

---

## 4. 评估指标说明

`test.py` 的 `compute_metrics` 函数计算以下 6 个指标（均以输出 Excel 形式保存）：

| 指标 | 公式 | 单位 | 越好方向 |
|---|---|---|---|
| **R²** | 1 - SS_res/SS_tot（全空间×时间） | — | ↑ 最大 1.0 |
| **MSE** | mean((pred-gt)²) | m² | ↓ |
| **RMSE** | √MSE | m | ↓ |
| **MAE** | mean(\|pred-gt\|) | m | ↓ |
| **PeakR²** | R² 仅在峰值时刻 t*（gt 空间均值最大） | — | ↑ |
| **CSI** | TP/(TP+FP+FN)，基于时间最大水深二值化 | — | ↑ 最大 1.0 |

CSI 阈值：`flood_thres=150 mm`（0.15 m），由 `config.py` 控制。

---

## 5. 数据准备命令速查

```bash
# ① 转换 LarFNO Futian 数据（在本地或服务器运行）
cd U-RNN
python tools/convert_larfno_data.py --dataset futian \
  --larfno_root /path/to/larFNO/project/benchmark/urbanflood \
  --dst_root /root/autodl-tmp/data/larfno_futian

# ② 转换 LarFNO UKEA 数据（先聚合降雨，再结构转换）
# Step 2a：1-min → 5-min 聚合，仅保留 20 个事件
python tools/preprocess_ukea_rainfall.py \
  --src_dir /path/to/larFNO/project/benchmark/urbanflood \
  --dst_dir /tmp/ukea_preprocessed
# Step 2b：结构转换为 U-RNN 格式
python tools/convert_larfno_data.py --dataset ukea \
  --larfno_root /tmp/ukea_preprocessed \
  --dst_root /root/autodl-tmp/data/ukea_8m_5min

# ③ 下采样 UrbanFlood24 → lite（在 AutoDL 运行）
cd /root/autodl-tmp/U-RNN
python tools/downsample_dataset.py \
  --src_root /root/autodl-tmp/data/urbanflood24 \
  --dst_root /root/autodl-tmp/data/urbanflood24_lite \
  --spatial_factor 4 --temporal_factor 10

# ④ 训练（各数据集独立 YAML）
python main.py --exp_config configs/futian_scratch.yaml
python main.py --exp_config configs/ukea_scratch.yaml
python main.py --exp_config configs/location1_lite.yaml

# ⑤ 测试
python test.py --exp_config configs/futian_scratch.yaml --timestamp <ts>
```

---

## 6. 架构对数据集的约束

| 约束 | 原因 | 说明 |
|---|---|---|
| H 和 W 均为 **4 的倍数** | Encoder 有 2× AvgPool2d | 500✓, 128✓, 400✓, 560✓, 52✓, 120✓ |
| `historical_nums` 决定输入通道数 C = nums×2+3 | net_params.py 的 input_channels | 改动需同步 configs |
| `duration` = `window_size` | SWP 范式假设事件等长 | 跨数据集不同 duration 需用不同 YAML |
| `flood_max` 影响模型输出范围 | MinMaxScaler 归一化 | 不同数据集校准不同 |
