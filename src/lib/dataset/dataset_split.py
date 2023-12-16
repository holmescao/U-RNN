import random
import os

def split_and_merge_datasets(events_list):
    # 设置随机种子以确保可重复性
    random.seed(1)

    # 根据名字前缀分割实测降雨和设计降雨
    real_rainfall = [event for event in events_list if event.startswith("G")]
    design_rainfall = [event for event in events_list if event.startswith("r")]

    # 随机选择实测降雨的21场和设计降雨的12场作为训练集
    train_real = random.sample(real_rainfall, 7)
    train_design = random.sample(design_rainfall, 17)

    # 剩余的场次作为测试集
    test_real = [rainfall for rainfall in real_rainfall if rainfall not in train_real]
    test_design = [rainfall for rainfall in design_rainfall if rainfall not in train_design]

    # 合并训练集和测试集
    train_set = train_real + train_design
    test_set = test_real + test_design

    return train_set, test_set



# 假设 events_list 包含了所有的降雨样本名
data_root = "/root/autodl-tmp/caoxiaoyan/urbanflood/data/urbanflood22"
data_dir = os.path.join(data_root, "train")

geo_root = os.path.join(data_dir, "geodata")
flood_root = os.path.join(data_dir, "flood")
locations = sorted(os.listdir(flood_root), key=lambda x: int(''.join(filter(str.isdigit, x))))

locations_dir = [os.path.join(flood_root, loc) for loc in locations]

event_names = sorted(os.listdir(locations_dir[0]))

filter_rains = [
        "G1162_intensity_137",
        "G3522_intensity_116T",
        "G3522_intensity_144",
        "G3522_intensity_148",
        "G1166_intensity_115",
        "G1166_intensity_115K",
        "G1166_intensity_119",
        "G1166_intensity_126",
        "G1166_intensity_128",
        "G1166_intensity_134",
        "G1166_intensity_136",
        "G3522_intensity_116",
        "G3538_intensity_119",
        "G3555_intensity_120",
        "G3555_intensity_136"
    ]
for r in filter_rains:
    event_names.remove(r)
print(len(event_names))

# 调用函数并获取训练集和测试集
train_set, test_set = split_and_merge_datasets(event_names)

# 保存训练集到文件
with open('plan2_train_set.txt', 'w') as train_file:
    for event in train_set:
        train_file.write(event + '\n')

# 保存测试集到文件
with open('plan2_test_set.txt', 'w') as test_file:
    for event in test_set:
        test_file.write(event + '\n')

# 打印结果
print("训练集已保存到 plan2_train_set.txt")
print("测试集已保存到 plan2_test_set.txt")
