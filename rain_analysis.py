import os
import numpy as np
import matplotlib.pyplot as plt
from rain_generate import rain_generate

# 生成不同重现期的降雨数据
reoccurrence_periods = [0.1, 0.2,0.3,0.4,0.5, 1, 2, 5, 10, 20, 50, 100]
rainfall_data = {}

for P in reoccurrence_periods:
    rain_total = rain_generate(P=P)
    rainfall_data[P] = rain_total
    print(f"重现期{P}：{rainfall_data[P]}")

# 遍历目录下的所有文件夹
data_dir = "/root/autodl-tmp/caoxiaoyan/urbanflood/data/urbanflood22/train/flood/location16"
count_above_1_year = 0  # 统计0.5年降雨重现期及以上的事件数量

event_above_1_year = []
event_rainfall = {}

for event_folder in sorted(os.listdir(data_dir)):
    if "r" in event_folder:
        continue
    event_path = os.path.join(data_dir, event_folder)
    
    # 确保是文件夹而不是文件
    if os.path.isdir(event_path):
        rainfall_file = os.path.join(event_path, "rainfall.npy")
        
        # 检查文件是否存在
        if os.path.exists(rainfall_file):
            # 读取降雨数据
            rainfall = np.load(rainfall_file)
            
            # 判断降雨事件属于哪个重现期：最接近哪个重现期就属于哪个
            closest_P = min(reoccurrence_periods, key=lambda x: abs(np.sum(rainfall) - np.sum(rainfall_data[x])))
            print(f"降雨事件 {event_folder} 属于 {closest_P} 年重现期，降雨量为 {np.sum(rainfall):.2f} 毫米")
            
            # 统计0.2年降雨重现期及以上的事件数量
            if closest_P >= 0.2:
                count_above_1_year += 1
                event_above_1_year.append((event_folder, closest_P, np.sum(rainfall),rainfall))

# 输出0.2年降雨重现期及以上的事件数量
print(f"0.2年降雨重现期及以上的事件数量为 {count_above_1_year} 个")

# 创建"rain_vs"文件夹
output_dir = "rain_vs"
os.makedirs(output_dir, exist_ok=True)

# 绘制降雨量条形图并保存
# 按降雨量从高到低排序并输出
sorted_events = sorted(event_above_1_year, key=lambda x: x[2], reverse=True)
for event, closest_P, total_volumn, rainfall in sorted_events:
    print(f"事件 {event}，降雨量为 {total_volumn:.2f} 毫米，属于 {closest_P} 年重现期")
    
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(rainfall)), rainfall, label=f"Event {event}")
    plt.title(f"Rainfall vs Time - Event {event}")
    plt.xlabel("Time")
    plt.ylabel("Rainfall (mm)")
    plt.legend()
    
    # 保存图像到"rain_vs"文件夹下
    output_file = os.path.join(output_dir, f"event_{event}.png")
    plt.savefig(output_file)
    plt.close()

print("Images have been saved to 'rain_vs' folder.")
