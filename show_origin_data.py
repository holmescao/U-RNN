from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import numpy as np
import os 



def rainfalls_split(rainfall_data,pure_name,categories):
    
    # Categorize the events
    for name, (max_value,rain_accum) in rainfall_data.items():
        if max_value > 2:
            categories["greater_than_2m"].append([name, max_value,rain_accum])
        elif max_value > 1.5:
            categories["greater_than_1.5m"].append([name, max_value,rain_accum])
        elif max_value > 1:
            categories["greater_than_1m"].append([name, max_value,rain_accum])
        elif max_value > 0.5:
            categories["greater_than_0.5m"].append([name, max_value,rain_accum])
        else:
            categories["less_than_0.5m"].append([name, max_value,rain_accum])
        
        if max_value > 2:
            pure_name["greater_than_2m"].append(name)
        elif max_value > 1.5:
            pure_name["greater_than_1.5m"].append(name)
        elif max_value > 1:
            pure_name["greater_than_1m"].append(name)
        elif max_value > 0.5:
            pure_name["greater_than_0.5m"].append(name)
        else:
            pure_name["less_than_0.5m"].append(name)

    return categories, pure_name


ori_root = "/root/autodl-tmp/caoxiaoyan/urbanflood/data/urbanflood22/train"
location = "location16"

flood_dir = os.path.join(ori_root,"flood",location)
geodata_dir = os.path.join(ori_root,"geodata",location)

event_names = sorted(os.listdir(flood_dir))

impervious_path = os.path.join(geodata_dir,"impervious.npy")
impervious = np.load(impervious_path)

"""降雨数据处理"""
# for event in event_names:
#     if  "G" in event:
#         event_dir = os.path.join(flood_dir,event)
#         # flood_path  = os.path.join(event_dir,"flood.npy")
#         rainfall_path  = os.path.join(event_dir,"rainfall.npy")
#         # print(flood_path)
#         # print(rainfall_path)

#         # flood = np.load(flood_path)
#         rainfall = np.load(rainfall_path)
#         # print(f"{event} rainfall max: {rainfall.max()};\t flood max:{flood.max()}")
        
#         # rainfall /= 60
#         # np.save(rainfall_path,rainfall)
        
#         print(f"{event} rainfall max: {rainfall.max()}, shape:{rainfall.shape}")

"""内涝数据处理"""
save_dir = "figs/input_floods"
if not os.path.exists(save_dir):
    os.makedirs(save_dir,exist_ok=True)
    
rainfall_data = {}
for event in tqdm(event_names):
    # if "G" not in event:
    #     continue
    event_dir = os.path.join(flood_dir,event)
    flood_path  = os.path.join(event_dir,"flood.npy")
    rainfall_path  = os.path.join(event_dir,"rainfall.npy")
    # print(flood_path)
    # print(rainfall_path)

    flood = np.load(flood_path)
    rain = np.load(rainfall_path)
    rain_accum = np.sum(rain)
 
    flood = flood[:,0,340:340+64, 320:320+64]
    max_value = np.max(flood)
    max_index = np.unravel_index(np.argmax(flood, axis=None), flood.shape)
    max_map = flood[max_index[0]]
    # print(f"{event} max_value:{max_value}, max_index:{max_index} shape:{max_map.shape}")
    
    rainfall_data[event] = (max_value, rain_accum)
    # plt.imshow(max_map)
    # plt.colorbar()
    # plt.savefig(f"{save_dir}/{event}_max_map.jpg")
    # plt.close()
    # sys.exit()

# Define the categories
categories = {
    "greater_than_2m": [],
    "greater_than_1.5m": [],
    "greater_than_1m": [],
    "greater_than_0.5m": [],
    "less_than_0.5m": []
}
pure_name = {
    "greater_than_2m": [],
    "greater_than_1.5m": [],
    "greater_than_1m": [],
    "greater_than_0.5m": [],
    "less_than_0.5m": []
}
categories,pure_name = rainfalls_split(rainfall_data,pure_name, categories)
for k,v in categories.items():
    print(f"{k}:{len(v)}")

print(pure_name)