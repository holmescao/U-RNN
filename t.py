# 事件名称列表
event_names = [
    "事件 G1135_intensity_156，降雨量为 180.70 毫米，属于 50 年重现期",
    "事件 G1162_intensity_131，降雨量为 173.02 毫米，属于 50 年重现期【测试】",
    "事件 G1162_intensity_151，降雨量为 166.78 毫米，属于 50 年重现期",
    "事件 G3531_intensity_128，降雨量为 132.88 毫米，属于 10 年重现期【测试】",
    "事件 G1166_intensity_246，降雨量为 108.84 毫米，属于 2 年重现期【测试】",
    "事件 G3543_intensity_271，降雨量为 106.69 毫米，属于 2 年重现期",
    "事件 G3555_intensity_284，降雨量为 102.60 毫米，属于 2 年重现期",
    "事件 G1135_intensity_163，降雨量为 98.19 毫米，属于 2 年重现期",
    "事件 G1162_intensity_170，降雨量为 94.63 毫米，属于 1 年重现期",
    "事件 G1135_intensity_161，降雨量为 84.74 毫米，属于 1 年重现期【测试】",
    "事件 G3535_intensity_215，降雨量为 74.16 毫米，属于 0.5 年重现期",
    "事件 G1162_intensity_132，降雨量为 65.36 毫米，属于 0.4 年重现期",
    "事件 G1166_intensity_120，降雨量为 63.64 毫米，属于 0.4 年重现期",
    "事件 G1135_intensity_179，降雨量为 47.72 毫米，属于 0.2 年重现期"
]

# 划分为训练和测试两类
training_events = []
testing_events = []

for event_name in event_names:
    if "【测试】" in event_name:
        testing_events.append(event_name)
    else:
        training_events.append(event_name)

# 输出训练和测试事件名称
print("训练事件名称：")
for event_name in training_events:
    event_name =event_name.split("事件 ")[1].split("，")[0]
    print(event_name)

print("\n测试事件名称：")
for event_name in testing_events:
    event_name =event_name.split("事件 ")[1].split("，")[0]
    print(event_name)
