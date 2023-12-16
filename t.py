import matplotlib.pyplot as plt
import numpy as np
import os

file_dir = r"/root/autodl-tmp/caoxiaoyan/urbanflood/data/urbanflood22/train/geodata/location16"
file_path = os.path.join(file_dir,"absolute_DEM.npy")

dem = np.load(file_path)
print(dem.max())
print(dem.min())

# 绘制DEM数据
plt.figure(figsize=(10, 8))
plt.imshow(dem, cmap='terrain')  # 使用地形颜色映射
plt.colorbar(label='Elevation (m)')  # 添加色标并标记单位
plt.title('Digital Elevation Model (DEM)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
# plt.show()
plt.savefig("dem.png",dpi=150)
plt.close()