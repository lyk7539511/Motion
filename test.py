import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建 3D 坐标轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 设置数据
xs = np.arange(10)
ys = np.random.randint(0, 10, size=10)
zs = np.random.randint(0, 10, size=10)

# 绘制散点图
sc = ax.scatter(xs, ys, zs)

# 在每个数据点位置处添加文本标注
for x, y, z, label in zip(xs, ys, zs, ['({},{},{})'.format(x,y,z) for x,y,z in zip(xs,ys,zs)]):
    ax.text(x, y, z, label)

# 添加坐标轴标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 显示图形
plt.show()
