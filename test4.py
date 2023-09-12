import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

data = np.load('result_all.npy')
print(data.shape)
data = np.transpose(data, (1,2,0))
x = data[0,:,0]
y = data[0,:,1]
z = data[0,:,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, s=17)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

sns.set_style("darkgrid")
plt.show()