import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/Users/yuankunliu/Downloads/ffmpeg"
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import copy
from copy import deepcopy
import imageio

output = np.load('X3D.npy')
vid = imageio.get_reader('AlphaPose_demo.mp4', 'ffmpeg')
# output[:,:,:2] = output[:,:,:2] - np.array(vid.get_meta_data()['size'])/2
# output = output/(min(vid.get_meta_data()['size'])/2)

x = output[0,:,0]
y = output[0,:,2]
z = output[0,:,1] * -1

A = np.array([x[0], y[0], z[0]])
B = np.array([x[1], y[1], z[1]])
C = np.array([x[2], y[2], z[2]])
# calculate the angle between AB and BC
angle = np.arccos(np.dot((A-B)/np.linalg.norm(A-B), (C-B)/np.linalg.norm(C-B)))
print(angle/np.pi*180)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(A[0], A[1], A[2], c='r', marker='o')
ax.scatter(B[0], B[1], B[2], c='r', marker='o')
ax.scatter(C[0], C[1], C[2], c='r', marker='o')
ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], c='b')
ax.plot([C[0], B[0]], [C[1], B[1]], [C[2], B[2]], c='b')
ax.scatter(x, y, z, s=17)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

sns.set_style("darkgrid")
plt.show()