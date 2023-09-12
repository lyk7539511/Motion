import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

def angle(A, B, C):
    # calculate the angle between AB and BC
    angle = np.arccos(np.dot((A-B)/np.linalg.norm(A-B), (C-B)/np.linalg.norm(C-B)))
    return np.round(angle/np.pi*180, 3)

def angle_list(m_type:str, all_keypoints):
    assert m_type in motion_type.keys(), 'motion type not found'
    a_list = []
    
    for t in range(np.shape(all_keypoints)[0]):
        a_cache = []
        for i in range(np.shape(motion_type[m_type])[0]):
            A = np.array(all_keypoints[t,motion_type[m_type][i][0],:])
            B = np.array(all_keypoints[t,motion_type[m_type][i][1],:])
            C = np.array(all_keypoints[t,motion_type[m_type][i][2],:])
            a_cache.append(angle(A, B, C))
        a_list.append(a_cache)
    return a_list

def flip_data(data):
    """
    horizontal flip
        data: [N, F, 17, D] or [F, 17, D]. X (horizontal coordinate) is the first channel in D.
    Return
        result: same
    """
    left_joints = [4, 5, 6, 11, 12, 13]
    right_joints = [1, 2, 3, 14, 15, 16]
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1                                               # flip x of all joints
    flipped_data[..., left_joints+right_joints, :] = flipped_data[..., right_joints+left_joints, :]
    return flipped_data

def plot_angle(m_type:str, a_list):
    assert m_type in motion_type.keys(), 'motion type not found'
    t = np.arange(0, data.shape[0], 1)

    def update(i):
        ax.clear()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        # 所有點位
        ax.scatter(data[i,:,0], data[i,:,2], data[i,:,1]*-1, s=17)
        # 關鍵點
        for m in range(np.shape(motion_type[m_type])[0]):
            for k in range(np.shape(motion_type[m_type])[1]):
                ax.scatter(data[i,motion_type[m_type][m][k],0], 
                           data[i,motion_type[m_type][m][k],2], 
                           data[i,motion_type[m_type][m][k],1]*-1, 
                           c='r', marker='o')
        # 關鍵點連線
        for m in range(np.shape(motion_type[m_type])[0]):
            # 三個關節中的第一個與第二個
            ax.plot([data[i,motion_type[m_type][m][0],0], data[i,motion_type[m_type][m][1],0]], 
                    [data[i,motion_type[m_type][m][0],2], data[i,motion_type[m_type][m][1],2]], 
                    [data[i,motion_type[m_type][m][0],1]*-1, data[i,motion_type[m_type][m][1],1]*-1], 
                    c='b')
            # 三個關節中的第二個與第三個
            ax.plot([data[i,motion_type[m_type][m][2],0], data[i,motion_type[m_type][m][1],0]], 
                    [data[i,motion_type[m_type][m][2],2], data[i,motion_type[m_type][m][1],2]], 
                    [data[i,motion_type[m_type][m][2],1]*-1, data[i,motion_type[m_type][m][1],1]*-1], 
                    c='b')
        # 角度
        for m in range(np.shape(motion_type[m_type])[0]):
            ang = angle(A=np.array(data[i,motion_type[m_type][m][0],:]),
                        B=np.array(data[i,motion_type[m_type][m][1],:]),
                        C=np.array(data[i,motion_type[m_type][m][2],:]))
            ax.text(data[i,motion_type[m_type][m][1],0], 
                    data[i,motion_type[m_type][m][1],2], 
                    data[i,motion_type[m_type][m][1],1]*-1, 
                    ang,
                    color='r' if ang>165 else 'b'    
                    )
        # 座標軸
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        #ax.view_init(30, 30)
        return ax

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ani = FuncAnimation(fig, update, frames=t, interval=100)
    plt.show()
    

if __name__ == "__main__":
    data = np.load('X3D.npy')
    print(data.shape)
    #data = flip_data(data)

    # 3D kepoints and their sepcification
    keypoints = np.array([[0,'Bottom torso'],
     [1,'Left hip'],
     [2,'Left knee'],
     [3,'Left foot'],
     [4,'Right hip'],
     [5,'Right knee'],
     [6,'Right foot'],
     [7,'Center torso'],
     [8,'Upper torso'],
     [9,'Neck base'],
     [10,'Center head'],
     [11,'Right shoulder'],
     [12,'Right elbow'],
     [13,'Right hand'],
     [14,'Left shoulder'],
     [15,'Left elbow'],
     [16,'Left hand']])
    keypoints = {item[0]: item[1] for item in keypoints}
    print(keypoints)

    # angle of keypoint combination
    motion_type = {
        'squat': [[1,2,3],[4,5,6]],
        'pushup': [[7,8,9],[10,11,12]],
    }
    print(np.shape(motion_type['squat']))
    print(angle_list(m_type='squat', all_keypoints=data))
    plot_angle(m_type='squat', a_list=angle_list(m_type='squat', all_keypoints=data))
    print(2)



'''
t = np.arange(0, output.shape[0], 1)

def update(i):
    #ax.clear()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.scatter(output[i,:,0], output[i,:,1], output[i,:,2], s=17)
    ax.scatter(output[i,0,0], output[i,0,1], output[i,0,2], c='r', marker='o')
    ax.scatter(output[i,1,0], output[i,1,1], output[i,1,2], c='r', marker='o')
    ax.scatter(output[i,2,0], output[i,2,1], output[i,2,2], c='r', marker='o')
    ax.plot([output[i,0,0], output[i,1,0]], [output[i,0,1], output[i,1,1]], [output[i,0,2], output[i,1,2]], c='b')
    ax.plot([output[i,2,0], output[i,1,0]], [output[i,2,1], output[i,1,1]], [output[i,2,2], output[i,1,2]], c='b')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.view_init(30, 30)
    return ax

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ani = FuncAnimation(fig, update, frames=t, interval=100)
plt.show()
'''



'''
x = output[0,:,0]
y = output[0,:,1]
z = output[0,:,2]

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
'''

