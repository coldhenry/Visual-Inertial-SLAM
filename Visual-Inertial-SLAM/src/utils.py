import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler

def load_data(file_name):
  '''
  function to read visual features, IMU measurements and calibration parameters
  Input:
      file_name: the input data file. Should look like "XXX_sync_KLT.npz"
  Output:
      t: time stamp
          with shape 1*t
      features: visual feature point coordinates in stereo images, 
          with shape 4*n*t, where n is number of features
      linear_velocity: IMU measurements in IMU frame
          with shape 3*t
      rotational_velocity: IMU measurements in IMU frame
          with shape 3*t
      K: (left)camera intrinsic matrix
          [fx  0 cx
            0 fy cy
            0  0  1]
          with shape 3*3
      b: stereo camera baseline
          with shape 1
      M: stero camera intrinsic matrix
      cam_T_imu: extrinsic matrix from IMU to (left)camera, in SE(3).
          close to 
          [ 0 -1  0 t1
            0  0 -1 t2
            1  0  0 t3
            0  0  0  1]
          with shape 4*4
  '''
  with np.load(file_name) as data:
      t = data["time_stamps"] # time_stamps
      features = data["features"].astype('float32') # 4 x num_features : pixel coordinates of features
      linear_velocity = data["linear_velocity"].astype('float32') # linear velocity measured in the body frame
      rotational_velocity = data["rotational_velocity"].astype('float32') # rotational velocity measured in the body frame
      K = data["K"].astype('float32') # intrindic calibration matrix
      b = data["b"].astype('float32') # baseline
      cam_T_imu = data["cam_T_imu"].astype('float32') # Transformation from imu to camera frame
      
      
      M = np.zeros((4,4))
      M[0,:3], M[1,:3], M[3,:3] = K[0,:],K[1,:],K[1,:]
      M[2,:] = np.hstack((K[0,:],-K[0,0]*b))
      
      t = t.squeeze()
      t -= t[0]
      dt = np.zeros(t.shape)
      t = t.squeeze()
      dt[1:] = t[1:]-t[:-1]
      
  return M,t, dt, features[:,::2,:],linear_velocity,rotational_velocity, cam_T_imu

def visualize_trajectory(ax, pose_pre, pose, title=None):
    '''
    function to visualize the trajectory in 2D
    Input:
        pose:   4*4*N matrix representing the camera pose,
                where N is the number of pose, and each
                4*4 matrix is in SE(3)
        landmarks: 4*M matrix with homogeneous coords
    '''

    # show trajectory
    ax.plot(pose_pre[0, 3, :], pose_pre[1, 3, :], 'b-', label='inertial only')
    ax.plot(pose[0, 3, :], pose[1, 3, :], 'r.', label='visual-inertial')
    # show orientation
    n_pose = pose_pre.shape[2]
    ax.scatter(pose[0, 3, 0], pose[1, 3, 0], marker='s', label="start")
    ax.scatter(pose[0, 3, -1], pose[1, 3, -1], marker='o', label="end")
    select_ori_index = list(range(0, n_pose, int(n_pose / 50)))
    yaw_list = []
    for i in select_ori_index:
        _, _, yaw = mat2euler(pose[:3, :3, i])
        yaw_list.append(yaw)
    dx = np.cos(yaw_list)
    dy = np.sin(yaw_list)
    dx, dy = [dx, dy] / np.sqrt(dx ** 2 + dy ** 2)
    ax.quiver(pose[0, 3, select_ori_index], pose[1, 3, select_ori_index], dx, dy,
              color="b", units="xy", width=1)
    ax.set_xlabel('x / m')
    ax.set_ylabel('y / m')
    ax.set_title(title)
    ax.axis('equal')
    ax.grid(False)
    ax.legend()

def visualize_trajectory_2d(pose,path_name="Unknown",show_ori=False):
  '''
  function to visualize the trajectory in 2D
  Input:
      pose:   4*4*N matrix representing the camera pose, 
              where N is the number of pose, and each
              4*4 matrix is in SE(3)
  '''
  fig,ax = plt.subplots(figsize=(5,5))
  n_pose = pose.shape[2]
  ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name)
  ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
  ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
  if show_ori:
      select_ori_index = list(range(0,n_pose,max(int(n_pose/50), 1)))
      yaw_list = []
      for i in select_ori_index:
          _,_,yaw = mat2euler(pose[:3,:3,i])
          yaw_list.append(yaw)
      dx = np.cos(yaw_list)
      dy = np.sin(yaw_list)
      dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
      ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
          color="b",units="xy",width=1)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.axis('equal')
  ax.grid(False)
  ax.legend()
  plt.show(block=True)
  return fig, ax


def visualize_landmark(ax, pose, landmarks0, landmarks, title=None):
    '''
    function to visualize the trajectory in 2D
    Input:
        pose:   4*4*N matrix representing the camera pose,
                where N is the number of pose, and each
                4*4 matrix is in SE(3)
        landmarks: 4*M matrix with homogeneous coords
    '''
    ax.plot(landmarks0[0], landmarks0[1], 'c.', label='initial landmarks')
    ax.plot(landmarks[0], landmarks[1], 'k.', label='updated ones')
    ax.plot(pose[0, 3, :], pose[1, 3, :], 'r-')
    ax.set_xlabel('x / m')
    ax.set_ylabel('y / m')
    ax.set_title(title)
    ax.axis('equal')
    ax.grid(False)
    ax.legend()
    
def visualize_covariance(ax, t, Sigma_pre, Sigma, title=None):
    ax.plot(t, np.log(np.linalg.norm(Sigma_pre, axis=(0, 1))), label='prediction only')
    ax.plot(t, np.log(np.linalg.norm(Sigma, axis=(0, 1))), label='prediction & update')
    ax.set_xlabel('time / s')
    ax.set_ylabel('matrix norm / dB')
    ax.set_title(title)
    ax.grid()
    ax.legend(loc=1)
