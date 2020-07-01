# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:07:46 2020

@author: coldhenry
"""
import tqdm
import numpy as np
import copy
from utils import load_data, visualize_trajectory, visualize_landmark, visualize_covariance
from math_utils import feature_T_xyz
import matplotlib.pyplot as plt
from ExtendedKalmanFilter import ExtendedKalmanFilter

#%% prediction-only
def IMU_prediction_only():    
    EKF_1 = ExtendedKalmanFilter(M, t, dt, features, linear_velocity, rotational_velocity, cam_T_imu)
    for i in range(1,EKF_1.len_t): #t.shape[0]
        EKF_1.prediction(i)
    
    # for plotting
    wld_T_imu_prediction = copy.deepcopy(EKF_1.wld_T_imu)
    sigma_imu_prediction = copy.deepcopy(EKF_1.sigma_imu)
    return wld_T_imu_prediction, sigma_imu_prediction


#%% landmarks update-only
def landmark_update_only():
    # initialization
    EKF_2 = ExtendedKalmanFilter(M, t, dt, features, linear_velocity, rotational_velocity, cam_T_imu)
    EKF_2.initialization("SEPERATE")
    
    # update with first observation
    #ind = np.where(features[0,:,0] != -1)[0]
    ind = np.flatnonzero(features[0, :, 0] != -1) 
    EKF_2.mu_feat[:,ind] = EKF_2.mu0_feat[:,ind] = feature_T_xyz(features[:,ind,0], EKF_2.M, EKF_2.wld_T_imu[:,:,0], EKF_2.imu_T_cam)
    
    for j in tqdm.trange(1,EKF_2.len_t, unit='frame'):
        # EKF prediction 
        EKF_2.prediction(j)
        
        # find features index at this step    
        ind, cur_feat, N_feat = EKF_2.find_indicies(j)
        
        if N_feat:
            # find and add features that appear for the first time 
            EKF_2.first_feature_update(j, N_feat, ind, "SEPERATE")
               
            # EKF features update
            feat_head, EKF_2.mu_feat = EKF_2.landmark_update(j, ind, cur_feat, N_feat)
            
            EKF_2.motion_update(j, ind, cur_feat, feat_head)

#%% Visual-Inertial EKF-based SLAM        
def Visual_inertial_SLAM():
    # initialization
    EKF_3 = ExtendedKalmanFilter(M, t, dt, features, linear_velocity, rotational_velocity, cam_T_imu)
    EKF_3.initialization("FUSED")
    
    # update with first observation
    #ind = np.where(features[0,:,0] != -1)[0]
    ind = np.flatnonzero(features[0, :, 0] != -1) 
    EKF_3.mu[:,ind] = EKF_3.mu0_feat[:,ind] = feature_T_xyz(features[:,ind,0], EKF_3.M, EKF_3.wld_T_imu[:,:,0], EKF_3.imu_T_cam)
    
    length = EKF_3.len_t
    for j in tqdm.trange(1, length, desc='VI-SLAM', unit='frame'):
        # EKF prediction 
        EKF_3.prediction(j)
        
        EKF_3.extract_Sigma(j, EKF_3.exp_term)
        
        # find features index at this step    
        ind, cur_feat, N_feat = EKF_3.find_indicies(j)
        
        if N_feat:
            # find and add features that appear for the first time 
            EKF_3.first_feature_update(j, N_feat, ind, "FUSED")
            
            # do EKF update on both landmarks and IMU pose
            EKF_3.EKF_Fused(j, N_feat, ind, cur_feat)
            
    return EKF_3
        


#%%
if __name__ == '__main__':
    num = "0034"
    filename = "./data/"+num+".npz"
    M, t, dt, features, linear_velocity, rotational_velocity, cam_T_imu = load_data(filename)
    
    wld_T_imu_prediction, sigma_imu_prediction = IMU_prediction_only()
    EKF_3 = Visual_inertial_SLAM()

    # plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    visualize_trajectory(axes[0], wld_T_imu_prediction, EKF_3.wld_T_imu, title='trajectory & orientation')
    visualize_landmark(axes[1], EKF_3.wld_T_imu, EKF_3.mu0_feat, EKF_3.mu_feat, title='trajectory & landmarks')
    visualize_covariance(axes[2], t, sigma_imu_prediction, EKF_3.sigma_imu, title='imu pose covariance')
    plt.savefig(num+"_3.png")
