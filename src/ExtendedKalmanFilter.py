# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 21:27:03 2020

@author: coldhenry
"""
import numpy as np
from scipy.linalg import expm
from math_utils import hat, curly_hat, feature_T_xyz, pi_func, der_pi, dot

class ExtendedKalmanFilter:
    
    def __init__(self, M, t, dt, features, linear_velocity, rotational_velocity, cam_T_imu):
        self.dt = dt
        self.features = features
        self.linear_velocity = linear_velocity
        self.M = M
        self.rotational_velocity = rotational_velocity
        self.cam_T_imu = cam_T_imu
        self.imu_T_cam = np.linalg.inv(self.cam_T_imu)
        self.len_t = len(t)
        self.imu_T_wld = np.zeros((4, 4, self.len_t))
        self.wld_T_imu = np.zeros((4, 4, self.len_t))
        self.sigma_imu = np.zeros((6, 6, self.len_t))
        self.W = 1e-3
        
        self.imu_T_wld[:,:,0] = np.eye(4)
        self.wld_T_imu[:,:,0] = np.eye(4)
        self.sigma_imu[:,:,0] = np.eye(6)
        
        # for convineience
        self.exp_term = np.zeros((6, 6))
         
        
    def initialization(self, flag):
        
        if flag == "SEPERATE":
            self.len_feat = self.features.shape[1]
            self.Identity_mat = np.eye(3 * self.len_feat)
            self.mu0_feat, self.mu_feat = np.zeros((4,self.len_feat)), np.zeros((4,self.len_feat))
            self.V = 1e2*np.eye(4)
            self.P = np.hstack((np.eye(3),np.zeros((3,1)))) #projection matrix
            self.sigma_feat = np.eye(3*self.len_feat)
        
        if flag == "FUSED":
            self.len_feat = self.features.shape[1]
            self.Identity_mat = np.eye(3 * self.len_feat)
            # include IMU pose
            self.mu_feat = np.zeros((4,self.len_feat))
            self.mu0_feat, self.mu = np.zeros((4,self.len_feat)), np.zeros((4,self.len_feat + 4))
            self.V = 1e2*np.eye(4)
            self.P = np.hstack((np.eye(3),np.zeros((3,1)))) #projection matrix
            # include 6*6 IMU covariance
            self.Sigma = np.eye(3*self.len_feat + 6)
            
    def extract_Sigma(self, i, Exp):
        """
        update the covariance of the imu and landmarks seperately
        (A) Landmark sigma: 3*features x 3*features
        (B) cross term 1
        (C) cross term 2 = transpose(B)
        (D) IMU sigma: 6*6
        Sigma = [ A  B ]
                [ C  D ]
        """
        self.Sigma[-6:, -6:] = self.sigma_imu[:, :, i]
        self.Sigma[:3*self.len_feat, 3*self.len_feat:] = self.Sigma[:3*self.len_feat, 3*self.len_feat:] @ Exp.T
        self.Sigma[3*self.len_feat:, :3*self.len_feat] = self.Sigma[:3*self.len_feat, 3*self.len_feat:].T
            
    
    def prediction(self, i):       
        
        u = np.hstack((self.linear_velocity[:,i],self.rotational_velocity[:,i])) # u = [v w]
        self.imu_T_wld[:,:,i] = expm(hat(-self.dt[i]*u)) @ self.imu_T_wld[:,:,i-1]
        self.exp_term = expm(-self.dt[i]*curly_hat(u))
        self.sigma_imu[:,:,i] = self.exp_term @ self.sigma_imu[:,:,i-1] @ self.exp_term.T + self.dt[i]**2 * self.W
        self.wld_T_imu[:,:,i] = np.linalg.inv(self.imu_T_wld[:,:,i]) # IMU pose    
        
    
    def motion_update(self, j, ind, cur_feat, feat_head):
        # EKF update: IMU
        H = np.zeros((4*self.N_feat,6))
        for k in range(self.N_feat):
            H[4*k:4*k+4] = self.H_function(j, k, ind,"imu")
               
        K = self.sigma_imu[:,:,j] @ H.T @ np.linalg.inv( H @ self.sigma_imu[:,:,j] @ H.T + np.kron(np.eye(self.N_feat),self.V))
        self.imu_T_wld[:, :, j] = expm(hat(K @ (cur_feat - feat_head).flatten('F'))) @ self.imu_T_wld[:, :, j]
        self.wld_T_imu[:,:,j] = np.linalg.inv(self.imu_T_wld[:,:,j])
        self.sigma_imu[:,:,j] = (np.eye(6, dtype='float32') - K @ H) @ self.sigma_imu[:,:,j]
           
        
    def landmark_update(self, j, ind, cur_feat, N_feat):
               
        # EKF features update
        H = np.zeros((4*N_feat,3*self.len_feat))
        feat_head = self.M @ pi_func(self.cam_T_imu , self.imu_T_wld[:,:,j] , self.mu_feat[:,ind])
        
        for k in range(N_feat):
            H[4*k:4*k+4, 3*ind[k]:3*ind[k]+3] = self.M @ der_pi(self.cam_T_imu @ self.imu_T_wld[:,:,j] @ self.mu_feat[:,ind[k]].reshape(4, -1)) \
                                                @ self.cam_T_imu @ self.imu_T_wld[:,:,j] @ self.P.T 
        K = self.sigma_feat @ H.T @ np.linalg.inv( H @ self.sigma_feat @ H.T + np.kron(np.eye(N_feat),self.V))       
        mu_feat_update = (self.mu_feat.flatten('F') + np.kron(np.eye(self.len_feat), self.P.T) \
                        @ K @ (cur_feat - feat_head).flatten('F')).reshape(self.len_feat, 4).T
        self.sigma_feat = np.dot((self.Identity_mat - np.dot(K, H)), self.sigma_feat)
        
        return feat_head, mu_feat_update
    
    def EKF_Fused(self, j, N_feat, ind, cur_feat):

        # landmarks & pose update via EKF
        H = np.zeros((4*N_feat, 3*self.len_feat + 6))
        feat_head = self.M @ pi_func(self.cam_T_imu , self.imu_T_wld[:,:,j] , self.mu[:,ind])
        #print(feat_head.shape)
        for k in range(N_feat):
            H[4*k:4*k + 4, 3*ind[k]:3*ind[k] + 3] = self.H_function(j, k, ind,"ldk")
            H[4*k:4*k + 4, -6:] = self.H_function(j, k, ind,"imu")
            print(self.H_function(j, k, ind,"imu").shape)

        K = self.Sigma @ H.T @ np.linalg.inv(H @ self.Sigma @ H.T + np.kron(np.eye(N_feat), self.V))
        #print(K.shape)
        self.test = np.kron(np.eye(self.len_feat), self.P.T)
        # update landmarks mean
        self.mu[:, :self.len_feat] = (self.mu[:, :self.len_feat].flatten('F') + np.kron(np.eye(self.len_feat), self.P.T) @ K[:3*self.len_feat, :] @
                           (cur_feat - feat_head).flatten('F')).reshape(self.len_feat, 4).T
        
        # update IMU pose mean
        self.mu[:, self.len_feat:] = expm(hat(K[-6:, :] @ (cur_feat - feat_head).flatten('F'))) @ self.imu_T_wld[:, :, j]

        # update fused covariance
        self.Sigma = np.dot((np.eye(3*self.len_feat + 6) - K @ H), self.Sigma)

        # update record
        self.mu_feat = self.mu[:, :self.len_feat]
        self.imu_T_wld[:, :, j] = self.mu[:, self.len_feat:]
        self.wld_T_imu[:, :, j] = np.linalg.inv(self.imu_T_wld[:, :, j])
        self.sigma_imu[:, :, j] = self.Sigma[-6:, -6:]
    
    def find_indicies(self, j):
        #ind = np.where(features[0,:,j] != -1)[0]
        ind = np.flatnonzero(self.features[0, :, j] != -1) 
        cur_feat = self.features[:,ind,j]
        N_feat = len(ind)           
        return ind, cur_feat, N_feat
    
    def first_feature_update(self, j, N_feat, ind, flag):
        
        # find and add features that appear for the first time
        #new_ind = np.where(mu_feat[3,ind]==0)[0]
        new_ind = ind[np.flatnonzero(self.mu0_feat[-1, ind] == 0)]
        cur_new_feat = self.features[:,new_ind,j]
        if flag=="SEPERATE":
            self.mu_feat[:,new_ind] = self.mu0_feat[:,new_ind] = \
                feature_T_xyz(cur_new_feat, self.M, self.wld_T_imu[:,:,j], self.imu_T_cam)
        if flag=="FUSED":
            self.mu[:,new_ind] = self.mu0_feat[:,new_ind] = \
                feature_T_xyz(cur_new_feat, self.M, self.wld_T_imu[:,:,j], self.imu_T_cam)
   
    
    def H_function(self, j, k, ind, flag):
        """
        first order Taylor expansion
        """
        if flag == "ldk":
            return self.M @ der_pi(self.cam_T_imu @ self.imu_T_wld[:, :, j] @ self.mu[:, ind[k]].reshape(4, -1)) @ \
                   self.cam_T_imu @ self.imu_T_wld[:, :, j] @ self.P.T
                   
        if flag == "imu":
            return self.M @ der_pi(self.cam_T_imu @ self.imu_T_wld[:, :, j] @ self.mu[:, ind[k]]) @ \
                   self.cam_T_imu @ dot(self.imu_T_wld[:, :, j] @ self.mu[:, ind[k]].reshape(4, -1))
    

        
    