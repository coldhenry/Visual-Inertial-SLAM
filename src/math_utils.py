# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:17:44 2020

@author: coldhenry
"""
import numpy as np



def hat(u):
    """
    dimension 3:
        [0   -x3   -x2]
        [x3    0   -x1]
        [-x2  x1     0]
        
    dimension 6: se(3) 
        zeta = [a b].T (6*1)
        zeta_hat = [b_hat  a]
                   [    0  0]  (4*4)
    """
    u = u.squeeze()
    if u.ndim != 1:
        raise ValueError('hat operation no supported')
    if len(u) == 3:
        return np.array([[    0,    -u[2],  u[1]],
                         [ u[2],        0, -u[0]],
                         [-u[1],     u[0],     0]])
    elif len(u) == 6:
        return np.block([[hat(u[3:]), u[:3].reshape((3, 1))],
                         [np.zeros((1, 4))                 ]])
    else:
        raise ValueError('out of hat scope')
    

def curly_hat(u):
    """
    u = [v w].T (6*1)
    curly hat u = [hat(w) hat(v)]
                  [    0  hat(w)]  (6*6)
    """
    return np.block([[      hat(u[3:]), hat(u[:3])],
                     [np.zeros((3, 3)), hat(u[3:])]])
    
def feature_T_xyz(feat, M, wld_T_imu, imu_T_cam):
    """
    d: disparity, d = uL- uR
    x = (uL - cu)*z / fsu
    y = (vL - cv)*z / fsv
    m_ : homogeneous coordinate in camera frame
    """
    d = feat[0]-feat[2]
    z = -M[2,3]/d
    x = (feat[0]-M[0,2])*z / M[0,0]
    y = (feat[1]-M[1,2])*z / M[1,1]
    m_ = np.vstack((x,y,z,np.ones(x.shape)))
    # turn to the world frame
    m_wld = wld_T_imu @ imu_T_cam @ m_
    return m_wld
    
def pi_func(cam_T_imu , imu_T_wld , mu_feat):
    """
    eliminate the information of depth
    output : 4*1
    """
    x = cam_T_imu @ imu_T_wld @ mu_feat
    return (x / x[2])

def der_pi(q):
    """
    derivative of pi_func
    """
    q = q.squeeze()
    return (1/q[2])*np.array([[1, 0, -q[0]/q[2], 0],
                              [0, 1, -q[1]/q[2], 0],
                              [0, 0,          0, 0],
                              [0, 0, -q[3]/q[2], 1]])
    
def dot(u):
    """
    dot([s 1].T) = [I  -hat(s)]
                   [0        0] (4*6)
    """
    return np.block([[u[3]*np.eye(3), -hat(u[0:3])],
                     [ np.zeros((1, 6))           ]])
    
