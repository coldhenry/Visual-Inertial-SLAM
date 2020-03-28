# Visual-Inertial SLAM 

## Introduction

​	In this paper, we aims to build a map that shows the trajectory where a car traveled and landmarks that were observed by the car. To reach the target, filtering is needed to derive a result that resists the noise coming from the environment and sensors. Therefore, we introduced a classic method of SLAM, which applies the Extended Kalman Filter (EKF) to estimate the true location of the car as well as the right position of the features it observed. Kalman Filtering is one of the Bayes Filtering. 

​	However, it has assumptions that the motion model and the observation model is linear in the state and affected by Gaussian noise. The EKF method expanded the assumptions by canceling the linear limitations, but forcing the predicted probability density functions and updated probabilities density functions to be Gaussian. 

​	There are two steps for implementing EKF, including a prediction step and a update step. We would estimate the positions of the car by its IMU pose and the position of landmarks by the pixel observed from the stereo camera. Through the EKF algorithm, the positions of the car and the landmarks would be updated to a more likely option. 

## Problem Formulation

### A. IMU-based Localization via EKF Prediction

In the first problem, we are trying to localize the car at each time step based on IMU pose only. Therefore, we would apply the EKF prediction step.

* Input: SE(3) kinematics. 

* Output 1: IMU pose over time.

* Output 2: the covariance of the IMU pose.

The process contains defining a Gaussian distribution given the previous state and observations that represents the inverse IMU pose, predicting its next state using the motion model.

### B. Landmark Mapping via EKF Update

For the second problem, we assume the predicted IMU trajectory from part A is correct and focus on estimating the landmark positions. 

* Input 1: Unknown landmarks positions.

* Input 2: Visual Observation at time.

* Output 1: the mean of landmarks.

* Output 2: the covariance of landmarks.

The process contains representing the landmark as a Gaussian distribution, predicting observations based on the mean at the last step, then updating the mean and the covariance using the EKF method.

### C. Visual-Inertial SLAM

For the last problem, we combined what we’ve done in the previous two parts: the IMU prediction step from part A with the landmark update step from part B. In addition, we update the IMU pose in this part based on the stereo camera observation model to obtain a complete visual-inertial SLAM algorithm.

* Input 1: IMU pose over time.

* Input 2: Unknown landmarks positions. 

* Input 3: Visual Observation at time.

* Output 1: the mean of landmarks and IMU.

* Output 2: the covariance of landmarks and IMU.

## Technical Approach

​	In this section, I would introduce how Extended Kalman Filter is applied to visual-inertial SLAM. I break down this task into three parts according to the request of the assignment: Localization-only problem, mapping-only problem, and the combination visual-inertial SLAM. 





## Results

