# Visual-Inertial SLAM 

![img](https://github.com/coldhenry/Visual-Inertial-SLAM/blob/master/pic/long-view.gif)

## Introduction

### Objective

In this paper, we aims to **build a map that shows the trajectory** where a car traveled and landmarks that were observed by the car. To reach the target, filtering is needed to derive a result that resists the noise coming from the environment and sensors. 

Therefore, we introduced a classic method of SLAM, which applies the **Extended Kalman Filter (EKF)** to estimate the true location of the car as well as the right position of the features it observed. 

### Extended Kalman Filter

Kalman Filtering is one of the **Bayes Filtering**. However, it has assumptions that the motion model and the observation model is linear in the state and affected by Gaussian noise. The EKF method expanded the assumptions by canceling the linear limitations, but forcing the predicted probability density functions and updated probabilities density functions to be Gaussian. 

### Implementation of EKF

There are two steps for implementing EKF, including a **prediction** step and an **update** step. We would estimate the positions of the car by its **IMU pose** and the position of landmarks by the pixel observed from **the stereo camera**. Through the EKF algorithm, the positions of the car and the landmarks would be updated to a more likely option. 



## Technical Approach

In this section, I would introduce how Extended Kalman Filter is applied to visual-inertial SLAM. I break down this task into three parts according to the request of the assignment: Localization-only problem, mapping-only problem, and the combination visual-inertial SLAM. 

please redirect to the [report](https://github.com/coldhenry/Visual-Inertial-SLAM/blob/master/report.pdf) for more information.


## Results

There are three datasets which are collected in different routes. In each dataset, we demonstrate three graphs to show the performance of Visual-Inertial SLAM. First graph indicates the difference between the trajectory generated from IMU data and the one that updated with both visual information and IMU poses. Second graph shows the landmarks that captured by the camera and the positions updated by the filter. The third graph presents the covariance of the IMU poses.

![img1](https://github.com/coldhenry/Visual-Inertial-SLAM/blob/master/pic/1.png)
![img1](https://github.com/coldhenry/Visual-Inertial-SLAM/blob/master/pic/2.png)
![img1](https://github.com/coldhenry/Visual-Inertial-SLAM/blob/master/pic/3.png)
