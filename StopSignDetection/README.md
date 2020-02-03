# Color Segmentation and Pattern Detection
<img src="https://github.com/coldhenry/Estimation-and-Sensing-of-Robotics/blob/master/StopSignDetection/pic/img2.png" alt="layout" width="200" /><img src="https://github.com/coldhenry/Estimation-and-Sensing-of-Robotics/blob/master/StopSignDetection/pic/img3.png" alt="layout" width="200" />
<img src="https://github.com/coldhenry/Estimation-and-Sensing-of-Robotics/blob/master/StopSignDetection/pic/img5.png" alt="layout" width="200" /><img src="https://github.com/coldhenry/Estimation-and-Sensing-of-Robotics/blob/master/StopSignDetection/pic/img69.png" alt="layout" width="200" />



## Introduction

This project aims to detect stop signs in an image and using a bounding box to label these stop signs. To achieve this goal, the process we need go through contains: to train a probabilistic color model from image data and use it to segment unseen images, detect stop signs in an image, and draw bounding boxes around them. 

Through this project, we could learn different types of supervised-learning approaches such as Simple Gaussian Generative Model, Gaussian Mixture Model, or Logistic Regression. Also, by implementing one of these methods into the given dataset, we could see both the strengths and the limits of the particular method and need to come up with several modifications to enhance the performance. 

In this paper, to classify the color in an image, I implemented the Naïve Bayes method. After deriving the classification results, I used the module Regionprop to analyze each region that exists on an image and filtered several regions out by applying some constraints on properties. 



## Problem Formulation

In this project, I will first break down each image of trainset into assigned pieces for further analysis. The whole trainset should be preprocessed before giving it to the classification function. Also, the output of the function is decided too. 

1. **Labeled data classification**

   First thing is to segment images into several categories of colors. In this project, I observed the trainset and decided to use 6 categories, including “RED”, “OTHER RED”, “BLUE”, “ORANGE”, “BROWN”, “OTHER”. Each pixel uses the same color code – RGB model. Therefore, the dataset is classified into an array of (n, 3), which number n refers to the size of each category.

2. **The input of the color segmentation function**

   To speed up the calculation, we will do the dimension reduction on the raw image. We turn the images that have 3-dimensions (width, height, color code) into a 2-dimensional array that has a size (width*height, 3). 

3. **The output of the color segmentation function**

   The output should be a binary mask that has a size that is as same as the image.

4. **Input / Output of the bounding box function**

   Input would be the original image that we are trying to classify color on and detect the stop sign. The output is coordinates of the boxes that represent the detection of the stop sign. The data structure of the boxes is lists of a list.

   Note that since that the autograder of Gradescope uses the coordinates different from the settings of Regionprop. It needs some modifications to meet the requirements. 

5. **The objective of the project**

   Finally, the objective of the project is to ensure the mask derived from the classification function is correct. Only the red objects are collected, and to ensure the boxes I derive are as same as the standard answer or lie in an acceptable interval. 

   

## Technical Approach

I choose the Gaussian Naïve Bayes (GNB) method as my model. This method uses a generative model for discrete labels, which are the six categories of colors. The assumption for this method is that, when conditioned on each label, the dimensions of a pixel are independent, which indicates that the color code R, G, and B has an independent relationship. The equations of the model:

|    <img src="https://github.com/coldhenry/Estimation-and-Sensing-of-Robotics/blob/master/StopSignDetection/pic/1.png" alt="layout" width="200" />  |  (1) |
| ---- | ---: |
|   <img src="https://github.com/coldhenry/Estimation-and-Sensing-of-Robotics/blob/master/StopSignDetection/pic/2.png" alt="layout" width="200" />    |  (2) |
|   <img src="https://github.com/coldhenry/Estimation-and-Sensing-of-Robotics/blob/master/StopSignDetection/pic/3.png" alt="layout" width="160" />  |  (3) |

Note that the symbol <img src="https://github.com/coldhenry/Estimation-and-Sensing-of-Robotics/blob/master/StopSignDetection/pic/3_1.png" alt="layout" width="60" />  refers to the value one if the category is k.

To build up the Gaussian model, we need the parameters mean and standard deviation. For the Gaussian Naïve Bayes method, I use the Maximum Likelihood Estimation (MLE) to maximize the likelihood of the trainset given the parameters.

### Parameters in Gaussian Model:

#### A.  Priors

After deriving six subsets of the trainset, I derived the prior probabilities of each color by calculating what portion does a particular color category is compared to all pixels. The equations are:

|  <img src="https://github.com/coldhenry/Estimation-and-Sensing-of-Robotics/blob/master/StopSignDetection/pic/4.png" alt="layout" width="160" />    |  (4) |
| :--- | ---: |
|  <img src="https://github.com/coldhenry/Estimation-and-Sensing-of-Robotics/blob/master/StopSignDetection/pic/5.png" alt="layout" width="200" />     |  (5) |

#### B.  Means

   Mean of each category is derived by the following equations:

|   <img src="https://github.com/coldhenry/Estimation-and-Sensing-of-Robotics/blob/master/StopSignDetection/pic/6.png" alt="layout" width="200" />    |  (6) |
| ---- | ---: |

#### C. Covariance

|   <img src="https://github.com/coldhenry/Estimation-and-Sensing-of-Robotics/blob/master/StopSignDetection/pic/7.png" alt="layout" width="200" />    |  (7) |
| ---- | ---: |

#### D. The output of the GNB classifier

After obtains the MLE estimates of the priors and the parameters. The classifier could produce the output by giving a category that maximizes the likelihood of a test example as following:

| <img src="https://github.com/coldhenry/Estimation-and-Sensing-of-Robotics/blob/master/StopSignDetection/pic/8.png" alt="layout" width="200" />      |  (8) |
| ---- | ---: |
|  <img src="https://github.com/coldhenry/Estimation-and-Sensing-of-Robotics/blob/master/StopSignDetection/pic/9.png" alt="layout" width="120" />     |  (9) |

  Or this could also be derived in the log-likelihood form:

|<img src="https://github.com/coldhenry/Estimation-and-Sensing-of-Robotics/blob/master/StopSignDetection/pic/10.png" alt="layout" width="200" />      | (10) |
| ---- | ---: |
| <img src="https://github.com/coldhenry/Estimation-and-Sensing-of-Robotics/blob/master/StopSignDetection/pic/11.png" alt="layout" width="120" />      | (11) |

   Use the classifier to go through each data point, and use the following condition to build up the binary mask:

For each point, 

|  <img src="https://github.com/coldhenry/Estimation-and-Sensing-of-Robotics/blob/master/StopSignDetection/pic/12.png" alt="layout" width="200" />     | (12) |
| ---- | ---: |

The mask contains all shapes that are classified as red. In the next step, we are trying to find out those are stop signs instead of other objects such as red lights or red cars. To filter out these objects, I first need to preprocess the mask to ensure the quality of the mask. In general, most masks would have some defects like small noises, the incomplete shape of the stop sign (e.g. lose one part of it or even split into half). To fix the problem, I have utilized the module openCV to do morphological operations such as opening and closing. After trial and errors, the best number and order of operations I had were:

1.  Dilate the mask one time

2. Operate the closing one time

After mask preprocessing, I use the module regionprop and derived the properties of each region, and set up some qualifications that a stop sign should meet with: 

* Area of contours
* The aspect ratio
* Extent between the bounding box and the stop sign
* The Euler number

There are 200 images in the trainset and 100 images contain one or more stop signs. Although they are the same objects, the detection of these stop signs could be both easy and challenging since they are ever-changing under various scenarios. A human can easily recognize these signs by combining much information at the same time, while the machine could only rely on the model to distinguish the stop sign from other similar objects such as red lights and red cars.

To let the machine understand what a stop sign should be, I provided four crucial qualifications to separate the stop sign from other red objects. All qualifications have a score on it. Whenever the region has met one qualification, it would have some points. In the end, there will be a bar to see if this region is qualified to be recognized as a stop sign.

#### A.  Contour Area

This is a factor that I take into consideration after tuning parameters for a period of time but actually important for introducing more complicated classification. The area of each contour gives you a sense that what the object might look like. If it is big, we could have an assumption that the pattern might be very clear. All details could be clearly present on the mask and therefore easy to distinguish if it is a stop sign or not as long as sufficient qualifications have given. However, if the area is small, the quality of the mask might be worse than you expected. Even if the contour is a stop sign, the features that a stop sign should have might be ambiguous in such a small area. In addition, when the area of a contour is small, the non-stop-sign objects have a higher chance to share the same properties with a stop sign. For example, they could have a perfect aspect ratio, a good extent between the bounding box and the red area, and even holes in it. Therefore, it is reasonable to separate these cases and give different qualifications. 

#### B.  Aspect Ratio

The aspect ratio refers to the ratio between the height and width of the bounding box of the contour. Since that the standard stop sign is a regular octagon. The aspect ratio is supposed to be 1. Nevertheless, the value would change if the stop sign doesn’t face the camera directly. As a result, I divided the ratio value into two sections and gave a score that depends on what section the contour lies in to. By doing so, I can filter some regions that are too tall or too fat.

#### C. Extent 

   After ensuring that the shape of the bounding box is a square-like shape. I further look at the extent. In general, a regular hexagon should cover 82% of the area of a minimal square that contains the hexagon. However, since the actual shape of stop signs changes from image to image. I set an acceptable range from 0.6 to 0.9. 

#### D. Euler Number

Euler number here refers to the Euler characteristic of the region, which is computed as number objects subtracted by a number of holes (using 8-connectivity) [1]. In short, it gives the number of holes in a region. This is a powerful but also risky tool. Ideally, it could provide a neat method to detect whether a region is a stop sign since there are four characters in a stop sign and would be holes in a mask. On the other hand, we might miss some stop signs since the holes in the region are filled by the morphological operation, and therefore it can not be used as a necessary condition.



## Results

<img src="https://github.com/coldhenry/Estimation-and-Sensing-of-Robotics/blob/master/StopSignDetection/pic/13.png" alt="layout" width="640" />

After building the model for color classification and setting up the qualifications for detecting the stop sign and drawing the bounding box, we are now ready for applying the function on the trainset again to see if the initial parameters are working.

Before submitting the code to the autograder to see the performance on the test set. No matter how many stop signs an image has, most of the cases could be perfectly detected. Nevertheless, I found out some problems that are difficult to deal with. The problems are described as following:

### A.  Stop sign is failed to detect

As I mentioned in the previous section, to enhance the quality and completeness of regions in a mask. I did the morphological operation, including closing and dilate particularly. That operation did help me erase the noise and fill the defects in some cases. Nevertheless, the method leads to bad performance sometimes. 

As shown in Fig.1., two stop signs are presented in an image and in the mask. The shape of the upper one is not complete and has a notch on its top while the lower one has a great shape and is clearly a stop sign through inspection. However, through the algorithm I designed, it is classified as a disqualified region. The reason is that this type of stop sign lies in the decision boundary, which has slightly over the standard aspect and the extent. To address this problem, I reduced the bar of the aspect and extent a bit to ensure that the good one won’t be excluded from the filter.

### B.  The stop sign connects with other objects

This problem happened a lot while I used dilation too many times on the same mask to fix the notches. Since the gap between the stop sign and the other objects that are red is small, chances are they would connect to a single object.

As you can see, Fig. 2. shows the problem which the stop sign and the stick have an only little area to separate them. In fact, the original mask is clearly separated. But since that we need to apply morphological operations on each image, this kind of mask is not able to detect correctly. As a result, I modified the operations from using dilation or erosion separately to using closing. This could ensure that I wouldn’t make a change too dramatically.

### C. The stop sign is split into half

The last and the most difficult problem is that the stop sign in the mask is split into half. This problem tends to happen at the time when the mask is generated, not because of the morphological operations. This is basically a classification problem from the model. As shown in Fig. 3., the stop sign on the left, when put into the classification and turn into the mask, has been separated into two parts. However, this is a dilemma to deal with this type of problem. Since that, if you are trying to fill the gap, it would need several closing operations. At the same time, doing too many operations cause other normal stop signs to have a chance to expand to an undetectable shape. After trial and error, I still maintain the number of operations because most cases won’t have this scenario.
