# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 21:32:20 2020

@author: Henry Liu / A53306080
- Project: Color Segmentation
- Description:
    (1) Train a probabilistic color model using a set of labeled pixel values
    (2) Use the model to classify the colors on unseen test images
    (3) Detect a stop sign based on the color segmentation and the known stop-sign shape

"""
import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import closing
import matplotlib.pyplot as plt


COLORS = ['COLOR_STOP_SIGN_RED', 'COLOR_OTHER_RED','COLOR_BROWN', 
          'COLOR_ORANGE','COLOR_BLUE', 'COLOR_OTHER']

#%% Parameters

# Labeled data
red = np.zeros((1,3),dtype=int);
ored = np.zeros((1,3),dtype=int);
brown = np.zeros((1,3),dtype=int);
blue = np.zeros((1,3),dtype=int);
orange = np.zeros((1,3),dtype=int);
other = np.zeros((1,3),dtype=int);
not_red = np.zeros((1,3),dtype=int)
sign = np.zeros((1,3),dtype=int);

# MLE parameters
theta = np.zeros((6,1))
miu = np.zeros((6,3))
sigma = np.zeros((3,3,6))
var = np.zeros((3,3,6))

# Number of trainset
num_train = 200



#%% Load Labeled data

print("--------Load Labeled Data--------")
for i in range(1,num_train+1):
    if i%10==0:
        print("Loading...",i)    
    try:
        test = np.load("C:/Users/coldhenry/Desktop/hw1_starter_code/trainset_labeled/"+str(i)+".npz")
    except FileNotFoundError:
        print("WARNING: "+str(i)+"th file missing")
        pass
    RED = test['COLOR_STOP_SIGN_RED']
    if RED.size>0:
        red = np.concatenate((red,RED),axis=0)      
    
    O_RED = test['COLOR_OTHER_RED']
    if O_RED.size>0:
        ored = np.concatenate((ored,O_RED),axis=0)        
        #not_red = np.concatenate((not_red,O_RED),axis=0)
    
    BROWN = test['COLOR_BROWN']
    if BROWN.size>0:
        brown = np.concatenate((brown,BROWN),axis=0)
        #not_red = np.concatenate((not_red,BROWN),axis=0)
    
    BLUE = test['COLOR_BLUE']
    if BLUE.size>0:
        blue = np.concatenate((blue,BLUE),axis=0)
        #not_red = np.concatenate((not_red,BLUE),axis=0)
    
    ORANGE = test['COLOR_ORANGE']
    if ORANGE.size>0:
        orange = np.concatenate((orange,ORANGE),axis=0)
        #not_red = np.concatenate((not_red,ORANGE),axis=0)
    
    OTHER = test['COLOR_OTHER']
    if OTHER.size>0:
        other = np.concatenate((other,OTHER),axis=0)
        #not_red = np.concatenate((not_red,OTHER),axis=0)

#Remove the first row which is unnecessary        
red = red[1:,:]
ored = ored[1:,:]
brown = brown[1:,:]
blue = blue[1:,:]
orange = orange[1:,:]
other = other[1:,:]
not_red = not_red[1:,:]
print("--------Finish Loading--------")

#%% Gaussian Naive Bayes

#Calculate the prior probability
total_pixel = red.shape[0]+ored.shape[0]+blue.shape[0]+brown.shape[0]+orange.shape[0]+other.shape[0]
theta[0] = red.shape[0]/total_pixel
theta[1] = ored.shape[0]/total_pixel
theta[2] = blue.shape[0]/total_pixel
theta[3] = brown.shape[0]/total_pixel
theta[4] = orange.shape[0]/total_pixel
theta[5] = other.shape[0]/total_pixel

# calculate the mean of each class
miu[0] = np.mean(red,axis=0)
miu[1] = np.mean(ored,axis=0)
miu[2] = np.mean(blue,axis=0)
miu[3] = np.mean(brown,axis=0)
miu[4] = np.mean(orange,axis=0)
miu[5] = np.mean(other,axis=0)

#calculate the covariance of each class
var[:,:,0] = np.cov(red.T)
var[:,:,1] = np.cov(ored.T) 
var[:,:,2] = np.cov(blue.T) 
var[:,:,3] = np.cov(brown.T) 
var[:,:,4] = np.cov(orange.T) 
var[:,:,5] = np.cov(other.T) 


#%% Classifier

#test image number
number = 68
#read the image and turn into RGB color code
IMG_FILE = "C:/Users/coldhenry/Desktop/hw1_starter_code/trainset/"+str(number)+".jpg"
img_raw = cv2.imread(IMG_FILE)
img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
#initiate the mask and data
mask = np.zeros((img.shape[0],img.shape[1]),dtype=int)
data = img.reshape(img.shape[0]*img.shape[1],3)
      
def Guassian_Classifier(data,miu,var):
    cov = np.linalg.cholesky(np.linalg.inv(var))
    exponent = np.exp(-np.sum(np.square(np.dot((data-miu),cov)), axis=1)/2)
    constant = np.sqrt((2*np.pi)**3*np.linalg.det(var))
    return exponent / constant

def Log_Guassian_Classifier(data,miu,var):
    cov = np.linalg.cholesky(np.linalg.inv(var))
    exponent = -np.sum(np.square(np.dot((data-miu),cov)), axis=1)/2
    const1 = np.log10(2*np.pi)
    #print(const1)
    constant = float(const1)*(-1.5)-np.log10(np.linalg.det(var))/2
    return exponent + constant

#put the results of each classifier into an array
val = np.zeros((img.shape[0]*img.shape[1],6))
for i in range(6):
    val[:,i] = Guassian_Classifier(data,miu[i,:],var[:,:,i])

#find the indexes of points that are label red and other
tmp = np.argmax(val,axis=1)
ind_other = np.where(tmp!=0)
ind_red = np.where(tmp==0)

#create a mask that has value 255 on the class red and 0 on other classes
tmp[ind_red]=255
tmp[ind_other]=0
mask = tmp.reshape(img.shape[0],img.shape[1])




# change the value type to uint8 for threshold
mask = mask.astype(np.uint8)
# change to Binary image
ret,mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
# do morphological operation
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#show the mask
plt.imshow(mask)
plt.savefig("mask"+str(number)+".png")
plt.show()

#use regionprop to derive the properties from the mask
label_img = label(mask,neighbors= 4, connectivity=1)
props = regionprops(label_img)
boxes = [] #boxes for submit
boxes_test = [] #all boxes
for prop in props:
    
    mi_r, mi_c, ma_r, ma_c = prop.bbox
    height = ma_r - mi_r
    width = ma_c - mi_c
    box_area = height*width
    aspect = height/width
    extent = prop.extent
    euler = prop.euler_number
    
    boxes_test.append([mi_c, mi_r, ma_c, ma_r])
    
    points = 0
    if width*height<300:
        points -= 5
    if box_area >9000:
        if extent>0.6 and extent<0.9:
            points += 2
        if aspect >= 0.7 and aspect <1.5:
            points += 1
        if aspect > 1 and aspect <=1.4:
            points += 2
    if box_area < 9000:
        if extent>0.7 and extent<0.8:
            points += 2
        if aspect > 1 and aspect <=1.4:
            points += 1
    if euler<0:
        points += 2
    if points >= 3:    
        boxes.append([mi_c, mi_r, ma_c, ma_r])
        #boxes.append([mi_c, img.shape[0]-ma_r, ma_c, img.shape[0]-mi_r])
        print("PASS points:"+str(points)+",  height:"+str(height)+", width:"+str(width)+", aspect:"+str(aspect)+", extent:"+str(extent)+", euler:"+str(euler))
        continue
    print("points:"+str(points)+",  height:"+str(height)+", width:"+str(width)+", aspect:"+str(aspect)+", extent:"+str(extent)+", euler:"+str(euler))


#draw boxes that are recognized as stop sign
for i in range(len(boxes_test)):
    cv2.rectangle(img_raw,(boxes_test[i][0],boxes_test[i][1]),(boxes_test[i][2],boxes_test[i][3]),(0, 0, 255),2)

#draw all boxes as a comparison
for i in range(len(boxes)):
    cv2.rectangle(img_raw,(boxes[i][0],boxes[i][1]),(boxes[i][2],boxes[i][3]),(0, 255, 0),5)

#resize the image into suitable size and show it
imgresize = cv2.resize(img_raw, (int(img.shape[1]/2), int(img.shape[0]/2)))
cv2.imshow('box',imgresize)
cv2.imwrite("img"+str(number)+".png",imgresize)
cv2.waitKey(0)




