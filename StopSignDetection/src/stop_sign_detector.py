# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:01:19 2020

@author: coldhenry
"""

import cv2
from skimage.measure import label, regionprops
import numpy as np

class StopSignDetector():
    def __init__(self):
        # initialize mean value for each class
        self.mean_dict = {}
        self.mean_dict['COLOR_STOP_SIGN_RED'] = np.array([161.656, 15.524, 26.4728])
        self.mean_dict['COLOR_OTHER_RED']	  = np.array([89.3373, 55.297, 56.0258])
        self.mean_dict['COLOR_BROWN']	  	  = np.array([111.774, 153.833, 205.296])
        self.mean_dict['COLOR_ORANGE']	  	  = np.array([133.4, 99.8498,  70.5891])
        self.mean_dict['COLOR_BLUE']	  	  = np.array([169.948, 114.914, 77.8291])
        self.mean_dict['COLOR_OTHER']	  	  = np.array([110.513, 112.163, 106.194])
        
        # initialize covariance matrix for each class
        self.cov_dict = {}
        self.cov_dict['COLOR_STOP_SIGN_RED']  = np.array([[2918.55, 336.986, 500.563],
           													  [336.986, 575.618, 526.843],
           													  [500.563, 526.843, 638.925]])
        self.cov_dict['COLOR_OTHER_RED']  	  = np.array([[1589.53, 718.233, 694.727],
           													  [718.233, 844.059, 831.001],
           													  [694.727, 831.001, 894.026]])
        self.cov_dict['COLOR_BROWN']  		  = np.array([[2244, 1813.7, 1027.7],
           													  [1813.7, 1730.1, 1120.93],
           													  [1027.7, 1120.93, 978.218]])
        self.cov_dict['COLOR_ORANGE']  		  = np.array([[3867.05, 2848.37, 1726.33],
           													  [2848.37, 2452.67, 1838.71],
           													  [1726.33, 1838.71, 1812.27]])
        self.cov_dict['COLOR_BLUE']  		  = np.array([[2717.98, 2472.1, 1186.04],
           													  [2472.1, 3002.01, 1676.97],
           													  [1186.04, 1676.97, 1610.37]])
        self.cov_dict['COLOR_OTHER']  		  = np.array([[3549.22, 3355.68, 3313.07],
           													  [3355.68, 3323.33, 3299.59],
           													  [3313.07, 3299.59, 3601.72]])
        # initialize prior for each class
        self.prior_dict = {}
        self.prior_dict['COLOR_STOP_SIGN_RED'] = 0.00787181
        self.prior_dict['COLOR_OTHER_RED'] 	   = 0.00900792
        self.prior_dict['COLOR_BROWN'] 		   = 0.187947
        self.prior_dict['COLOR_ORANGE'] 	   = 0.0449346
        self.prior_dict['COLOR_BLUE'] 		   = 0.00108392
        self.prior_dict['COLOR_OTHER'] 		   = 0.749154
        
    def Guassian_Classifier(self, data, miu, var):
        cov = np.linalg.cholesky(np.linalg.inv(var))
        exponent = -np.sum(np.square(np.dot((data-miu),cov)), axis=1)/2
        const1 = np.log10(2*np.pi)
        constant = float(const1)*(-1.5)-np.log10(np.linalg.det(var))/2
        return exponent + constant   
     
    def segment_image(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask_img = np.zeros((img.shape[0],img.shape[1]),dtype=int)
        data = img.reshape(img.shape[0]*img.shape[1],3)
        val = np.zeros((img.shape[0]*img.shape[1],6))
        
        val[:,0] = self.Guassian_Classifier(data, self.mean_dict['COLOR_STOP_SIGN_RED'], self.cov_dict['COLOR_STOP_SIGN_RED'])
        val[:,1] = self.Guassian_Classifier(data, self.mean_dict['COLOR_OTHER_RED'], self.cov_dict['COLOR_OTHER_RED'])
        val[:,2] = self.Guassian_Classifier(data, self.mean_dict['COLOR_BROWN'], self.cov_dict['COLOR_BROWN'])
        val[:,3] = self.Guassian_Classifier(data, self.mean_dict['COLOR_ORANGE'], self.cov_dict['COLOR_ORANGE'])
        val[:,4] = self.Guassian_Classifier(data, self.mean_dict['COLOR_BLUE'], self.cov_dict['COLOR_BLUE'])
        val[:,5] = self.Guassian_Classifier(data, self.mean_dict['COLOR_OTHER'], self.cov_dict['COLOR_OTHER'])
        
        # find the points that are classified as red and others
        tmp = np.argmax(val,axis=1)
        ind_other = np.where(tmp!=0)
        ind_red = np.where(tmp==0)
        
        # assign the value 1 to the indexes that the points are red 
        # and value 0 to the indexes that the points are other colors
        tmp[ind_red]=255
        tmp[ind_other]=0
        
        mask_img = tmp.reshape(img.shape[0],img.shape[1])
       
        return mask_img
    
    def get_bounding_box(self,img):
        
        mask = self.segment_image(img)
        # change the value type to uint8 for threshold
        mask = mask.astype(np.uint8)
        # change to Binary image
        ret,thresh1 = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        
        label_img = label(mask, connectivity=1)
        props = regionprops(label_img)
        boxes = []
        for prop in props:
            mi_r, mi_c, ma_r, ma_c = prop.bbox
            height = ma_r - mi_r
            width = ma_c - mi_c
            box_area = height*width
            aspect = height/width
            extent = prop.extent
            euler = prop.euler_number
            
            points = 0
            if  box_area<300:
                points -= 5
            if aspect >= 0.7 and aspect <2:
                points += 1
            if aspect > 1 and aspect <=1.4:
                points += 2
            if extent>0.6 and extent<0.9:
                points += 2
            if euler<0:
                points += 2
            if points >= 3:    
                #boxes.append([mi_c, mi_r, ma_c, ma_r])
                boxes.append([mi_c, img.shape[0]-ma_r, ma_c, img.shape[0]-mi_r])
            print("points:"+str(points)+", properties: height:"+str(height)+", width:"+str(width)+", aspect:"+str(aspect)+", extent:"+str(extent)+", euler:"+str(euler))
        
        if(len(boxes) > 1):
            boxes.sort(key = lambda x: x[0])
        
        return boxes
        
        
        
if __name__ == '__main__':
    folder = "trainset"
    my_detector = StopSignDetector()
    img = cv2.imread('trainset/74.jpg')
    mask_img = my_detector.segment_image(img)
    boxes = my_detector.get_bounding_box(img)
    for i in range(len(boxes)):
        cv2.rectangle(img,(boxes[i][0],boxes[i][1]),(boxes[i][2],boxes[i][3]),(0, 255, 0),4)
        
    imgresize = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
    cv2.imshow('box', imgresize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
#original test code       
#    for filename in os.listdir(folder):
#        img = cv2.imread(os.path.join(folder,filename))
#        mask_img = my_detector.segment_image(img)
#        boxes = my_detector.get_bounding_box(img)
#        
#        cv2.waitKey(0)
#        cv2.destroyAllWindow()
    
        
        
        
        
        
        
        
        
        
        