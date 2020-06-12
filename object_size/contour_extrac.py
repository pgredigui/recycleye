# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 15:03:06 2020

@author: paulg
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:55:01 2020

@author: paulg
"""

from skimage.measure import compare_ssim
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


import xlsxwriter

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import subprocess as sp


import os.path

# methods for contour extraction

#structural similarity index
def Str_sim_in(background,image):
    
    grayA = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    
    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    
    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 80, 255,
    	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    c=c[:,0,:]
    return c 


#add the caracteristics of the extraction of the contour to allow more flexibility
    
def contour_extrac(image):
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(7,7),0)
    
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    
    
    cnts = cv2.findContours( edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    cnts = imutils.grab_contours(cnts)
    
    #extract the biggest contour
    c = max(cnts, key=cv2.contourArea)
    c=c[:,0,:]
    return c

imageA = cv2.imread('image2/IMG_2746.jpg')
imageB = cv2.imread('image2/IMG_2749.jpg')

imageB2 =imageB.copy()

cnt1=Str_sim_in(imageA,imageB)
cnt2=contour_extrac(imageB)

print(cv2.contourArea(cnt1))
print('area:',cv2.contourArea(cnt2))
print('perimeter:',cv2.arcLength(cnt1,True))
print('perimeter:',cv2.arcLength(cnt2,True))
rect = cv2.minAreaRect(cnt2)
print('rect:',rect)


cv2.drawContours(imageB,[cnt1],-1,(255,0,0),1)
cv2.drawContours(imageB2,[cnt2],-1,(255,0,0),1)


    
cv2.imshow("Imagestr",imageB)
cv2.imshow("Imagecnt",imageB2)
cv2.waitKey()



