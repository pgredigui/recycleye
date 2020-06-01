# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:35:47 2020

@author: paulg
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import imutils




#conversion of the image into binary scale here only the biggest contour is extracted to prevent issue where contour are detected 
#ans should not be


img = cv.imread('top.JPG')

cv.imshow("Image",img)
cv.waitKey(0)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

edged = cv.Canny(gray, 50, 100)
edged = cv.dilate(edged, None, iterations=1)
edged = cv.erode(edged, None, iterations=1)


cnts = cv.findContours( edged,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE )
cnts = imutils.grab_contours(cnts)

#extract the biggest contour
c = max(cnts, key=cv.contourArea)


#outside of the contour fillled in black
stencil = np.zeros(gray.shape).astype(gray.dtype)

color = [255, 255, 255]
cv.fillPoly(stencil, [c], color)
result = cv.bitwise_and(gray, stencil)

#indisde of the contour filled in white
thresh= cv.threshold(result, 0, 255, cv.THRESH_BINARY)[1]

cv.imshow("Image",thresh)
cv.waitKey(0)

cv.imwrite("result.jpg", thresh)

cnt=c
cnt=cnt[:,0,:]
cnt=np.int0(cnt)
cv.drawContours(img,[cnt],-1,(0,255,255),2)
    
    
cv.imshow("Image",img)
cv.waitKey(0)