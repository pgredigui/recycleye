# -*- coding: utf-8 -*-
"""
Created on Sun May 17 14:08:50 2020

@author: paulg
"""



from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import subprocess as sp


### the goal of this code is to extract area  boxand draw the contorus of the objects

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


image = cv2.imread('test.JPG')

cv2.imshow("Image",image)
cv2.waitKey(0)

area_ref=45.9

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)




cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)
pixelsPerArea = cv2.contourArea(cnts[0])/area_ref


for c in cnts:
    
    if cv2.contourArea(c) < 100:
        continue
    orig = image.copy()
    # compute the rotated bounding box of the contour
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    box = perspective.order_points(box)
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    

    cnt=c
    cnt=cnt[:,0,:]
    cv2.drawContours(orig,[cnt],-1,(0,255,255),2)
    
    area=cv2.contourArea(c)/pixelsPerArea
    cv2.putText(orig, "{:.1f}cm^2".format(area),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)
    cv2.imshow("Image", orig)
    cv2.waitKey(0)
    