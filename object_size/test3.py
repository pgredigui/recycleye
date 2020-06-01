# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:12:31 2020

@author: paulg
"""

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

import numpy

img = cv2.imread(r'top3.jpg')


def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center = rect[0]
    size = rect[1]
    angle = rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]
    #print("width: {}, height: {}".format(width, height))

    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height))

    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

cv2.imshow("Image",edged)
cv2.waitKey(0)


cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

c = max(cnts, key=cv2.contourArea)

stencil = numpy.zeros(gray.shape).astype(gray.dtype)

color = [255, 255, 255]
cv2.fillPoly(stencil, [c], color)
result = cv2.bitwise_and(gray, stencil)

thresh= cv2.threshold(result, 45, 255, cv2.THRESH_BINARY)[1]

cv2.imshow("Image",thresh)
cv2.waitKey(0)
    

cv2.imwrite("result.jpg", result)

width=8.5

mult = 1.0 
img_box =  thresh.copy()
pixelsPerMetric = None

for cnt in cnts:
    if cv2.contourArea(cnt) < 100:
        continue
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img_box, [box], 0, (0,255,0), 2) # this was mostly for debugging you may omit

    img_crop= crop_rect(img, rect)
    size=rect[1]
    W=size[0]
    H=size[1]
    if pixelsPerMetric is None:
        pixelsPerMetric=W/width
	# compute the size of the object
    dimA=H / pixelsPerMetric
    dimB=W / pixelsPerMetric

    print(dimA,dimB)
    cv2.imshow("Image",img_crop)
    cv2.waitKey(0)

