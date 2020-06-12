# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:52:28 2020

@author: paulg
"""

import numpy as np 
import cv2 


# this algorithm is made to do a color filter, for example for glass classification
#  for the example
# transparent: Hmin:0, Hmax:179 , Satmin:0, Satmax:29, Valmin:0, Valmax:255, 
# dark amber: Hmin:0, Hmax:12/13 , Satmin:85, Satmax:255, Valmin:32, Valmax:255, 
# Vert fonce: Hmin:30, Hmax:57, Satmin:130, Satmax:233, Valmin:0, Valmax:255, 
# antique green and amber: Hmin:18, Hmax:28 , Satmin:85, Satmax:255, Valmin:0, Valmax:255, 

def empty(a):
    pass

cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars",640,240)

#creation of trackbars
cv2.createTrackbar("Hue min","Trackbars",0,179,empty)
cv2.createTrackbar("Hue max","Trackbars",19,179,empty)
cv2.createTrackbar("Sat min","Trackbars",110,255,empty)
cv2.createTrackbar("Sat max","Trackbars",240,255,empty)
cv2.createTrackbar("Val min","Trackbars",153,255,empty)
cv2.createTrackbar("Val max","Trackbars",255,255,empty)

img=cv2.imread('glass2.png')
imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

#funtion to put image next to another 
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


# update of the values 
while True: 
    h_min=cv2.getTrackbarPos("Hue min","Trackbars")
    h_max=cv2.getTrackbarPos("Hue max","Trackbars")
    s_min=cv2.getTrackbarPos("Sat min","Trackbars")
    s_max=cv2.getTrackbarPos("Sat max","Trackbars")
    v_min=cv2.getTrackbarPos("Val min","Trackbars")
    v_max=cv2.getTrackbarPos("Val max","Trackbars")

    print(h_min,h_max,s_min,s_min,v_min,v_max)
    lower=np.array([h_min,s_min,v_min])
    upper=np.array([h_max,s_max,v_max])
    
    mask=cv2.inRange(imgHSV,lower,upper)
    imgResult=cv2.bitwise_and(img,img,mask=mask)
    # cv2.imshow("Image",img)
    # cv2.imshow("Image HSV",imgHSV)
    # cv2.imshow("Mask",mask)
    # cv2.imshow("Result",imgResult)
    
    imgStack=stackImages(0.6,([img,imgHSV],[mask,imgResult]))
    cv2.imshow('stacked images',imgStack)
    cv2.waitKey(1)