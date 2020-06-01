# -*- coding: utf-8 -*-
"""
Created on Fri May 29 08:54:30 2020

@author: paulg
"""

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

#here the code is only for single objects
#extract all geometric caracteristics from a set of images. the first imge is the calibration
 

def contour_extraction(file):
    ### only return the biggest contour
    
    #load the image, here image is added because it is in another subfoldef
    img = cv2.imread(file)
    # convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    #extract the biggest contour
    c = max(cnts, key=cv2.contourArea)
    return c



### create the xls file 
    
# Create a workbook and add a worksheet.
workbook = xlsxwriter.Workbook('carac-geo.xlsx')
worksheet = workbook.add_worksheet('top')
worksheet2 = workbook.add_worksheet('side')


# Start from the first cell. Rows and columns are zero indexed.
row = 0
col = 0

title = ['Name', 'Area','Perimeter','length','height']

# create the name of the title
for t in (title):
    worksheet.write(row, col,t)

    col += 1
col = 0
row = 1

# absolute path need to be changed 
path="C:/Users/paulg/Documents/ENS/M2R/Courses/Projet/code/carac-geo/image"

files=os.listdir(path)

#parameters measured on the calibration pictures

area_ref=25
width_ref=5
length_ref=5
perimeter_ref=20

#calibration parameter

pixelsPerA= None
pixelsPerW= None
pixelsPerL= None
pixelsPerP= None


for f in files:
    f2=f
    f='image/'+f

    contour=contour_extraction(f)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    size=rect[1]
    W=size[0]
    L=size[1]
    A=cv2.contourArea(contour)
    P=cv2.arcLength(contour,True)

    if pixelsPerA is None:
        pixelsPerA=A/area_ref
        pixelsPerW=W/width_ref        
        pixelsPerL=L/length_ref
        pixelsPerP=P/length_ref
    A=A/pixelsPerA
    W=W/pixelsPerW
    L=L/pixelsPerL
    P=P/pixelsPerP
    carac_geo=[f2,A,W,L,P]
    for c in (carac_geo):
        worksheet.write(row, col,c)
        col+=1
    row+=1
    col=0

workbook.close()