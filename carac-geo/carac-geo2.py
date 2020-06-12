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

# extract the geometric caractestics and genrate an excel fle 

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
    return c 


#add the caracteristics of the extraction of the contour to allow more flexibility

#other config ite dila=6 and ite erode=3
def contour_extrac(image):
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(7,7),0)
    
    
    
    
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    
    
    cnts = cv2.findContours( edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    #extract the biggest contour
    c = max(cnts, key=cv2.contourArea)
    return c

def create_excel(file_name):
    workbook = xlsxwriter.Workbook(file_name)
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
    return workbook,worksheet,worksheet2
    
def carac_geo_extra(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    size=rect[1]
    W=min(size)
    L=max(size)
    A=cv2.contourArea(contour)
    P=cv2.arcLength(contour,True)
    return A,W,L,P

workbook,worksheet,worksheet2=create_excel('carac-geo.xlsx')


path="C:/Users/paulg/Documents/ENS/M2R/Courses/Projet/code/carac-geo/image3"

files=os.listdir(path)

#parameters measured on the calibration pictures

area_ref=10*10
width_ref=10
length_ref=10
perimeter_ref=(10+10)*2

#calibration parameter

back= cv2.imread('background.jpg')

pixelsPerA= None
pixelsPerW= None
pixelsPerL= None
pixelsPerP= None

col = 0
row = 1

method=1

for f in files:
    f2=f
    f='image3/'+f
    img = cv2.imread(f)
    if method==1: 
        contour=contour_extrac(img)
    if method==2:
        contour=Str_sim_in(back,img)
    A,W,L,P=carac_geo_extra(contour)

    if pixelsPerA is None:
        pixelsPerA=A/area_ref
        pixelsPerW=W/width_ref        
        pixelsPerL=L/length_ref
        pixelsPerP=P/perimeter_ref
    A=A/pixelsPerA
    W=W/pixelsPerW
    L=L/pixelsPerL
    P=P/pixelsPerP
    carac_geo=[f2,A,P,W,L]
    for c in (carac_geo):
        worksheet.write(row, col,c)
        col+=1
    row+=1
    col=0



workbook.close()