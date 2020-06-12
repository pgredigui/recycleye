# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:52:23 2020

@author: paulg
"""

# import the necessary packages
from imutils import paths
import numpy as np
import imutils
import cv2


def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)
	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key = cv2.contourArea)
	# compute the bounding box of the of the paper region and return it
	return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

# initialize the known distance from the camera to the object, which
# in this case is 20 cm
KNOWN_DISTANCE = 20.0
# initialize the known object width, which in this case, the piece of
# paper is 10 cm wide
KNOWN_WIDTH = 10.0
# load the furst image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length
image = cv2.imread("images/20.jpg")
marker = find_marker(image)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH


#verify the distance size relationship
    
for imagePath in sorted(paths.list_images("images")):
    print(imagePath)
    image=cv2.imread(imagePath)
    marker=find_marker(image)
    inches=distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
    box=cv2.cv.Boxpoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
    box=np.int0(box)
    cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
    cv2.putText(image, "%.2fcm" % (inches ),(int(image.shape[1]/2), int(image.shape[0]/2)), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 1)
    cv2.imshow("image", image)
    cv2.waitKey(0)

    
