# -*- coding: utf-8 -*-
"""
Created on Wed May 27 18:57:01 2020

@author: paulg
"""

import numpy as np
import cv2


#function used to compute the volume of an object with 3 side view

# generation of a side view. here the test has been masde with a cube


contours = np.array( [ [50,50], [50,150], [150, 150], [150,50] ] )
img = np.zeros( (200,200) ) # create a single channel 200x200 pixel black image 
cv2.fillPoly(img, pts =[contours], color=(255,255,255))
cv2.imshow(" ", img)
cv2.waitKey()




# img2 = cv2.imread('result.jpg')

# cv2.imshow(" ", img2)
# cv2.waitKey()

# cnt=img2
# cnt=cnt[:,:,0]

#dimension of the bounding box in with the object is contained 


x=2
y=2
z=2


# number of points for the onte carlo integral. The more point the more accurate the integral will be 
N=10**3


        
def monte_carlo(N,x,y,z,top,side1,side2):
    V=x*y*z
    dimTop=np.shape(top)
    dimside1=np.shape(side1)
    dimside2=np.shape(side2)
    x=np.random.random_sample(N)
    y=np.random.random_sample(N)
    z=np.random.random_sample(N)
    count=0
    for i in range(N):
        if top[int((dimTop[0])*x[i]),int((dimTop[1]-1)*y[i])]==255 and side1[int((dimside1[0]-1)*x[i]),int((dimside1[1]-1)*z[i])]==255 and side2[int((dimside2[0]-1)*y[i]),int((dimside2[1]-1)*z[i])]==255:
            count+=1
    V=V*(count/N)
    return V

V= monte_carlo(N,x,y,z,img,img,img)