# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 12:14:25 2020

@author: paulg
"""

"""
Created on Fri Jun 12 16:23:42 2020

@author: paulg
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import imutils
from skimage.measure import compare_ssim
#####################################################
##              Volume measurement                 ##
#####################################################

# THIS ALGORITHM MEASURE THE GEMETRIC PROPETIES OF A SHAPE:AREA, DEPTH, VOLUME, BOUNDING BOX

# method=0 contour extraction with variable erosion and dilatation
# method=1 contour extraction with a filter depth



############# Functions #############


def Str_sim_in(background,image,gau_blur):
    
    grayA = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gau_blur==1:
        grayA= cv2.GaussianBlur(grayA, (3, 3), 0)
        grayB= cv2.GaussianBlur(grayB, (3, 3), 0)
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
    cv2.imshow('diff', thresh)
    return cnts

def empty(a):
    #dunction for trackbars
    pass

def contour_extrac2(image,dila,ero):
    # contour extraction for the filter depth method
    
    #conversion to gray scale a
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(7,7),0)
    #thresholding is used here because the depth filter convert the background in white
    
    thresh= cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    
    # edged detection because the black and white suitable forthe detection (the object is in black). 
    #Converts every pixel to the opposite binary value could also work
    
    edged = cv2.Canny(thresh, 50, 100)
    edged = cv2.dilate(edged, None, iterations=dila)
    edged = cv2.erode(edged, None, iterations=ero)
    cnts = cv2.findContours( edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    cnts = imutils.grab_contours(cnts)
    
    return cnts


def contour_extrac(image,dila,ero):
    # contour extraction when the depth mask is not used 
    # more accurate erode=1 and dila=1
    # for some objects with issue backgourn erode=3 and dila=6 or five is better but slighly to high area 
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(7,7),0)
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=dila)
    edged = cv2.erode(edged, None, iterations=ero)
    
    
    cnts = cv2.findContours( edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    cnts = imutils.grab_contours(cnts)    

    return cnts


def geo_property(depth_image,cnt,color_image,dist_came_back,length_per_pixel,area_per_pixel):
    area=cv2.contourArea(cnt)
    if area>500:
        cv2.drawContours(color_image,[cnt],-1,(0,255,0),2)
        #(x, y, w, h) = cv2.boundingRect(cnt)
        rect = cv2.minAreaRect(cnt)
        pos=rect[0]
        size=rect[1]
        x=int(pos[0])
        y=int(pos[1])
        w=min(size)
        h=max(size)
        cimg = np.zeros_like(depth_image)
        cv2.drawContours(cimg, [cnt],-1, color=255, thickness=-1)
        

        # Access the image pixels and create a 1D numpy array then add to list
        #the condition is extremely important 
        pts = np.where((cimg == 255)*(depth_image!=0))
        
        depth=dist_came_back-np.mean(depth_image[pts[0], pts[1]])
        
        correct_factor=(dist_came_back-depth)/dist_came_back
        
        w_real=w*length_per_pixel*correct_factor
        h_real=h*length_per_pixel*correct_factor
        area_real=area*area_per_pixel*(correct_factor**2)
        print(area_real,h_real,w_real,depth*100)             
        
        cv2.putText(color_image, 'area='+ "%.3fcm^2" %(area_real), (x-100, y-100), 1, 1.2, (0, 0, 255))
        cv2.putText(color_image,"depth="+ "%.3fcm" %(depth*100), (x-100, y-70), 1, 1.2, (0, 0, 255))
    return color_image

def geo_property2(depth_image,cnt,color_image,depth_image_ini,length_per_pixel,area_per_pixel):
    area=cv2.contourArea(cnt)
    if area>500:
        cv2.drawContours(color_image,[cnt],-1,(0,255,0),1)
        #(x, y, w, h) = cv2.boundingRect(cnt)
        rect = cv2.minAreaRect(cnt)
        pos=rect[0]
        size=rect[1]
        x=int(pos[0])
        y=int(pos[1])
        w=min(size)
        h=max(size)
        cimg = np.zeros_like(depth_image)
        cv2.drawContours(cimg, [cnt],-1, color=255, thickness=-1)
        

        # Access the image pixels and create a 1D numpy array then add to list
        #the condition is extremely important 
        pts = np.where((cimg == 255)*(depth_image!=0))
        
        dist_came_back=np.mean(depth_image_ini[pts[0], pts[1]])
        depth=dist_came_back-np.mean(depth_image[pts[0], pts[1]])
        correct_factor=(dist_came_back-depth)/dist_came_back
        
        w_real=w*length_per_pixel*correct_factor
        h_real=h*length_per_pixel*correct_factor
        area_real=area*area_per_pixel*(correct_factor**2)
        print(area_real,h_real,w_real)             
        
        cv2.putText(color_image, 'area='+str(area_real), (x, y), 1, 1, (0, 255, 0))
        #cv2.putText(color_image,"depth="+ "%.3fcm" %(depth*100), (x-50, y+20), 1, 1, (0, 255, 0))
    return color_image

############# Functions #############
    


############# Calibration #############
    
#need to be adjusted depending on the setup 

# distance of the camera from the background
dist_came_back=0.86

# area and length of a pixel atthe background level. Try to do it as precosely as possible

# studio 1
# area_per_pixel=100/12640
# length_per_pixel=10/113.7

#error
#area_per_pixel=100/13000
#length_per_pixel=10/115

#studio2 correct
# area_per_pixel=100/4950
# length_per_pixel=10/71.5

area_per_pixel=100/5190.8
length_per_pixel=10/72.7

# filter on struc_similarity index 1=yes other=nont

#if you want to work on the full window chose x1=None 
x1,x2,y1,y2=120,420,100,390
filt_stu=0
############# Calibration #############


############# Camera parameter #############

# Create a pipeline
pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 10)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 10)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)
init_bg0=None
init_bg1=None
init_bg2=None
# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away


# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

############# Camera parameter #############

############# Trackbar creation #############


cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars",640,240)

#method 0 without backgroud extraction method 1 with background extraction

cv2.createTrackbar("Method","Trackbars",1,2,empty)
cv2.createTrackbar("Dilate","Trackbars",1,15,empty)
cv2.createTrackbar("Erode","Trackbars",1,15,empty)
cv2.createTrackbar("Max_dist","Trackbars",530,1000,empty)
cv2.createTrackbar("Min_dist","Trackbars",10,1000,empty)
############# Trackbar creation #############



############# Real time video analysis #############

num_im=70

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        colorizer = rs.colorizer()
        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        

        # Convert images to numpy arrays
        
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        if x1!=None:
            depth_image = depth_image[x1:x2,y1:y2]
            color_image = color_image[x1:x2,y1:y2]
            colorized_depth=colorized_depth[x1:x2,y1:y2]
        

        
        #choose the method
        method=cv2.getTrackbarPos("Method","Trackbars")
        if method==0:
            if init_bg0==None:
                depth_image_ini=depth_image*depth_scale 
                init_bg0=1
                    #scale the depth
            depth_image = depth_image * depth_scale   
            dilate=cv2.getTrackbarPos("Dilate","Trackbars")
            erode=cv2.getTrackbarPos("Erode","Trackbars")
            cnts=contour_extrac(color_image,dilate,erode)
            for cnt in cnts:
                #color_image=geo_property2(depth_image,cnt,color_image,depth_image_ini,length_per_pixel,area_per_pixel)
                color_image=geo_property(depth_image,cnt,color_image,dist_came_back,length_per_pixel,area_per_pixel)
        if method==1:
            if init_bg1==None:
                
                depth_image_ini=depth_image*depth_scale
                init_bg1=1

            # Remove background - Set pixels further than clipping_distance to filt_color
            # max_clipping_distance=cv2.getTrackbarPos("Max_dist","Trackbars")/(10000*depth_scale)
            # min_clipping_distance=cv2.getTrackbarPos("Min_dist","Trackbars")/(10000*depth_scale)
            max_clipping_distance=cv2.getTrackbarPos("Max_dist","Trackbars")/(1000)
            min_clipping_distance=cv2.getTrackbarPos("Min_dist","Trackbars")/(1000)
            filt_color = 255
            depth_image=depth_image*depth_scale
            depth_image_3d = np.dstack((depth_image,depth_image,depth_image))#depth image is 1 channel, color is 3 channels
            dist_1D= abs(depth_image_ini-depth_image)
            dist_3D=np.dstack((dist_1D,dist_1D,dist_1D))
            bg_removed = np.where((depth_image_3d!=0)*((dist_3D > max_clipping_distance) | (dist_3D< min_clipping_distance)), filt_color, color_image)
             
            #the depth_image_3d!=0 is to remove the point equal to zero
  
            # bg_removed = np.where((depth_image_3d > max_clipping_distance) | (depth_image_3d < min_clipping_distance), filt_color, color_image)
            dilate=cv2.getTrackbarPos("Dilate","Trackbars")
            erode=cv2.getTrackbarPos("Erode","Trackbars")            
            cnts=contour_extrac2(bg_removed,dilate,erode)
                    #scale the depth
            for cnt in cnts:
                color_image=geo_property(depth_image,cnt,color_image,dist_came_back,length_per_pixel,area_per_pixel)
            cv2.imshow('bg removed', bg_removed)
            
        if method==2:
            if init_bg2==None:
                first_frame=color_image
                depth_image_ini=depth_image*depth_scale 
                init_bg2=1
                 #scale the depth
            cnts=Str_sim_in(first_frame,color_image,filt_stu)
            depth_image = depth_image * depth_scale   
            test=depth_image 
            for cnt in cnts:
                #color_image=geo_property2(depth_image,cnt,color_image,depth_image_ini,length_per_pixel,area_per_pixel)            
                color_image=geo_property(depth_image,cnt,color_image,dist_came_back,length_per_pixel,area_per_pixel)
        # Show images
        images = np.hstack((color_image, colorized_depth))
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', images)
        
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window

        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        if key & 0xFF == ord('s'):
            cv2.imwrite('images5/bootle_cnt_color'+str(num_im)+'.jpg', color_image)
            cv2.imwrite('images5/bootle_depth'+str(num_im)+'.jpg', colorized_depth)
            num_im+=1
finally:
    pipeline.stop()

############# Real time video analysis #############