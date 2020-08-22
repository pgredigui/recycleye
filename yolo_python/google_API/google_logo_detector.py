# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 09:59:57 2020

@author: paulg
"""


import os, io
from google.cloud import vision
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import time
import cv2




os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'ServiceAccountToken.json'
client = vision.ImageAnnotatorClient()

file_name = 'test-4.jpg'
image_folder = 'C:/Users/paulg/Documents/ENS/M2R/Courses/Projet/code/YOLO/test_set_raw'
folder=os.listdir(image_folder)

t0= time.clock()
n=len(folder)
for i in range(39,n):
    file_name=folder[i]
    image_path = os.path.join(image_folder, file_name)
    img=cv2.imread(image_path)
    
    
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.types.Image(content=content)
    response = client.logo_detection(image=image)
    logos = response.logo_annotations
    for logo in logos:
        logo_name=logo.description
        logo_name_low=logo_name.lower()
        if 'heineken' in logo_name_low:
            color=[0,255,0]
            name='Heineken'
        elif 'coca' in logo_name_low:
            color=[0,0,255]
            name='Coca'
        elif 'pepsi' in logo_name_low:
            color=[255,0,0]
            name='Pepsi'
        else:
            color=[255,255,0]
            name=logo_name
        name=name+' '+ "%.2f" %(logo.score*100)+'%'
        (text_width, text_height) = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)[0]
        
        
    
        vertices = logo.bounding_poly.vertices
        scale=(vertices[2].x-vertices[0].x)/text_width
    
        cv2.rectangle(img, (vertices[0].x,vertices[0].y),(vertices[2].x,vertices[2].y),(color[0],color[1],color[2]),2)
    
        cv2.rectangle(img, (vertices[0].x,vertices[0].y), (vertices[0].x+text_width,vertices[0].y-12-text_height), color, cv2.FILLED)
        cv2.putText(img,name,(vertices[0].x,vertices[0].y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    cv2.imwrite('images/result'+str(i)+'.jpg',img)
    print('Compteur', i)
    
t1 = time.clock() - t0
print("Time elapsed: ", t1)

# import os, io
# from google.cloud import vision
# import pandas as pd
# from PIL import Image, ImageDraw, ImageFont
# import time
# import cv2



# t0= time.clock()
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'ServiceAccountToken.json'
# client = vision.ImageAnnotatorClient()

# file_name = 'test-9.jpg'
# image_folder = 'C:/Users/paulg/Documents/ENS/M2R/Courses/Projet/code/YOLO/test_set_raw'
# image_path = os.path.join(image_folder, file_name)
# img=cv2.imread(image_path)


# with io.open(image_path, 'rb') as image_file:
#     content = image_file.read()

# image = vision.types.Image(content=content)
# response = client.logo_detection(image=image)
# logos = response.logo_annotations
# for logo in logos:
#     logo_name=logo.description
#     logo_name_low=logo_name.lower()
#     if 'heineken' in logo_name_low:
#         color=[0,255,0]
#         name='Heineken'
#     elif 'coca' in logo_name_low:
#         color=[0,0,255]
#         name='Coca'
#     elif 'pepsi' in logo_name_low:
#         color=[255,0,0]
#         name='Pepsi'
#     else:
#         color=[255,255,0]
#         name=logo_name
    
    
#     name=name+' '+ "%.2f" %(logo.score*100)+'%'
#     (text_width, text_height) = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)[0]
#     vertices = logo.bounding_poly.vertices
#     scale=(vertices[2].x-vertices[0].x)/text_width

#     cv2.rectangle(img, (vertices[0].x,vertices[0].y),(vertices[2].x,vertices[2].y),(color[0],color[1],color[2]),2)

#     cv2.rectangle(img, (vertices[0].x,vertices[0].y), (vertices[0].x+text_width,vertices[0].y-12-text_height), color, cv2.FILLED)
#     cv2.putText(img,name,(vertices[0].x,vertices[0].y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)

# cv2.imshow('result',img)
# cv2.waitKey()
    
# t1 = time.clock() - t0
# print("Time elapsed: ", t1)