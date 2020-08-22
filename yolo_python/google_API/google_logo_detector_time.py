# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 11:11:34 2020

@author: paulg
"""

import time
import cv2
import os, io
from google.cloud import vision
import pandas as pd
from PIL import Image, ImageDraw, ImageFont



os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'ServiceAccountToken.json'
client = vision.ImageAnnotatorClient()

image_folder = 'C:/Users/paulg/Documents/ENS/M2R/Courses/Projet/code/YOLO/test_set_raw'
folder=os.listdir(image_folder)

t0= time.clock()
n=len(folder)
for i in range(n):
    file_name=folder[i]
    image_path = os.path.join(image_folder, file_name)
    img=cv2.imread(image_path)
    
    
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.types.Image(content=content)
    response = client.logo_detection(image=image)
    logos = response.logo_annotations
    print('Compteur:', i)
    t1 = time.clock() - t0
    print('time:',t1)
    
t1 = time.clock() - t0
print("Time elapsed: ", t1)