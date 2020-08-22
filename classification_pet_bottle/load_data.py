# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 16:44:48 2020

@author: paulg
"""

from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
import random
import tensorflow as tf
import pandas as pd
#Import wandb libraries

from sklearn.utils import class_weight
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import matplotlib
import matplotlib.image as mpimg

hyperparameter_defaults = dict(
  dropout = 0.2,
  hidden_layer_size = 12,
  learn_rate = 0.01,
  decay = 1e-6,
  momentum = 0.9,
  epochs = 300,
)


CSV_COLUMN_NAMES = ['Volume', 'Area', 'Length', 'Width']
dataset = pd.read_csv("echantillonage2.csv",names=CSV_COLUMN_NAMES, header=0)
dataset = dataset.values
X = dataset[:,1:4].astype(float)
max_values=np.max(X,axis=0)
min_values=np.min(X,axis=0)

train=pd.read_csv("echantillonage_train.csv",names=CSV_COLUMN_NAMES, header=0)
validate=pd.read_csv("echantillonage_validate.csv",names=CSV_COLUMN_NAMES, header=0)
#test=pd.read_csv("echantillonage_test.csv",names=CSV_COLUMN_NAMES, header=0)



test=pd.read_csv("echantillonage_test_2.csv",names=CSV_COLUMN_NAMES, header=0)

train=train.values
validate=validate.values
test=test.values

x_train=train[:,1:4].astype(float)
y_train = train[:,0]*2-1


x_validate=validate[:,1:4].astype(float)
y_validate = validate[:,0]*2-1

x_test=test[:,1:4].astype(float)
y_test = test[:,0]*2-1
y_test1=y_test

x_train=np.multiply(1/(max_values-min_values),x_train-min_values)

x_validate=np.multiply(1/(max_values-min_values),x_validate-min_values)

x_test=np.multiply(1/(max_values-min_values),x_test-min_values)

y_train= to_categorical(y_train)
y_validate= to_categorical(y_validate)
y_test= to_categorical(y_test)

model = keras.models.load_model('ep_344_hdl_5')
#model = keras.models.load_model('ep_300_hdl_12_improved')

class_predi=model.predict_classes(x_test)
proba=model.predict(x_test)




# proba_m=np.max(proba,axis=1)

# path='C:/Users/paulg/Documents/ENS/M2R/Courses/Projet/images/illustration/volume_mass/bottles/test'
# filess=os.listdir(path)

# plt.figure(figsize=(40,30))
# fig=plt.figure()
# fig.set_size_inches(40,37.5)

# classes_names=['0.5L','1L','1.5L','2L']
# for i in range(13):
#     real_n= classes_names[int(y_test1[i])]
#     predict_names=classes_names[int(class_predi[i])]
#     prob=proba_m[i]
#     plt.subplot(3,5,i+1)
#     img = mpimg.imread(path+'/'+filess[i])
#     plt.imshow(img)
#     plt.axis('off')   
#     plt.title("P: "+predict_names+" "+"A: "+real_n+", "+"%.2f" %(prob*100)+"%", fontsize=30)
# img = mpimg.imread(path+'/'+filess[12])
# plt.imshow(img)
# plt.axis('off')
# fig=plt.figure()  
# fig.set_size_inches(320,300)
# plt.axis('off')
# plt.imshow(img)
# img2=cv2.imread(path+'/'+filess[12])
# cv2.imshow('filename',img2)
# cv2.waitKey()

# f, axarr = plt.subplots(3, 5)
# plt.figure(figsize=(20,15))
# classes_names=['0.5L','1L','1.5L','2L']
# for i in range(13):
#     real_n= classes_names[int(y_test1[i])]
#     predict_names=classes_names[int(class_predi[i])]
#     prob=proba_m[i]
#     #plt.subplot(3,5,i+1)
#     img = mpimg.imread(path+'/'+filess[i])
#     axarr[i//5,i%5].imshow(img)
#     axarr[i//5,i%5].axis('off')
#     axarr[i//5,i%5].set_title("P: "+predict_names+" "+"A: "+real_n+", "+"%.2f" %(prob*100)+"%")


