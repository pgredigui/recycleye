# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 09:11:30 2020

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
import wandb
from wandb.keras import WandbCallback
from sklearn.utils import class_weight
import numpy as np
from sklearn.model_selection import train_test_split
import os 

hyperparameter_defaults = dict(
  dropout = 0.2,
  hidden_layer_size = 8,
  learn_rate = 0.01,
  decay = 1e-6,
  momentum = 0.9,
  epochs = 300,
)

wandb.init(project="bootle-classification")

wandb.init(config=hyperparameter_defaults)
config = wandb.config

CSV_COLUMN_NAMES = ['Volume', 'Area', 'Length', 'Width']

train = pd.read_csv("echantillonage2.csv",names=CSV_COLUMN_NAMES, header=0)
dataset = train.values
X_train = dataset[:,1:4].astype(float)
X_train= np.multiply(1/(np.max(X_train,axis=0)-np.min(X_train,axis=0)),X_train-np.min(X_train,axis=0))
Y_train = dataset[:,0]*2-1


x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train, test_size=0.2, shuffle= True)




class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)

model = Sequential()
model.add(Dense(config.hidden_layer_size, input_dim=3, activation='relu'))
model.add(Dense(4, activation='softmax'))


sgd = SGD(lr=config.learn_rate, decay=config.decay, momentum=config.momentum,
                            nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(x_train, y_train,  validation_data=(x_valid, y_valid),shuffle=True, epochs=config.epochs,class_weight =class_weights,
    callbacks=[WandbCallback()])
