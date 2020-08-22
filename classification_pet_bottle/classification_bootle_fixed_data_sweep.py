# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 14:54:39 2020

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

hyperparameter_defaults = dict(
  dropout = 0.2,
  hidden_layer_size = 12,
  learn_rate = 0.01,
  decay = 1e-6,
  momentum = 0.9,
  epochs = 300,
)

wandb.init(project="bootle-classification-sweep")

wandb.init(config=hyperparameter_defaults)
config = wandb.config

CSV_COLUMN_NAMES = ['Volume', 'Area', 'Length', 'Width']
dataset = pd.read_csv("echantillonage2.csv",names=CSV_COLUMN_NAMES, header=0)
dataset = dataset.values
X = dataset[:,1:4].astype(float)
max_values=np.max(X,axis=0)
min_values=np.min(X,axis=0)

train=pd.read_csv("echantillonage_train.csv",names=CSV_COLUMN_NAMES, header=0)
validate=pd.read_csv("echantillonage_validate.csv",names=CSV_COLUMN_NAMES, header=0)
test=pd.read_csv("echantillonage_test.csv",names=CSV_COLUMN_NAMES, header=0)

train=train.values
validate=validate.values
test=test.values

x_train=train[:,1:4].astype(float)
y_train = train[:,0]*2-1


x_validate=validate[:,1:4].astype(float)
y_validate = validate[:,0]*2-1

x_test=test[:,1:4].astype(float)
y_test = test[:,0]*2-1


x_train=np.multiply(1/(max_values-min_values),x_train-min_values)

x_validate=np.multiply(1/(max_values-min_values),x_validate-min_values)

x_test=np.multiply(1/(max_values-min_values),x_test-min_values)

y_train= to_categorical(y_train)
y_validate= to_categorical(y_validate)
y_test= to_categorical(y_test)


model = Sequential()
model.add(Dense(config.hidden_layer_size, input_dim=3, activation='relu'))
model.add(Dense(4, activation='softmax'))


sgd = SGD(lr=config.learn_rate, decay=config.decay, momentum=config.momentum,
                            nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(x_train, y_train,  validation_data=(x_validate, y_validate),shuffle=True, epochs=config.epochs,
    callbacks=[WandbCallback()])



