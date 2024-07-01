# -- coding:utf-8 --
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import losses, optimizers

import scipy.io as sio
from numpy import average
from numpy import array

### Define models
def define_model(input_size, learning_rate):
    input_data = keras.layers.Input(shape=(input_size, ))

    leaky_relu = tf.nn.leaky_relu

    x = Dense(128, activation=leaky_relu)(input_data)
    x = Dropout(rate=0.3)(x)
    x = Dense(64, activation=leaky_relu)(x)
    x = Dropout(rate=0.3)(x)
    x = Dense(32, activation=leaky_relu)(x)
    x = Dropout(rate=0.3)(x)
    x = Dense(16, activation=leaky_relu)(x)
    x = Dropout(rate=0.3)(x)
    x = Dense(8, activation=leaky_relu)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_data, outputs=x)
    model.compile(optimizer=optimizers.Adam(lr=learning_rate), loss=losses.binary_crossentropy)

    # print('Model==============')
    # model.summary()
    return model

def eval_model_DNN(input_factor, input_data, model, SavePath):

    model.load_weights(SavePath + 'model_best.h5')
    pred_dnn = model.predict(input_data).squeeze()
    return pred_dnn