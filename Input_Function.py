# -- coding:utf-8 --
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import losses, optimizers

### Define models
def define_model(input_size=914):
    x_input = tf.keras.Input(shape=(input_size, ), name='input_data')
    
    x = tf.keras.layers.Dense(units=512, kernel_regularizer = tf.keras.regularizers.l2(0.01))(x_input)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)
    
    x = tf.keras.layers.Dense(units=256, kernel_regularizer = tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)
    
    x = tf.keras.layers.Dense(units=128, kernel_regularizer = tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)
    
    
    x = tf.keras.layers.Dense(units=64, kernel_regularizer = tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)
    
    x = tf.keras.layers.Dense(units=32,kernel_regularizer = tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)
    
    x = tf.keras.layers.Dense(units=16,kernel_regularizer = tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)
    
    x = tf.keras.layers.Dense(units=8,kernel_regularizer = tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)
    

    
    x_out = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)

    # === Build Model
    model = Model(inputs=x_input, outputs=x_out)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # ###=== Model summary
    # model.summary()
    return model

def eval_model_DNN(input_data, model, SavePath):
    for num_folds in range(1,6):
        tf.keras.backend.clear_session()
        model.load_weights(SavePath + 'model_best_fold' + str(num_folds) + '.h5')
        globals()["test_prob{}".format(num_folds)] = model.predict(input_data)
        tf.keras.backend.clear_session()
    
    test_prob = [(test_prob1[i] + test_prob2[i] + test_prob3[i] + test_prob4[i] + test_prob5[i])/5 for i in range(len(test_prob1))]
    test_prob = test_prob[0][0]

    return test_prob