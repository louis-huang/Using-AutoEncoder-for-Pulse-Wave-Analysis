#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 12:31:35 2018

@author: louis
"""

from keras.models import Model
from keras.layers import Dense, Input
from keras import optimizers
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import peak_loss

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

data = pd.read_csv('good_data_starts_valley.csv')
from sklearn.model_selection import train_test_split
random_state = 11
training_set, test_set = train_test_split(data, test_size = 0.2, random_state = random_state)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler((0,1))
training_set = sc.fit_transform(training_set.T).T

encoding_dim = 50
input_wave = Input(shape = (1500,))

# encoder layers
encoded = Dense(200, activation='elu')(input_wave)
encoded = Dense(100, activation='elu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)

# decoder layers
decoded = Dense(100, activation='elu')(encoder_output)
decoded = Dense(200, activation='elu')(decoded)
decoded = Dense(1500, activation='sigmoid')(decoded)

#construct autoencoder
autoencoder = Model(inputs=input_wave, outputs=decoded)

# compile autoencoder

optim = optimizers.Adam(lr = 0.001)
autoencoder.compile(optimizer= optim, loss='mse')

autoencoder.fit(training_set, training_set,
                epochs=10000,
                batch_size=256,
                shuffle=True)
#visualization

arrange = 240
plt.figure(figsize = (20,10))
plt.title('123')
c_id = 0
for p in range(1,9,1):
    arrange += 1
    c_plot = plt.subplot(arrange)
    c_input = training_set[c_id].reshape(1,-1)
    c_encoded = autoencoder.predict(c_input)
    c_plot.plot(c_input[0], color = 'yellow')
    c_plot.plot(c_encoded[0], color = 'red')
    c_id += 1

#scale x_test
sc2 = MinMaxScaler((0,1))
test_set = sc2.fit_transform(test_set.T).T
predicted_test = autoencoder.predict(test_set)
from sklearn.metrics import mean_squared_error
test_mse = mean_squared_error(test_set, predicted_test)

#visualize test data
arrange = 240
plt.figure(figsize = (20,10))
plt.title('123')
c_id = 0
for p in range(1,9,1):
    arrange += 1
    c_plot = plt.subplot(arrange)
    c_input = test_set[c_id].reshape(1,-1)
    c_encoded = autoencoder.predict(c_input)
    c_plot.plot(c_input[0], color = 'yellow')
    c_plot.plot(c_encoded[0], color = 'red')
    c_id += 1
    loss = peak_loss.cal_loss(c_encoded[0], c_input[0])
    print(loss)

autoencoder.save('auto_keras_elu.h5')
    
    