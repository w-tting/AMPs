import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv1D, ZeroPadding1D, MaxPooling1D, Dropout, Flatten, Dense, LSTM


def cnn_model():
    model = keras.models.Sequential()
    model.add(ZeroPadding1D(5, input_shape=(MAX_SEQUENCE_LENGTH, character_to_index)))
    model.add(Conv1D(64, kernel_size=5, strides=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Dropout(0.5))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model
