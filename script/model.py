import numpy as np
import pandas as pd
from utils import MAX_SEQUENCE_LENGTH, num_chars, split_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, ZeroPadding1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def cnn_model(filters=64, kernel_size=5, dropout_rate=0.5, dense_units=100, learning_rate=0.001):
    model = keras.models.Sequential()
    model.add(ZeroPadding1D(5, input_shape=(MAX_SEQUENCE_LENGTH, num_chars)))
    model.add(Conv1D(filters, kernel_size=kernel_size, strides=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Dropout(dropout_rate))
    model.add(Conv1D(filters, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model

if __name__ == '__main__':
    data_path='../data/ecoli.csv'
    model_path='../model/Base.h5'
    X_train, x_test, y_train, y_test=split_data(data_path)
    model=cnn_model()
    model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.1, verbose=0)
    model.save(model_path)