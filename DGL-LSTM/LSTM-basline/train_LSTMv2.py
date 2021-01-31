import json
import numpy as np
import keras
import h5py
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers import LeakyReLU
from keras.layers import Bidirectional
from tensorflow.keras.utils import HDF5Matrix
epochs = 80

x_train = HDF5Matrix('train_x_hamming_ln.hdf5', 'train_x')
y_train = HDF5Matrix('train_y_hamming_ln.hdf5', 'train_y')
x_test = HDF5Matrix('test_x_hamming_ln.hdf5', 'test_x')
y_test = HDF5Matrix('test_y_hamming_ln.hdf5', 'test_y')


model = Sequential()
model.add(LSTM(1024, input_shape=(5, 512,), return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(1024, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(512, activation='relu'))


opt = keras.optimizers.Adam(lr=0.0001)
model.compile(optimizer=opt, loss='MSE')

class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 10 == 0:
            self.model.save("LSTM_0db_hamming_v2_mse_0_0001_batch_100_{}.hd5".format(epoch))

saver = CustomSaver()
history = model.fit(x_train, y_train, epochs=epochs, callbacks=[saver],
                    batch_size=100, validation_data=(x_test, y_test), shuffle='batch')


model.save('LSTM_0db_hamming_v2_mse_0_0001_batch_100.h5')
json.dump(history.history, open('LSTM_0db_hamming_v2_mse_0_0001_batch_100.json', 'w'))
