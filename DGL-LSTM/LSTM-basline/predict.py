import librosa
import numpy as np
from keras.models import load_model
import keras
import keras.losses
import tensorflow as tf
import keras.backend as K
from keras.utils.io_utils import HDF5Matrix

source = HDF5Matrix('test_x_hamming_ln.hdf5', 'test_x')


model = load_model('LSTM_0db_hamming_v2_mse_0_0001_batch_100.h5')

data = model.predict(source)
print(data.shape)

np.save('LSTM_0db_hamming_v2_mse_0_0001_batch_100.npy', data)
