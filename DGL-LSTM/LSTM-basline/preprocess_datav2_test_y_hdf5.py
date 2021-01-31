import os
import h5py
import librosa
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

dir_name = '../testOrigin'

files = os.listdir(dir_name)
outfile = "test_y_hamming_ln.hdf5"

window = signal.hamming(512)

hf = h5py.File(outfile, 'w')
frame = np.zeros((0, 512))
dset = hf.create_dataset('test_y', data=frame, maxshape=(None, 512), chunks=True)
for index, f in enumerate(files):
    y, sr = librosa.load(dir_name + '/' + f, sr=16000)
    temp = y[:(y.shape[0] // 512) * 512].reshape(-1)

    tt = np.array([])
    for j in range(256, len(temp)-256, 256):
        tt = np.append(tt, window * temp[j: j+512])

    tt = tt.reshape(-1, 512)
    tt = tt[2:-2]
    print('tt:', tt.shape)

    temp = np.abs(np.fft.fft(tt))
    temp = temp + 1
    temp = np.log(temp)
    print(temp.shape)
    hf['test_y'].resize((hf['test_y'].shape[0] + temp.shape[0]), axis=0)
    hf['test_y'][-1 * temp.shape[0]:] = temp
    print(str(index) + '/' + str(len(files)))

hf.close()