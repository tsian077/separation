import os
import h5py
import librosa
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

dir_name = '../train0dB'

files = os.listdir(dir_name)
outfile = "train_x_hamming_ln.hdf5"

window = signal.hamming(512)

hf = h5py.File(outfile, 'w')
frame = np.zeros((0, 5, 512))
dset = hf.create_dataset('train_x', data=frame, maxshape=(None, 5, 512), chunks=True)
for index, f in enumerate(files):
    y, sr = librosa.load(dir_name + '/' + f, sr=16000)
    temp = y[:(y.shape[0] // 512) * 512].reshape(-1)

    tt = np.array([])
    for j in range(256, len(temp)-256, 256):
        tt = np.append(tt, window * temp[j: j+512])
    tt = tt.reshape(-1, 512)
    print('tt:', tt.shape)

    ttt = np.zeros((tt.shape[0] - 4, 5, 512))
    cnt = 0
    for i in range(2, tt.shape[0] - 2):
        ttt[cnt] = tt[i-2:i+3]
        cnt = cnt + 1
    temp = np.abs(np.fft.fft(ttt))
    temp = temp + 1
    temp = np.log(temp)
    print(temp.shape)
    hf['train_x'].resize((hf['train_x'].shape[0] + temp.shape[0]), axis=0)
    hf['train_x'][-1 * temp.shape[0]:] = temp
    print(str(index) + '/' + str(len(files)))

hf.close()
