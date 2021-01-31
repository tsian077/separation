import os
import librosa
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def cal_snr(origin, noised):
    snr = np.linalg.norm(origin) / np.linalg.norm(noised - origin)
    snr_db = 20 * np.log10(snr)
    return snr_db


def normalize(signal):
    max_number = np.max(signal)
    min_number = np.abs(np.min(signal))

    if max_number > min_number:
        signal = signal / max_number
    else:
        signal = signal / min_number

    return signal


dir_name = '../test0dB'

files = os.listdir(dir_name)

output_dir_name = 'LSTM_0db_hamming_v2_mse_0_0001_batch_100'

os.makedirs(output_dir_name, exist_ok=True)

window = signal.hamming(512)


data_raw = np.load('LSTM_0db_hamming_v2_mse_0_0001_batch_100.npy').reshape(-1, 512)
start = 0
snr_lst = []
for index, f in enumerate(files):
    y, sr = librosa.load(dir_name + '/' + f, sr=16000)
    temp = y[:(y.shape[0] // 512) * 512].reshape(-1)

    tt = np.array([])
    for j in range(256, len(temp)-256, 256):
        tt = np.append(tt, window * temp[j: j+512])
    tt = tt.reshape(-1, 512)
    print('tt:', tt.shape)


    tt = tt.reshape(-1, 512)
    tt = tt[2:-2]
    print('tt:', tt.shape)

    angle = np.angle(np.fft.fft(tt))

    size = tt.shape[0]
    print(start, start + size)

    abs_data = data_raw[start: start + size]
    print('abs_data:', abs_data.shape)
    print('angle:', angle.shape)

    abs_data = np.exp(abs_data)
    abs_data = abs_data - 1

    data = abs_data * (np.cos(angle) + np.sin(angle) * 1j)
    data = np.fft.ifft(data)
    data = np.real(data)
    print('err:', np.sum(tt-data))
    print('snr:', cal_snr(tt, data))

    ttt = np.array([])
    for i in range(1, data.shape[0] - 1, 2):
        t = data[i]

        left = data[i - 1]
        left = left[256:]
        left = np.append(left, np.zeros(256))

        right = data[i + 1]
        right = right[:256]
        right = np.append(np.zeros(256), right)

        t = t + left + right

        ttt = np.append(ttt, t)

    ttt = ttt.reshape(-1, 512) / 1.08
    print(ttt.shape)
    print('err:', np.sum(temp.reshape(-1, 512)[2:-2]-ttt))
    snr = cal_snr(temp.reshape(-1, 512)[2:-2], ttt)
    print('snr:', snr)
    snr_lst.append(snr)

    ttt = ttt.reshape(-1)
    ttt = np.asfortranarray(ttt)

    # data = normalize(data)
    print(f)

    librosa.output.write_wav(output_dir_name + '/' + f, ttt, sr=16000)

    start = start + size
    print(str(index) + '/' + str(len(files)))

snr = np.array(snr)
print(np.mean(snr))
