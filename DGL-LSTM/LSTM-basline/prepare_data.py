import librosa
from pesq import pesq
import numpy as np
import glob
from tqdm import tqdm

def cal_snr(origin, noised):
    snr = np.linalg.norm(origin) / np.linalg.norm(noised - origin)
    snr_db = 20 * np.log10(snr)
    return snr_db

import os

output_dir = 'train_x'
os.makedirs(output_dir, exist_ok=True)
origin = glob.glob('../train0dB/*.wav')

for i in tqdm(origin):
    ref, rate = librosa.load(i, sr=16000)

    # to 512
    ref = ref[:(ref.shape[0] // 512) * 512].reshape(-1, 512)
    ref = ref[2:-2]
    ref = ref.reshape(-1)

    librosa.output.write_wav(output_dir + '/' + i.split('/')[-1], ref, rate)
    
