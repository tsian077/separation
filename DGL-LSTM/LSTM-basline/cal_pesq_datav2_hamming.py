import librosa
from pesq import pesq
import numpy as np
import glob
from pystoi import stoi
import librosa.core as core
import math

def extract_overlapped_windows(x,nperseg,noverlap,window=None):
    step = nperseg - noverlap
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step, nperseg)
    strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    if window is not None:
        result = window * result
    return result

def cal_snr(origin, noised):
    snr = np.linalg.norm(origin) / np.linalg.norm(noised - origin)
    snr_db = 20 * np.log10(snr)
    return snr_db

def numpy_LSD(origianl_waveform, target_waveform):
    original_spectrogram = librosa.core.stft(origianl_waveform, n_fft=1024)
    target_spectrogram = librosa.core.stft(target_waveform, n_fft=1024)

    original_log = np.log10(np.abs(original_spectrogram) ** 2)
    target_log = np.log10(np.abs(target_spectrogram) ** 2)
    original_target_squared = (original_log - target_log) ** 2
    target_lsd = np.mean(np.sqrt(np.mean(original_target_squared, axis=0)))
    return target_lsd

def SNRseg(clean_speech, processed_speech,fs, frameLen=0.03, overlap=0.75):
    eps=np.finfo(np.float64).eps

    winlength   = round(frameLen*fs)
    skiprate    = int(np.floor((1-overlap)*frameLen*fs))
    MIN_SNR     = -10
    MAX_SNR     =  35

    hannWin=0.5*(1-np.cos(2*np.pi*np.arange(1,winlength+1)/(winlength+1)))
    clean_speech_framed=extract_overlapped_windows(clean_speech,winlength,winlength-skiprate,hannWin)
    processed_speech_framed=extract_overlapped_windows(processed_speech,winlength,winlength-skiprate,hannWin)
    
    signal_energy = np.power(clean_speech_framed,2).sum(-1)
    noise_energy = np.power(clean_speech_framed-processed_speech_framed,2).sum(-1)
    
    segmental_snr = 10*np.log10(signal_energy/(noise_energy+eps)+eps)
    segmental_snr[segmental_snr<MIN_SNR]=MIN_SNR
    segmental_snr[segmental_snr>MAX_SNR]=MAX_SNR
    segmental_snr=segmental_snr[:-1]
    return np.mean(segmental_snr)

noised = glob.glob('LSTM_0db_hamming_v2_mse_0_0001_batch_100/*.wav')
origin = glob.glob('../testOrigin_0dB/*.wav')


pesq_lst_wb = []
pesq_lst_nb = []
snr_lst = []
stoi_lst= []
lsd_lst = []
ssnr_lst = []

for i in range(len(noised)):

    ref, rate = librosa.load(origin[i], sr=16000)
    print(ref.shape)
    # to 512
    ref = ref[:(ref.shape[0] // 512) * 512].reshape(-1, 512)
    print(ref.shape)
    ref = ref[2:-2]
    print(ref.shape)
    ref = ref.reshape(-1)
    print(ref.shape)

    deg, rate = librosa.load(noised[i], sr=16000)
    print(deg.shape)

    wb = pesq(rate, ref, deg, 'wb')
    nb = pesq(rate, ref, deg, 'nb')

    snr = cal_snr(ref, deg)

    st = stoi(ref, deg, 16000, extended=False)
    
    lsd = numpy_LSD(ref, deg)
    ssnr = SNRseg(ref, deg, 16000)

    pesq_lst_wb.append(wb)
    pesq_lst_nb.append(nb)
    snr_lst.append(snr)
    stoi_lst.append(st)
    lsd_lst.append(lsd)
    ssnr_lst.append(ssnr)

    print(str(i) + '/' + str(len(noised)), 'wb:', wb, 'nb:', nb, 'snr:', snr, 'stoi:', st, 'lsd:', lsd, 'ssnr:', ssnr)

pesq_lst_wb = np.array(pesq_lst_wb)
pesq_lst_nb = np.array(pesq_lst_nb)
snr_lst = np.array(snr_lst)
stoi_lst = np.array(stoi_lst)
lsd_lst = np.array(lsd_lst)
ssnr_lst = np.array(ssnr_lst)

print('pesq_wb', np.mean(pesq_lst_wb))
print('pesq_nb', np.mean(pesq_lst_nb))
print('snr_lst', np.mean(snr_lst))
print('stoi_lst', np.mean(stoi_lst))
print('lsd_lst', np.mean(lsd_lst))
print('ssnr_lst', np.mean(ssnr_lst))
