Requirements:
python >= 3.7 (because of dataclass)
MALTAB engine for Python (because of the PESQ, STOI calculation)
PyTorch >= 1.2 (because of the tensorboard support)
tensorboard
numpy
scipy
tqdm
librosa

1. add_noisev2.py
混合A1 A2 B1 B2的人聲至固定dB。
需設定32行的snr_db = 固定dB以及35行的dir_name看是要test還是train。
執行前需要先準備好A1_train、A1_test、A2_train、A2_test、B1_train、B1_test、B2_train、B2_test資料。
執行後會產生train0dB、trainOrigin、test0dB、testOrign

2. LSTM/preprocess_datav2_*.py產生更個.hdf5訓練資料

3. LSTM/train.py 訓練

4. LSTM/predict.py 預測

5. LSTM/reconstruct.py 合成回聲音

6. LSTM/cal_pesq_datav2_hamming.py 測試 snr pesq

7. LSTM/prepare_data.py 準備給 deepGLA 的資料
需修改第 14 行與 16 行看要產生什麼資料，產生出 test_x、test_y、train_x、train_y
其實就是調整前後大小而已

8. 將 train_x、test_x 分別搬到deep-griffinlim-iteration-X/data/TIMIT/TRAIN、TEST，
並執行deep-griffinlim-iteration-X/create.py TRAIN 和 TEST，產生deep-griffinlim-iteration-X/data/TRAIN 與 TEST

9. 將 train_y、test_y 分別搬到deep-griffinlim-iteration-Y/data/TIMIT/TRAIN、TEST，
並執行deep-griffinlim-iteration-Y/create.py TRAIN 和 TEST，產生deep-griffinlim-iteration-Y/data/TRAIN 與 TEST

10. 將 deep-griffinlim-iteration-X/data/TRAIN 與 TEST 搬至 data_forgl/X_TRAIN、data_forgl/X_TEST 
以及 deep-griffinlim-iteration-Y/data/TRAIN 與 TEST 搬至 data_forgl/Y_TRAIN、data_forgl/Y_TEST 
並執行 group_test.py、group_train.py 產生 NEW_TRAIN、NEW_TEST
group*.py 
# ['spec_noisy', 'spec_clean', 'mag_clean', 'path_speech', 'length'] 
# spec_noisy = X:spec_clean		// X的clean為髒的
# spec_clean = Y:spec_clean 		// Y的clean為乾淨的
# mag_clean = predict
可以自己合成回來聽聽看

11. 將剛剛產生的 NEW_TRAIN、NEW_TEST 搬移至 deep-griffinlim-iteration/data 的 TRAIN 與 TEST
並執行 deep-griffinlim-iteration/main.py --train 與 -—test 訓練及測試 deepGLA

12. 將 LSTM/LSTM_0db_hamming_v2_mse_0_0001_batch_100 內所有聲音檔搬至 deep-griffinlim-iteration-Val/data/TIMIT/TEST，
並執行 deep-griffinlim-iteration-Val/create.py TEST，產生 deep-griffinlim-iteration-Val/data/TEST

13. 將 deep-griffinlim-iteration-Val/data/TEST 搬至 data_forgl/X_VAL 並執行 data_forgl/group_val.py 產生 NEW_VAL

14. cp 整個訓練好的 deep-griffinlim-iteration 為新的 deep-griffinlim-iteration-predict ，
並把 data_forgl/NEW_VAL 搬至 deep-griffinlim-iteration-predict/data/TEST 
然後執行 deep-griffinlim-iteration-predict/main.py —test 預測產生出最後的聲音
