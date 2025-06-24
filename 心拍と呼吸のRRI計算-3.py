import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog

# GUIでファイル選択
def choose_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title='CSVファイルを選択してください',
        filetypes=[("CSV Files", "*.csv")]
    )
    return file_path

# 入力ファイル選択
input_csv = choose_file()
if not input_csv:
    raise ValueError("ファイルが選択されませんでした。")

# 同じフォルダに出力するパスを作成
output_csv = os.path.join(os.path.dirname(input_csv), 'rri_output.csv')

# CSVファイルからデータを読み込む
data = pd.read_csv(input_csv)

# 時間データ、ECG、BRの列を抽出
time_data = data['time'].values
ecg_data = data['ECG'].values
br_data = data['BR'].values

# サンプリング周波数の計算
time_diff = np.diff(time_data)
if len(time_diff) > 0:
    sampling_rate = 1 / np.mean(time_diff)
else:
    raise ValueError("時間データが不十分です。")

# ECGのRピークを検出してRRIを計算
def calculate_rri_ecg(ecg_data, time_data):
    peaks, _ = find_peaks(ecg_data, distance=sampling_rate/2, height=np.mean(ecg_data))
    rri = np.diff(time_data[peaks]) * 1000.0  # RRIをミリ秒に変換
    return rri, peaks

# 呼吸波形の前処理：ローパスフィルタでノイズ除去
def preprocess_br_signal(br_data, cutoff_freq=0.5):
    nyquist_freq = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(5, normal_cutoff, btype='low', analog=False)
    filtered_br = filtfilt(b, a, br_data)
    return filtered_br

# 呼吸データから谷を検出してRRIを計算
def calculate_rri_br(filtered_br_data, time_data):
    inverted_br = -1 * filtered_br_data  # 谷の検出のために反転
    peaks, _ = find_peaks(inverted_br, distance=sampling_rate/2)
    rri = np.diff(time_data[peaks]) * 1000.0  # RRIをミリ秒に変換
    return rri, peaks

# 計算
ecg_rri, ecg_peaks = calculate_rri_ecg(ecg_data, time_data)
filtered_br_data = preprocess_br_signal(br_data)
br_rri, br_peaks = calculate_rri_br(filtered_br_data, time_data)

# NaNでリストの長さを揃える
max_length = max(len(ecg_rri), len(br_rri))
ecg_rri_padded = np.pad(ecg_rri, (0, max_length - len(ecg_rri)), constant_values=np.nan)
br_rri_padded = np.pad(br_rri, (0, max_length - len(br_rri)), constant_values=np.nan)

# RRIデータをCSVファイルに保存
rri_df = pd.DataFrame({
        'ECG RRI (ms)': pd.Series(np.round(ecg_rri_padded)).astype('Int64'), 
        'BR RRI (ms)': pd.Series(np.round(br_rri_padded)).astype('Int64')
})
try:
    rri_df.to_csv(output_csv, index=False)
    print(f"CSVファイルを保存しました: {output_csv}")
except Exception as e:
    print("CSV保存時にエラーが発生しました：", e)

# グラフ表示
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(time_data, ecg_data)
plt.scatter(time_data[ecg_peaks], ecg_data[ecg_peaks], color='red', marker='x')
plt.title('ECG Signal with R Peaks')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(4, 1, 2)
plt.plot(time_data, br_data, label='Original BR Signal')
plt.plot(time_data, filtered_br_data, label='Filtered BR Signal', color='orange')
plt.scatter(time_data[br_peaks], filtered_br_data[br_peaks], color='green', marker='x', label='Detected Troughs')
plt.title('BR Signal with Detected Troughs')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(ecg_rri, label='ECG RRI', marker='o')
plt.title('ECG RRI')
plt.xlabel('Beat Number')
plt.ylabel('RRI (ms)')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(br_rri, label='BR RRI', marker='x')
plt.title('BR RRI')
plt.xlabel('Trough Number')
plt.ylabel('RRI (ms)')
plt.legend()

plt.tight_layout()
plt.show()

