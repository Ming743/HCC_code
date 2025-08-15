import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("_Liver_16388_39_features_4_label_20240910.csv")
# === Step 1: 取得原始 time 資料 ===
time_vals = data["time"].values
event_vals = data["event"].values

# === Step 2: 建立比例曲線 ===
T_curve = np.sort(np.unique(time_vals))  # 用實際出現過的時間點

ratio_curve = []
for T in T_curve:
    valid_mask = ~((time_vals <= T) & (event_vals == 0))
    valid_time = time_vals[valid_mask]
    ratio = np.mean(valid_time >= T)
    ratio_curve.append(ratio)

# 找最接近目標比例的 T 值
target_ratios = np.linspace(0.9, 0.1, 9)
T_selected = []

for r in target_ratios:
    idx = np.argmin(np.abs(np.array(ratio_curve) - r))
    T_selected.append(T_curve[idx])  # 這裡的 T 一定是資料內真實出現過的值

