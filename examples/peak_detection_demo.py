
import numpy as np
import matplotlib.pyplot as plt
from gwexpy.frequencyseries import FrequencySeries
import astropy.units as u

# 1. データの準備
# 複数のピークを持つ擬似的なスペクトルデータを作成します
fs = 1000  # サンプリング周波数 [Hz]
df = 1.0   # 周波数分解能 [Hz]
f_axis = np.arange(0, 500, df)
data = np.exp(-f_axis/100) * np.random.normal(1, 0.1, len(f_axis)) # ノイズ成分

# 特定の周波数にピークを追加
peak_info = [(50, 2.0), (120, 1.5), (130, 1.8), (300, 1.2)]
for f, amp in peak_info:
    idx = int(f / df)
    data[idx] += amp

# FrequencySeries オブジェクトの作成
spec = FrequencySeries(data, df=df, unit='m')

print(f"FrequencySeries created with units: {spec.unit}")

# 2. 基本的なピーク検出 (find_peaks)
# 閾値 (threshold) を指定してピークを探します
# ここでは threshold=1.5 とします
peak_indices, props = spec.find_peaks(threshold=1.5)
peak_freqs = spec.frequencies[peak_indices]
peak_values = np.abs(spec[peak_indices])

print(f"Detected peaks at frequencies: {peak_freqs}")

# 3. 可視化 (Visualization)
plt.figure(figsize=(12, 6))
plt.plot(spec.frequencies, np.abs(spec), label='Magnitude Spectrum', color='navy', alpha=0.7)
plt.scatter(peak_freqs, peak_values, color='red', marker='o', s=100, label='Basic Peaks (threshold=1.5)')

# ピークに値を表示 ( units による TypeError を避けるため v.value を使用)
for f, v in zip(peak_freqs.value, peak_values.value):
    plt.text(f, v + 0.1, f"{f:.0f}Hz\n{v:.2f}", ha='center', va='bottom', color='red', weight='bold')

# 4. 高度なピーク検出 (distance, prominence)
# distance: ピーク間の最小間隔を指定（近いピークを無視する）
# prominence: ピークの「際立ち（プロミネンス）」を指定
peak_indices_adv, props_adv = spec.find_peaks(distance=20, prominence=0.5)
peak_freqs_adv = spec.frequencies[peak_indices_adv]
peak_values_adv = np.abs(spec[peak_indices_adv])

plt.scatter(peak_freqs_adv, peak_values_adv, color='gold', marker='x', s=150, linewidths=3, label='Advanced Peaks (dist=20, prom=0.5)')

plt.xlabel(f'Frequency [{spec.frequencies.unit}]')
plt.ylabel(f'Amplitude [{spec.unit}]')
plt.title('Peak Detection Example in FrequencySeries')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.savefig('peak_detection_demo.png')
print("Demo plot saved to 'peak_detection_demo.png'")
plt.show()

# 5. 異なるメソッドでの検出 (db)
# デシベルスケールでのピーク検出も可能です
peak_indices_db, _ = spec.find_peaks(threshold=0, method='db') # 0dB以上
print(f"Detected peaks in dB above 0dB: {spec.frequencies[peak_indices_db]}")
