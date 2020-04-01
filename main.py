import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from first_stage.chrom import chrom
from first_stage.pos import pos
from first_stage.ica import ica
from second_stage.fourier_analysis import fourier_analysis
from second_stage.interbeats_analysis import interbeats_analysis
from second_stage.wavelet_analysis import wavelet_analysis

# Чтение исходных данных
data = np.loadtxt('test_data.csv', delimiter='\t', unpack=False)
BGR_data = data[:, :3]
BGR_data = BGR_data[21:]

# Построение таблицы с отрывком исходных данных
params = ["B-chanel", "G-chanel", "R-chanel"]
np_resuts = np.array(BGR_data[:5])
data_pd = pd.DataFrame(np_resuts)
data_pd.columns = params
print("Отрывок исходных данных:")
print(data_pd.to_string())

# Пример вызова функций CHROM, POS, ICA
signal_chrom = chrom(BGR_data, 15, 32)
signal_pos = pos(BGR_data, 15, 20)
signal_ica = ica(BGR_data, 15)

# Вызов функции с Фурье-анализом
hr_fourier_pos = fourier_analysis(signal_pos, 15)
hr_fourier_chrom = fourier_analysis(signal_chrom, 15)
hr_fourier_ica = fourier_analysis(signal_ica, 15)
# Вызов функции с Вейвлет-анализом
hr_wavelet_pos = wavelet_analysis(signal_pos, 15)
# Вызов функции с Межпиковым-анализом
hr_interbeats_pos = interbeats_analysis(signal_pos, 15)

# Построение итоговых графиков (для наглядности взят отрывок в 300 кадров)
BGR_data = BGR_data[300:600]
signal_chrom = signal_chrom[300:600]
signal_pos = signal_pos[300:600]
signal_ica = signal_ica[300:600]

signals = [signal_chrom, signal_pos, signal_ica]
signals_names = ["chrom", "pos", "ica"]

fig, axes = plt.subplots(len(signals)+1, 1, figsize=(10, 6))  

line = np.linspace(0,len(BGR_data),len(BGR_data))
axes[0].plot(line, BGR_data[:, 2:3], label = 'R', color = 'red')
axes[0].plot(line, BGR_data[:, 1:2], label = 'G', color = 'green')
axes[0].plot(line, BGR_data[:, 0:1], label = 'B', color = 'blue')
axes[0].legend(loc="upper right", fontsize=8)
axes[0].set_xlabel("Номер кадра", fontsize=9)
axes[0].set_ylabel("Значение", fontsize=9)

for i in range(1,len(signals)+1,1):
    axes[i].plot(line, signals[i-1], label = signals_names[i-1], color = 'green')
    axes[i].hlines([min(signals[i-1]), max(signals[i-1])], 0, len(signals[i-1]), linestyles='--')
    axes[i].legend(loc="upper right", fontsize=8)
    axes[i].set_xlabel("Номер кадра", fontsize=9)
    axes[i].set_ylabel("Значение", fontsize=9)
    axes[i].set_ylim(2*min(signals[i-1]),2*max(signals[i-1]))

plt.subplots_adjust(hspace=1)
plt.show()

# Построение таблицы с результатами методов
params = ["", "chrom", "pos", "ica"]
resuts = []
resuts.append(['interbeats', "-", np.around(hr_interbeats_pos*60,decimals=3),"-"])
resuts.append(['fourier', np.around(hr_fourier_chrom*60,decimals=3), np.around(hr_fourier_pos*60,decimals=3), np.around(hr_fourier_ica*60,decimals=3)])
resuts.append(['wavelet', "-", np.around(hr_wavelet_pos*60,decimals=3),"-"])
np_resuts = np.array(resuts)
data_pd = pd.DataFrame(np_resuts)
data_pd.columns = params
print("\nРезультаты методов, единица измерения - ударов/мин:")
print(data_pd.to_string())