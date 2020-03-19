import numpy as np
import pywt # библиотеку нужно качать (pip install PyWavelets)
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Вейвлет анализ
def wavelet_analysis(ppg_signal, fps, wavelet=None):
    
    length_data = len(ppg_signal)
    if(length_data == 0):
        return None
    
    if (wavelet==None):
        wavelet = 'mexh'
    else: wavelet = wavelet
    
    min_freq = 36  # Минимальный bpm
    max_freq = 240  # Максимальный bpm
    min_freq_hz = min_freq/60 # Минимальный bpm в герцах
    min_scale = min_freq_hz/2 # Нижняя граница масштаба (расчет по статье Unakafov2018 раздел 2.2.5)
    max_scale = fps/2 # Верхняя граница масштаба (расчет по статье Unakafov2018 раздел 2.2.5)
    sampling_period = 1/fps # Период дискретизации
    scales = np.arange(min_scale,max_scale, 2**0.03125) # Массив масштабов (шаг взят со статьи Unakafov2018 примечание 7)
    
    # Расчет вейвлета
    # Доступные вейвлет-функции: print(pywt.wavelist())
    coef, freqs = pywt.cwt(data = ppg_signal, scales = scales, wavelet = wavelet, sampling_period = sampling_period)
    
    # Поиск вейвлета с максимальной суммой коэффициентов (по алгоритму со статьи Huang2016 формула 13)
    max_sum = 0
    index_max = 0
    for i in range(len(coef)):
        sum_coef = 0
        for j in range(len(coef[i])):
            sum_coef = sum_coef+coef[i][j]
        if(sum_coef>max_sum):
            max_sum = sum_coef
            index_max = i
    
    # Визуальный анализ аномалий в пульсе
    plt.figure(figsize=(30,5))
    plt.title("Анализ аномалий в пульсе ({})".format(wavelet), fontsize = 25)
    plt.imshow(coef)
    
    # Применение к выбранному вейвлету межпикового анализа (межпиковый анализ используется в алгоритме со статьи Huang2016)
    length_data = len(coef[index_max])
    max_num_peaks = ((length_data/fps)/60)*max_freq # Максимальное количество пиков
    min_distance = length_data/max_num_peaks # Минимальное расстояние между пиками
    peaks = find_peaks(coef[index_max], distance=min_distance-1, prominence=10)[0] # Индексы пиков
    distances = [] # Расстояние между пиками
    for i in range(len(peaks)-1):
        distances.append(peaks[i+1] - peaks[i])
        
    distances = sorted(distances) # Сортировка по возрастанию
    M = max(1,int(len(distances)*0.5)) # Усреднение М медиан
    # Уменьшенный массив расстояний (учет выбросов)
    distances_small = distances[int(len(distances)//2 - M//2) : int(len(distances)//2 - M//2 + M)]
    one_beat_time = (sum(distances_small)/len(distances_small))/fps # Время одного сердцебиения
    hr_estimated = 1/one_beat_time # Предполагаемая ЧСС

    # Визуализация пиков
    plt.figure(figsize=(30,5))
    plt.title("Выбранный {}-вейвлет с отображением пиков".format(wavelet), fontsize = 25)
    plt.plot(np.arange(0, len(coef[index_max])), coef[index_max], color = 'green', label = wavelet)
    plt.plot(np.arange(0, len(coef[index_max])), ppg_signal, color = 'red', label = "signal")
  
    plt.legend(loc = 'best', fontsize=25)
    for item in peaks:
        plt.axvline(x=item)
        
    return hr_estimated  