from scipy.signal import find_peaks

# Межпиковый анализ
def interbeats_analysis(ppg_signal, fps):
    
    length_data = len(ppg_signal)
    #min_freq = 45  # Минимальный bpm
    max_freq = 160  # Максимальный bpm
    #min_num_peaks = ((length_data/fps)/60)*min_freq # Минимальное количество пиков
    max_num_peaks = ((length_data/fps)/60)*max_freq # Максимальное количество пиков
    min_distance = length_data/max_num_peaks # Минимальное расстояние между пиками
    #max_distance = length_data/min_num_peaks # Максимальное расстояние между пиками
    peaks = find_peaks(ppg_signal, distance=min_distance-1, prominence=10)[0] # Индексы пиков
    distances = [] # Расстояние между пиками
    for i in range(len(peaks)-1):
        distances.append(peaks[i+1] - peaks[i])   
        
    import numpy as np
    import matplotlib.pyplot as plt
    line = np.linspace(0,len(ppg_signal),len(ppg_signal))
    plt.figure(figsize=(30,5))
    plt.plot(line, ppg_signal)
    for item in peaks:
        plt.axvline(x=item)
    
    if(len(distances) == 0):
        return None
    else:
        distances = sorted(distances) # Сортировка по возрастанию
        M = max(1,int(len(distances)*0.5)) # Усреднение М медиан
        # Уменьшенный массив расстояний (учет выбросов)
        distances_small = distances[int(len(distances)//2 - M//2) : int(len(distances)//2 - M//2 + M)]
        one_beat_time = (sum(distances_small)/len(distances_small))/fps # Время одного сердцебиения
        hr_estimated = 1/one_beat_time # Предполагаемая ЧСС

        return hr_estimated    