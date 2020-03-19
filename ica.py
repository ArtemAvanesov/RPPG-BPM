import numpy as np
from sklearn.decomposition import FastICA
from scipy.signal import butter, lfilter

# Функция вычисления ICA-сигнала
def ica(BGR_signal, fps):
    
    # Количество кадров в исходных данных
    num_frames = len(BGR_signal)
    
    # Проверка допустимости размера исходных данных
    if(num_frames==0):
        # Массив исходных данных пуст
        raise NameError('EmptyData')
    
    # Проверка допустимости значения fps
    if(fps<9):
        # Недопустимое значение fps, для работы полосового фильтра требуется fps>=9
        raise NameError('WrongFPS')
    
    # Разделение исходного сигнала на каналы R,G,B (y1, y2, y3)
    y1 = BGR_signal[:, 2]
    y2 = BGR_signal[:, 1]
    y3 = BGR_signal[:, 0]

    # Нормализация
    y1_norm = np.zeros(num_frames)
    y2_norm = np.zeros(num_frames)
    y3_norm = np.zeros(num_frames)
    for i in range(num_frames):
        y1_norm[i] = (y1[i] - y1.mean())/y1.std()
        y2_norm[i] = (y2[i] - y2.mean())/y2.std()
        y3_norm[i] = (y3[i] - y3.mean())/y3.std()
    
    # Функция полосовой фильтрации
    def bandpass_filter(data, lowcut, highcut):
        fs = fps # Частота дискретизации (количество измерений сигнала в 1 сек)
        nyq = 0.5 * fs # Частота Найквиста
        low = float(lowcut) / float(nyq)
        high = float(highcut) / float(nyq)
        order = 6.0 # Номер фильтра в scipy.signal.butter
        b, a = butter(order, [low, high], btype='band')
        bandpass = lfilter(b, a, data)
        return bandpass  
    
    # Полосовая фильтрация каналов
    y1_filtered = bandpass_filter(y1_norm, 0.7, 4.0)
    y2_filtered = bandpass_filter(y2_norm, 0.7, 4.0)
    y3_filtered = bandpass_filter(y3_norm, 0.7, 4.0)
    y_filtered = []
    for i in range(num_frames):
        y_filtered.append([y1_filtered[i],y2_filtered[i],y3_filtered[i]])
    
    # Вычисление ICA-сигнала
    ica = FastICA(n_components=1, random_state=0)
    ICA = ica.fit_transform(y_filtered).reshape(1, -1)[0]
    
    return ICA