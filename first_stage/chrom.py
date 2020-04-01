import numpy as np
from scipy.signal import butter, lfilter
from scipy.signal import get_window

# Функция вычисления хром-сигнала
def chrom(BGR_signal, fps, interval_length = None):
    
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
     
    # Проверка допустимости значения ширины окна Хеннинга и установка значения в случае допустимого
    if(interval_length == None):
        # В статье Haan2013 использовались записи в 20 кадров/сек при ширине окна равной 32
        # Для сохранения пропорций введен множитель 32/20 (при fps = 20, получим окно шириной 32)
        interval_size = int(fps*(32.0/20.0))
    elif(interval_length>0):
        interval_size = interval_length//1
    else: 
        # Недопустимое значение ширины окна Хеннинга, значение должно быть не менее 32
        raise NameError('WrongIntervalLength')
    
    # Проверка допустимости размера исходных данных
    if(num_frames<interval_size):
        # Недопустимая длина исходных данных, длина исходных данных должна быть не меньше окна Хеннинга
        raise NameError('NotEnoughData')
    
    # Разделение исходного сигнала на каналы R,G,B
    R = BGR_signal[:, 2]
    G = BGR_signal[:, 1]
    B = BGR_signal[:, 0]

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

    #-------------------------------------------------------------------
    # Функция вычисления сигнала S на интервале
    def S_signal_on_interval(low_limit,high_limit):
        
        # Выделение отрывков R,G,B на интервале и их нормализация
        if (low_limit<0.0):
            num_minus = abs(low_limit)
            R_interval = np.append(np.zeros(num_minus), R[0:high_limit+1])
            R_interval_norm = R_interval/R_interval[num_minus:interval_size].mean()
            G_interval = np.append(np.zeros(num_minus), G[0:high_limit+1])
            G_interval_norm = G_interval/G_interval[num_minus:interval_size].mean()
            B_interval = np.append(np.zeros(num_minus), B[0:high_limit+1])
            B_interval_norm = B_interval/B_interval[num_minus:interval_size].mean()
        elif (high_limit>num_frames):
            num_plus = high_limit-num_frames
            R_interval = np.append(R[low_limit:num_frames], np.zeros(num_plus+1))
            R_interval_norm = R_interval/R_interval[0:interval_size-num_plus-1].mean()
            G_interval = np.append(G[low_limit:num_frames], np.zeros(num_plus+1))
            G_interval_norm = G_interval/G_interval[0:interval_size-num_plus-1].mean()
            B_interval = np.append(B[low_limit:num_frames], np.zeros(num_plus+1))
            B_interval_norm = B_interval/B_interval[0:interval_size-num_plus-1].mean()
        else:
            R_interval = R[low_limit:high_limit+1] 
            R_interval_norm = R_interval/R_interval.mean()
            G_interval = G[low_limit:high_limit+1] 
            G_interval_norm = G_interval/G_interval.mean()
            B_interval = B[low_limit:high_limit+1]
            B_interval_norm = B_interval/B_interval.mean()           
        
        # Вычисление составляющих Xs и Ys
        Xs,Ys = np.zeros(interval_size), np.zeros(interval_size)
        Xs = 3.0*R_interval_norm - 2.0*G_interval_norm
        Ys = 1.5*R_interval_norm + G_interval_norm - 1.5*B_interval_norm
        
        # Вызов функции фильтрации (фильтрация Xs и Ys полосовым фильтром от 0.5 до 4 Гц)
        Xf = bandpass_filter(Xs, 0.5, 4.0)
        Yf = bandpass_filter(Ys, 0.5, 4.0)
        
        # Вычисление сигнала S до применения окна Хеннинга
        alpha = Xf.std()/Yf.std()
        S_before = Xf - alpha*Yf
        
        return S_before
    #-------------------------------------------------------------------
        
    # Поиск количества интервалов
    number_interval = 2.0*num_frames/interval_size+1
    number_interval = int(number_interval//1)
    
    # Поиск границ интервалов и вычисление в них итогового сигнала
    intervals = []
    S_before_on_interval = []
    for i in range(int(number_interval)):
        i_low = int((i-1)*interval_size/2.0 + 1)
        i_high = int((i+1)*interval_size/2.0)
        intervals.append([i_low, i_high])
        S_before_on_interval.append(S_signal_on_interval(i_low,i_high))  
    
    # Вычисление окна Хеннинга
    wh = get_window('hamming', interval_size)    
    
    # Поиск индексов точек, в которых нет пересечения окон Хеннинга
    index_without_henning = []
    # Слева
    for i in range(intervals[0][0], intervals[1][0], 1):
        if(i>=0): 
            index_without_henning.append(i)
    # Справа
    for i in range(intervals[len(intervals)-2][1]+1, intervals[len(intervals)-1][1], 1):
        if(i<=num_frames): 
            index_without_henning.append(i)
    
    # Расчет итогового сигнала
    S_after = np.zeros(num_frames)
    for i in range(num_frames):
        for j in intervals:
            if(i>=j[0] and i <=j[1]):
                num_interval = intervals.index(j)
                num_element_on_interval = i - intervals[num_interval][0]
                if(i not in index_without_henning):
                    S_after[i] += S_before_on_interval[num_interval][num_element_on_interval]*wh[num_element_on_interval]
                else: 
                    S_after[i] += S_before_on_interval[num_interval][num_element_on_interval]
                    
    return S_after