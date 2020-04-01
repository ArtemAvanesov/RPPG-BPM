import numpy as np
import numpy.linalg as linalg

# Функция вычисления 2SR-сигнала
def SSR(X, fps, l = None):
    
    # Количество кадров в исходных данных
    num_frames = len(X)

    # Добавка к знаменателю и числителю дроби
    eps = 1e-1
    
    # Проверка допустимости размера исходных данных
    if(num_frames==0):
        # Массив исходных данных пуст
        raise NameError('EmptyData')
    
    # l - длина временного шага
    if(l == None):
        # В статье Wang2016 оптимальным считается значение 20
        l = 20
    
    # C - корреляционная матрица
    C = np.zeros([num_frames,3,3])
    # A - собственные числа матрицы C
    A = np.zeros([num_frames,3])
    # U - собственные векторы матрицы C
    U = np.zeros([num_frames,3,3])
    # P - пульс-сигнал
    P = np.zeros(num_frames)
    
    for k in range(num_frames):
        # Вычисление корреляционной матрицы С (dot - произведение двух массивов)
        C[k] = np.matmul( X[k].reshape(len(X[k]), -1), X[k].reshape(-1, len(X[k])) ) /num_frames
        # Вычисление собственных чисел и собственных векторов матрицы С
        A[k], U[k] = linalg.eig(C[k])
        
        tau = k-l+1
        if (tau>0):
            # Матрица участвующая в подпространственном вращении
            SR_array = []
            for t in range(tau,k,1):
                # Из-за особенностей numpy векторы изначально хранятся как строки, поэтому операции транспонирования инвертированы
                # Первое слагаемое выражения из 7-й строки алгоритма (проблема - при делении участвуют нули или малые значения, в итоге возникают NaN, inf и -inf значения)
                first_term = (np.sqrt(abs((eps + A[t][0])/(eps + A[tau][1]))))*U[t][0]*(np.transpose(U[tau][1].reshape(len(U[tau][1]), -1)))*U[tau][1]
                # Второе слагаемое выражения из 7-й строки алгоритма (аналогичная проблема как и с первым слагаемым)
                second_term = (np.sqrt(abs((eps + A[t][0])/(eps + A[tau][2]))))*U[t][0]*(np.transpose(U[tau][2].reshape(len(U[tau][2]), -1)))*U[tau][2]
                # Для удобства выражение разбито на два слагаемых
                SR = first_term + second_term
                #
                SR_array.append(SR[0])
    print(np.asarray(SR_array, dtype=np.float32))
    return P