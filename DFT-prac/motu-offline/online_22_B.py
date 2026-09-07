from Offline_Code.transforms import (FFTTransformer, next_power_of_two)
import numpy as np

def fft(a):
    n = len(a)
    n = next_power_of_two(n)
    a_new = np.zeros(n)
    a_new[:n]= a 
    engine = FFTTransformer()
    return engine.transform(a_new)

def ifft(a):
        n = len(a)
        n = next_power_of_two(n)
        a_new = np.zeros(n, dtype=complex)
        a_new[:n]= a 
        engine = FFTTransformer()
        return engine.inverse(a_new)


def weighted_polynomial_multiply(P, Q, W):
    #implement
    P = np.array(P)
    Q = np.array(Q)
    npp = len(P)
    nq = len(Q)
    window = npp + nq -1
    window2 = next_power_of_two(window)
    P_n = np.zeros(window2)
    Q_n = np.zeros(window2)
    P_n[:npp]= P * W 
    Q_n[:nq]= Q
    
    a = fft(P_n)
    b = fft(Q_n)
    result = a * b 
    engine = FFTTransformer()
    result = engine.inverse(result).real
    return np.round(result[:window]).astype(int)
    

if __name__ == "__main__":
    P = [1, 3, 2, 6, 7]
    Q = [4,1]
    W = [3, 2, 1, 5, 6]
 

    R = weighted_polynomial_multiply(P, Q, W)

    print("Result:", R)

    # 12, 27, 14, 122, 198, 42