import numpy as np
import matplotlib.pyplot as plt
import math

class DiscreteSignal:
    def __init__(self, s, e):
        self.s = s
        self.e = e
        self.a = np.zeros(e-s+1)

    def set_value_at_time(self, t, value):
        self.a[t - self.s] = value

    def get_value_at_time(self, t):
        out_of_range = (t < self.s) | (t>self.e)
        if out_of_range:
            return 0
        
        return self.a[t-self.s]
    
    def shift(self, k):
        newSignal = DiscreteSignal(self.s + k, self.e + k)
        newSignal.a = self.a.copy()
        return newSignal
    

class LTISystem:
    def __init__(self, h_signal):
        self.h_signal = h_signal
    def output(self, input_signal):
        s = self.h_signal.s + input_signal.s
        e = self.h_signal.e + input_signal.e

        y = DiscreteSignal(s, e)

        for n in range(s, e+1):
            y_n = 0
            for k in range(input_signal.s, input_signal.e+1):
                y_n += input_signal.get_value_at_time(k) * self.h_signal.get_value_at_time(n-k)

            y.set_value_at_time(n, y_n)

        return y


def main():
    s = -2
    e = 4
    t1 = [i for i in range(s, e+1)]
    sig1 = DiscreteSignal(s, e)
    sig1.set_value_at_time(0, 3)
    sig1.set_value_at_time(1, 2)
    sig1.set_value_at_time(2, -5)

    t2 = [i+2 for i in t1]
    sig2 = sig1.shift(2)
    print(sig2.get_value_at_time(4))

    print(sig1.a)
    print(sig2.a)

    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=False)

    ax[0].stem(t1, sig1.a, linefmt="r-", markerfmt="ro", basefmt="r-", label="Original")
    # Plot second array in blue
    ax[0].stem(t2, sig2.a, linefmt="b-", markerfmt="bo", basefmt="b-", label="Shifted")
    ax[0].set_title("Shifting")
    ax[0].legend()

    h_sig = DiscreteSignal(-1, 2)
    t_impulse = [i for i in range(-1, 3)]

    for t in t_impulse:
        h_sig.set_value_at_time(t, 1)

    ax[1].stem(t_impulse, h_sig.a, linefmt="r-", markerfmt="ro", basefmt="r-", label="Impulse Function")
    ax[1].set_title("Impulse Function")

    conv = LTISystem(h_sig)
    y = conv.output(sig1)

    t_conv = list(range(y.s, y.e + 1))

    ax[2].stem(t_conv, y.a, linefmt="r-", markerfmt="ro", basefmt="r-", label="Convolution")
    ax[2].set_title("Convolution")
    ax[2].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()