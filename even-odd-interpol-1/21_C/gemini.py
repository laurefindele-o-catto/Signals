import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

INF = 8

def plot(
        signal, 
        title=None, 
        y_range=(-1, 3), 
        figsize = (8, 3),
        x_label='n (Time Index)',
        y_label='x[n]',
        saveTo=None
    ):
    plt.figure(figsize=figsize)
    plt.xticks(np.arange(-INF, INF + 1, 1))
    
    y_range = (y_range[0], max(np.max(signal), y_range[1]) + 1)
    plt.ylim(*y_range)
    plt.stem(np.arange(-INF, INF + 1, 1), signal)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    if saveTo is not None:
        plt.savefig(saveTo)
    plt.show() # Ensures windows render on your screen

def init_signal():
    return np.zeros(2 * INF + 1)


def time_reverse_signal(x : np.ndarray) -> np.ndarray:
    """
    Returns the time-reversed version of the signal, x[-n].
    Uses np.flip to reverse the array indices across the symmetric axis.
    """
    return np.flip(x)


def odd_even_decomposition(x : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decomposes a signal into its Even and Odd parts.
    Returns: (odd_signal, even_signal)
    """
    # 1. Obtain the time-reversed version x[-n] using np.flip
    x_reversed = time_reverse_signal(x)
    
    # 2. Vectorized mathematical formulas:
    # Even part: x_even[n] = 0.5 * (x[n] + x[-n])
    # Odd part:  x_odd[n]  = 0.5 * (x[n] - x[-n])
    even_signal = 0.5 * (x + x_reversed)
    odd_signal = 0.5 * (x - x_reversed)
    
    return odd_signal, even_signal


def main():
    img_root_path = '.'
    signal = init_signal()
    signal[INF] = 1
    signal[INF+1] = .5
    signal[INF-1] = 2
    signal[INF + 2] = 1
    signal[INF - 2] = .5

    plot(signal, title='Original Signal(x[n])', saveTo=f'{img_root_path}/x[n].png')
    reversed_sig = time_reverse_signal(signal)
    plot(reversed_sig, title='x[-n]', saveTo=f'{img_root_path}/x[-n].png')
    plot(time_reverse_signal(reversed_sig), title='x[-(-n)]', saveTo=f'{img_root_path}/x[-(-n)].png')
    odd_signal, even_signal = odd_even_decomposition(signal)
    plot(odd_signal, title='Odd Signal(x[n])', saveTo=f'{img_root_path}/x_odd[n].png')
    plot(even_signal, title='Even Signal(x[n])', saveTo=f'{img_root_path}/x_even[n].png')

if __name__ == '__main__':
    main()
