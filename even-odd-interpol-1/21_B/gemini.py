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
    plt.show() # Added to ensure plots pop up directly on your screen

def init_signal():
    return np.zeros(2 * INF + 1)


def time_scale_signal(x : np.ndarray, k : int) -> np.ndarray:
    """
    Task 1: Time scaling x[n/k] (upsampling / stretching).
    Intermediate newly created samples are set to 0.
    """
    # 1. Define the math time grid [-8 to 8]
    n = np.arange(-INF, INF + 1)
    
    # 2. To build x[n/k], the original data points come from positions where (n / k) is an integer
    # Check which values dividing perfectly by k fall within our bounds
    valid_mask = (n % k == 0)
    
    # 3. Calculate what mathematical index from the original signal maps here
    # For example, if k=3, new index n=3 maps to original index 3/3 = 1
    orig_n = n[valid_mask] // k
    
    # 4. Map the valid mathematical index to the physical memory slot [0 to 16]
    orig_array_idx = orig_n + INF
    
    # 5. Assign values to our empty template
    scaled = np.zeros_like(x)
    scaled[valid_mask] = x[orig_array_idx]
    
    return scaled


def time_scale_signal_interpolate(x : np.ndarray, k : int) -> np.ndarray:
    """
    Task 2: Time scaling x[n/k] with linear interpolation for intermediate samples.
    Intermediate samples are set to the average of the two original surrounding samples.
    """
    # Quick bypass: if no scaling happens, return original signal directly
    if k == 1:
        return x.copy()
        
    # 1. Start with the structured grid from Task 1
    scaled = time_scale_signal(x, k)
    
    # 2. Define the mathematical grid [-8 to 8]
    n = np.arange(-INF, INF + 1)
    
    # 3. Identify all intermediate samples (the positions where n is NOT perfectly divisible by k)
    inter_mask = (n % k != 0)
    inter_n = n[inter_mask]
    
    # 4. Find the two surrounding ORIGINAL mathematical indices in x
    # left_orig corresponds to the original point immediately to the left: floor(n/k)
    # right_orig corresponds to the original point immediately to the right: ceil(n/k)
    left_orig = np.floor(inter_n / k).astype(int)
    right_orig = np.ceil(inter_n / k).astype(int)
    
    # 5. Check boundary constraints: if a surrounding math index falls outside [-8, 8],
    # its value defaults to 0 according to problem specs.
    left_valid = (left_orig >= -INF) & (left_orig <= INF)
    right_valid = (right_orig >= -INF) & (right_orig <= INF)
    
    # 6. Extract values from original array safely using boolean masking
    left_vals = np.zeros(len(inter_n))
    right_vals = np.zeros(len(inter_n))
    
    left_vals[left_valid] = x[left_orig[left_valid] + INF]
    right_vals[right_valid] = x[right_orig[right_valid] + INF]
    
    # 7. Compute the average values and drop them into the intermediate mask gaps
    scaled[inter_mask] = (left_vals + right_vals) / 2.0
    
    return scaled


def main():
    img_root = '.'
    signal = init_signal()
    signal[INF] = 1
    signal[INF+1] = .5
    signal[INF-1] = 2
    signal[INF + 2] = 1
    signal[INF - 2] = .5

    plot(signal, title='Original Signal(x[n])', saveTo=f'{img_root}/x[n].png')
    plot(time_scale_signal(signal, 3), title='x[n/3]', saveTo=f'{img_root}/x[n divided by 3].png')
    plot(time_scale_signal(signal, 1), title='x[n/1]', saveTo=f'{img_root}/x[n divided by 1].png')
    plot(time_scale_signal_interpolate(signal, 3), title='x[n/3] with interpolation', saveTo=f'{img_root}/x[n divided by 3]_with_interpolation.png')
    plot(time_scale_signal_interpolate(signal, 1), title='x[n/1] with interpolation', saveTo=f'{img_root}/x[n divided by 1]_with_interpolation.png')

if __name__ == '__main__':
    main()
