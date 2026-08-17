import numpy as np
import matplotlib.pyplot as plt

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
    # set y range of 
    plt.ylim(*y_range)
    plt.stem(np.arange(-INF, INF + 1, 1), signal)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    if saveTo is not None:
        plt.savefig(saveTo)
    # plt.show()

def init_signal():
    return np.zeros(2 * INF + 1)



def time_shift_signal(x : np.ndarray, k : int) -> np.ndarray:
    """
    Shifts the signal by k units to create x[n - k].
    Values shifted beyond the [-8, 8] window are replaced with zeros.
    """
    # Create an empty template array filled with zeros
    shifted = np.zeros_like(x)
    
    # Calculate index bounds to safely copy slicing windows without loop logic
    # Positive k moves elements right; Negative k moves elements left
    start_src = max(0, -k)
    end_src = min(len(x), len(x) - k)
    
    start_dest = max(0, k)
    end_dest = min(len(x), len(x) + k)
    
    # Vectorized assignment transfers the valid block instantly
    shifted[start_dest:end_dest] = x[start_src:end_src]
    return shifted


def time_scale_signal(x : np.ndarray, k : int, *args) -> np.ndarray:
    """
    Downsamples the signal by factor k to create x[kn].
    Uses vectorized time-index maps for perfect index matching.
    """
    # 1. Define the actual math time index grid [-8 to 8]
    n = np.arange(-INF, INF + 1)
    
    # 2. Find the scaled indices we need to look up (kn)
    scaled_n = k * n
    
    # 3. Mask out any positions where scaled index lands outside our [-8, 8] buffer
    valid_mask = (scaled_n >= -INF) & (scaled_n <= INF)
    
    # 4. Map scaled coordinates directly to physical array positions (0 to 16)
    # Physical index = Math Index + Offset (INF)
    array_indices = scaled_n[valid_mask] + INF
    
    # 5. Populate and return output
    scaled = np.zeros_like(x)
    scaled[valid_mask] = x[array_indices]
    return scaled
