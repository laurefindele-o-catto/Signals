import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Time axis
# ----------------------------
T_MIN, T_MAX, N = -4.0, 4.0, 4001


def x_of_t(t: np.ndarray) -> np.ndarray:
    """
    Base signal x(t): sinusoidal signal
    """
    return (
        np.sin(2 * np.pi * 0.5 * t)
        + 0.5 * np.sin(2 * np.pi * 1.5 * t)
    )


# ==========================================================
# ANSWER IMPLEMENTATION
# ==========================================================

def interpolate_signal(
    t_original: np.ndarray,
    x_original: np.ndarray,
    t_query: np.ndarray
) -> np.ndarray:
    """
    Interpolate using average of two neighboring samples.
    """
    t_0 = t_original[0]
    d = t_original[1]-t_original[0]
    n = len(t_original)

    idx_float = (t_query - t_0)/ d
    idx_round = np.round(idx_float)
    is_close = np.isclose(idx_float, idx_round, atol=1e-6)

    left_idx = np.floor(idx_float)
    right_idx = np.ceil(idx_float)
    left_idx_adj = np.where(is_close,idx_round,left_idx).astype(int)
    right_idx_adj = np.where(is_close, idx_round,right_idx).astype(int)

    left_idx_adj = np.clip(left_idx_adj, 0, n-1)
    right_idx_adj= np.clip(right_idx_adj, 0 , n-1)

    x_new = 0.5*(x_original[left_idx_adj]+x_original[right_idx_adj])

    return x_new



def time_scale(
    t: np.ndarray,
    x: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Time sub-scaling:
        y(t) = x(t / k)
    """
    t_k = t/ k 
    x_interpolated = interpolate_signal(t,x,t_k)

    out_of_range = (t_k < t[0]) | (t_k > t[-1])
    x_interpolated = np.where(out_of_range, np.nan, x_interpolated)

    return x_interpolated


def plot_pair(t: np.ndarray, x: np.ndarray, y: np.ndarray, title: str):
    """
    Plot graphs.
    """
    plt.figure(figsize=(10,5))
    plt.plot(t,x, label='x(t)', color='r', linewidth=0.5)
    plt.plot(t,y, label='y(t)', linewidth=0.9, color='g')
    plt.xlabel("time")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


# ----------------------------
# Main
# ----------------------------
def main():
    t = np.linspace(T_MIN, T_MAX, N)
    x = x_of_t(t)

    k = 2   # sub-scaling factor
    y = time_scale(t, x, k)

    plot_pair(
        t,
        x,
        y,
        title=f"Time Sub-scaling: y(t) = x(t / {k})"
    )
    plt.show()


if __name__ == "__main__":
    main()
