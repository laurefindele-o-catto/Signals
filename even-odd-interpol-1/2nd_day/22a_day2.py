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
    dt = t_original[1] - t_original[0]
    n = len(t_original)
    t0 = t_original[0]
    
    idx = (t_query - t0)/dt
    idx_round = np.round(idx)
    
    is_exact = np.isclose(idx, idx_round, atol = 1e-6)
    
    idx_left = np.where(is_exact, idx_round, np.floor(idx))
    idx_right = np.where(is_exact, idx_round, np.ceil(idx))
    
    #clipping
    idx_left = np.clip(idx_left, 0, n-1).astype(int)
    idx_right = np.clip(idx_right, 0, n-1).astype(int)
    
    x_query = 0.5*(x_original[idx_left] + x_original[idx_right])
    return x_query


def time_scale(
    t: np.ndarray,
    x: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Time sub-scaling:
        y(t) = x(t / k)
    """
    t_query = t/k
    y = interpolate_signal(t, x, t_query)
    
    #clip out of ranges
    out_of_range = (t_query<t[0]) | (t_query > t[-1])
    y = np.where(out_of_range, np.nan, y)
    
    return y


def plot_pair(t: np.ndarray, x: np.ndarray, y: np.ndarray, title: str):
    """
    Plot graphs.
    """
    plt.figure(figsize=(10,5))
    plt.plot(t,x, label="x(t)", linewidth = 1.5)
    plt.plot(t, y, label = "y(t)", linewidth = 1.5, linestyle = "--")
    plt.xlabel("t")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()


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
