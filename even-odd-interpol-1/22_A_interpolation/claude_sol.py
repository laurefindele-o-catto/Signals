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

    For each query time tq, find its position on the (uniform) original
    time grid as a fractional index. The sample just to the left is
    floor(idx), the one just to the right is ceil(idx). The interpolated
    value is the plain average of those two neighboring samples. If tq
    lands exactly on an original sample, left == right, so the average
    just returns that exact sample (e.g. y(0) = x(0)).
    """
    dt = t_original[1] - t_original[0]
    t0 = t_original[0]
    n = len(t_original)

    # Fractional index of each query point on the original grid
    idx_float = (t_query - t0) / dt

    # Snap to an exact integer index when we're numerically "on" a sample
    # (protects against floating point noise, e.g. 4.999999999 vs 5.0)
    idx_round = np.round(idx_float)
    is_exact = np.isclose(idx_float, idx_round, atol=1e-6)

    idx_left = np.where(is_exact, idx_round, np.floor(idx_float))
    idx_right = np.where(is_exact, idx_round, np.ceil(idx_float))

    # Keep indices inside valid bounds
    idx_left = np.clip(idx_left, 0, n - 1).astype(int)
    idx_right = np.clip(idx_right, 0, n - 1).astype(int)

    x_query = 0.5 * (x_original[idx_left] + x_original[idx_right])
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
    t_query = t / k
    y = interpolate_signal(t, x, t_query)

    # Ignore (blank out) any query points that fall outside the
    # original signal's time range
    out_of_range = (t_query < t[0]) | (t_query > t[-1])
    y = np.where(out_of_range, np.nan, y)
    return y


def plot_pair(t: np.ndarray, x: np.ndarray, y: np.ndarray, title: str):
    """
    Plot graphs.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(t, x, label="x(t)", linewidth=1.5)
    plt.plot(t, y, label="y(t)", linewidth=1.5, linestyle="--")
    plt.xlabel("t")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
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