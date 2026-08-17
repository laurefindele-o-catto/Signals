import numpy as np
import matplotlib.pyplot as plt

T_MIN, T_MAX, N = -4.0, 4.0, 4001

def x_of_t(t: np.ndarray) -> np.ndarray:
    """
    Base signal x(t).
    """
    tri0 = np.zeros_like(t, dtype=float)
    m0 = np.abs(t) <= 1.0
    tri0[m0] = 1.0 - np.abs(t[m0])

    ramp = np.zeros_like(t, dtype=float)
    m1 = np.abs(t) <= 1.0
    ramp[m1] = t[m1]

    tri_shift = np.zeros_like(t, dtype=float)
    u = t - 1.2
    m2 = np.abs(u) <= 1.0
    tri_shift[m2] = 1.0 - np.abs(u[m2])

    return tri0 + 0.6 * ramp + 0.4 * tri_shift


def time_reverse(x: np.ndarray) -> np.ndarray:
    """
    Given samples x(t), return samples of x(-t)
    Works because t is a symmetric, uniform grid: t[i] = -t[N-1-i],
    so x(-t[i]) = x(t[N-1-i]) = flip(x)[i].
    """
    return np.flip(x)


def even_odd_decompose(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Must call time_reverse(...) inside this function.
    """
    xr = time_reverse(x)
    xe = 0.5 * (x + xr)
    xo = 0.5 * (x - xr)
    return xe, xo


# ----------------------------
# Provided plotting (do not modify)
# ----------------------------
def plot_three(t: np.ndarray, x: np.ndarray, xe: np.ndarray, xo: np.ndarray):
    plt.figure()
    plt.plot(t, x, label="x(t)")
    plt.plot(t, xe, label="xe(t)")
    plt.plot(t, xo, label="xo(t)")
    plt.title("Even–Odd Decomposition")
    plt.xlabel("t")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()


def plot_pair(t: np.ndarray, x: np.ndarray, xr: np.ndarray):
    plt.figure()
    plt.plot(t, x, label="x(t)")
    plt.plot(t, xr, label="x(-t)")
    plt.title("Time Reversal")
    plt.xlabel("t")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()


# ----------------------------
# Main (provided)
# ----------------------------
def main():
    t = np.linspace(T_MIN, T_MAX, N)
    x = x_of_t(t)

    xr = time_reverse(x)
    xe, xo = even_odd_decompose(x)

    plot_pair(t, x, xr)
    plot_three(t, x, xe, xo)
    plt.show()


if __name__ == "__main__":
    main()