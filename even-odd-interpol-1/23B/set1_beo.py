import numpy as np
import matplotlib.pyplot as plt

DT = 0.05 # sampling interval for the time axis
T_MIN, T_MAX = -np.pi, np.pi # x(t) is defined only on this range

def generate_time_axis(t_min=T_MIN, t_max=T_MAX, dt=DT):
    return np.arange(t_min, t_max + dt / 2, dt)


def base_signal(t):
    x = np.sin(t)
    x[(t < T_MIN) | (t > T_MAX)] = 0
    return x

def interpolate_signal(t, x, query_t):
    
    # TODO: implement interpolation
    
    dt = t[1] - t[0]
    t0 = t[0]
    n = len(t)
    
    idx = (query_t-t0)/dt
    idx_round = np.round(idx)
    
    is_close = np.isclose(idx, idx_round, atol = 1e-6)
    idx_left = np.where(is_close, idx_round, np.floor(idx))
    idx_right = np.where(is_close, idx_round, np.ceil(idx))
    
    idx_left = np.clip(idx_left, 0, n-1).astype(int)
    idx_right = np.clip(idx_right, 0, n-1).astype(int)
    
    x_interp = 0.5*(x[idx_left] + x[idx_right])
    
    return x_interp

def transform_signal(t, x, alpha, beta):
    
    # TODO: implement transformation
    
    t_new = t*alpha + beta
    
    y = interpolate_signal(t, x, t_new)
    
    out_of_bounds = (t_new > T_MAX) | (t_new < T_MIN)
    y = np.where(out_of_bounds, np.nan, y)
    
    return y

def plot_signals(t, x, y, alpha, beta):
    plt.figure(figsize=(9, 5))
    plt.plot(t, x, label="x(t)", linewidth=2)
    plt.plot(t, y, label=f"y(t) = x({alpha}t + {beta})", linewidth=2, linestyle="--")
    plt.title("Time Scaling and Shifting of a Signal")
    plt.xlabel("t")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    t = generate_time_axis()
    x = base_signal(t)

    print("Enter alpha and beta to plot y(t) = x(alpha*t + beta).")
    print("Type 'q' at any prompt to quit.\n")

    while True:
        line = input("Enter alpha and beta with a space seperating them :\n")
        values = line.split()
        if values[0]=='q' :
            break
        else :
            alpha = float(values[0])
            beta = float(values[1])
            
        y = transform_signal(t,x, alpha, beta)

    print("Exiting.")


if __name__ == "__main__":
    main()