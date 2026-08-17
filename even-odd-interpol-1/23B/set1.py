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
    d = t[1]- t[0]
    n = len(t)
    t_0 = t[0]

    idx_float = (query_t-t_0)/d
    idx_round = np.round(idx_float)

    is_close = np.isclose(idx_float, idx_round, atol=1e-6)
    idx_left = np.where(is_close, idx_round, np.floor(idx_float)).astype(int)
    idx_right = np.where(is_close, idx_round, np.ceil(idx_float)).astype(int)

    idx_left = np.clip(idx_left,0, n-1)
    idx_right= np.clip(idx_right,0,n-1)

    x_interpolated = 0.5 * (x[idx_left]+x[idx_right])
    return x_interpolated



    
   

def transform_signal(t, x, alpha, beta):
    
    # TODO: implement transformation
    query_t = (t*alpha) + beta
    y = interpolate_signal(t,x,query_t)
    out_of_bounds = (query_t > T_MAX) | (query_t < T_MIN)
    y = np.where(out_of_bounds, np.nan,y)
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
        
        # TODO: complete the loop
        line = input("Enter alpha and beta with a space seperating them :\n")
        values = line.split()
        if values[0]=='q' :
            break
        else :
            alpha = float(values[0])
            beta = float(values[1])
        y = transform_signal(t,x,alpha, beta)
        plot_signals(t,x,y,alpha,beta)

    print("Exiting.")


if __name__ == "__main__":
    main()