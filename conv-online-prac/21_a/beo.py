import numpy as np
import matplotlib.pyplot as plt

# Assuming your DiscreteSignal (or Signal) and LTISystem classes are defined above

# Stock Market Prices as a Python List
# price_list = list(map(int, input("Stock Prices: ").split()))
# n = int(input("Window size: "))
# alpha = float(input("Alpha: "))

# You may use the following input for testing purpose
price_list = [10, 11, 12, 9, 10, 13, 15, 16, 17, 18]
n = 3
alpha = 0.8

# ---- 1. Determine the proper input signal x[n] ----
# The input signal spans from index 0 to len(price_list) - 1
m = len(price_list)
x_signal = DiscreteSignal(0, m - 1)
for i, price in enumerate(price_list):
    x_signal.set_value_at_time(i, float(price))

# ---- 2. Determine the proper impulse response h[k] ----
# h[k] = alpha * (1 - alpha)^k for k = 0 ... n-1
h_signal = DiscreteSignal(0, n - 1)
for k in range(n):
    h_val = alpha * ((1 - alpha) ** k)
    h_signal.set_value_at_time(k, h_val)

# ---- 3. Convolve using your LTISystem machinery ----
system = LTISystem(h_signal)
y_signal = system.output(x_signal)

# ---- 4. Extract the relevant window outputs ----
# The first complete window calculation occurs at time index (n - 1)
exsm = []
for time_idx in range(n - 1, m):
    exsm.append(y_signal.get_value_at_time(time_idx))

# Print according to required I/O format
print("Exponential Smoothing: " + ", ".join(f"{num:.2f}" for num in exsm))
# Output should be: 11.68, 9.47, 9.82, 12.29, 14.40, 15.62, 16.64, 17.63
