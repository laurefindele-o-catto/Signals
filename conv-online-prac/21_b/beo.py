import numpy as np

# Assuming your DiscreteSignal (or Signal) and LTISystem classes are defined above

# Stock Market Prices as a Python List
price_list = list(map(int, input("Stock Prices: ").split()))
n = int(input("Window size: "))

# Setup basic parameters
m = len(price_list)

# ---- 1. Formulate the Input Signal x[n] ----
x_signal = DiscreteSignal(0, m - 1)
for i, price in enumerate(price_list):
    x_signal.set_value_at_time(i, float(price))

# ---- 2. Formulate Unweighted Moving Average Impulse Response h_uma[k] ----
h_uma = DiscreteSignal(0, n - 1)
for k in range(n):
    h_uma.set_value_at_time(k, 1.0 / n)

# ---- 3. Formulate Weighted Moving Average Impulse Response h_wma[k] ----
h_wma = DiscreteSignal(0, n - 1)
sum_of_weights = sum(range(1, n + 1))  # Normalization denominator (n * (n + 1) / 2)

for k in range(n):
    # Due to convolution flipping, k=0 matches the most recent day (highest weight)
    # and k=n-1 matches the oldest day (lowest weight)
    weight = float(n - k) / sum_of_weights
    h_wma.set_value_at_time(k, weight)

# ---- 4. Compute Outputs using your LTISystem Machinery ----
sys_uma = LTISystem(h_uma)
sys_wma = LTISystem(h_wma)

y_uma = sys_uma.output(x_signal)
y_wma = sys_wma.output(x_signal)

# ---- 5. Extract Valid Outputs ----
# Valid moving windows begin when the filter is completely full, at index n-1
uma = []
wma = []
for time_idx in range(n - 1, m):
    uma.append(y_uma.get_value_at_time(time_idx))
    wma.append(y_wma.get_value_at_time(time_idx))

# Print the two moving averages exactly as required
print("Unweighted Moving Averages: " + ", ".join(f"{num:.2f}" for num in uma))
print("Weighted Moving Averages:   " + ", ".join(f"{num:.2f}" for num in wma))
