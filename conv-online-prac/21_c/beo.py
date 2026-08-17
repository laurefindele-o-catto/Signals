# Assuming your DiscreteSignal (or Signal) and LTISystem classes are defined above

# Input for first polynomial
d1 = int(input("Degree of the first Polynomial: "))
poly1 = list(map(int, input("Coefficients: ").split()))

# Input for second polynomial
d2 = int(input("Degree of the second Polynomial: "))
poly2 = list(map(int, input("Coefficients: ").split()))


# ---- 1. Map Polynomial 1 to the Input Signal x[n] ----
x_signal = DiscreteSignal(0, d1)
for i, coef in enumerate(poly1):
    x_signal.set_value_at_time(i, float(coef))

# ---- 2. Map Polynomial 2 to the Impulse Response h[n] ----
h_signal = DiscreteSignal(0, d2)
for i, coef in enumerate(poly2):
    h_signal.set_value_at_time(i, float(coef))


# ---- 3. Multiply the polynomials using Discrete-Time Convolution ----
system = LTISystem(h_signal)
y_signal = system.output(x_signal)


# ---- 4. Extract and Print the Result ----
output_degree = d1 + d2
output_coefficients = []

# Gather coefficients from index 0 to output_degree
for i in range(output_degree + 1):
    # Round to integer as polynomial coefficients are given as integers
    val = round(y_signal.get_value_at_time(i))
    output_coefficients.append(val)

# Print output in the required format
print(f"Degree of the Polynomial: {output_degree}")
print("Coefficients: " + " ".join(map(str, output_coefficients)))
