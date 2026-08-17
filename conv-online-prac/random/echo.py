# Assuming DiscreteSignal and LTISystem are already available

def main_problem_3():
    print("\n=== Problem 3: Echo Cancellation (Deconvolution) ===")
    obs_start, obs_end = 0, 4

    # 1. Build clear original input signal x[n] = [10.0, -4.0, 2.0] for n=0...2
    x = DiscreteSignal(0, 2)
    x.set_value_at_time(0, 10.0)
    x.set_value_at_time(1, -4.0)
    x.set_value_at_time(2, 2.0)

    # 2. Channel echo impulse response: h1[n] = delta[n] + 0.5*delta[n-1]
    h1 = DiscreteSignal(0, 1)
    h1.set_value_at_time(0, 1.0)
    h1.set_value_at_time(1, 0.5)

    # 3. Form finite clearing inverse filter: h2[n] = (-0.5)^n for n = 0...8
    h2 = DiscreteSignal(0, 8)
    for n in range(0, 9):
        h2.set_value_at_time(n, (-0.5) ** n)

    # Build LTI processing components
    echo_channel = LTISystem(h1)
    clearing_filter = LTISystem(h2)

    # 4. Cascade execution: Clear signal -> Echo System -> Clearing Inverse System
    v = echo_channel.output(x)          # Distorted signal
    y_recovered = clearing_filter.output(v)  # Recovered clear signal

    # 5. Output comparison results over observation window n = 0...4
    print("n  |  Original x[n]  |  Recovered y[n]")
    print("-" * 38)
    diffs = []
    for n in range(obs_start, obs_end + 1):
        orig_val = x.get_value_at_time(n)
        recv_val = y_recovered.get_value_at_time(n)
        print(f"{n:2d} | {orig_val:14.4f} | {recv_val:14.4f}")
        diffs.append(abs(orig_val - recv_val))

    max_diff = max(diffs)
    print("-" * 38)
    print(f"Maximum cleanup discrepancy error: {max_diff:.6e}")
    print("Echo suppression successful:", max_diff < 1e-5)
