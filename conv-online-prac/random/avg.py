# Assuming DiscreteSignal and LTISystem are already available

def compute_analytical_cascade_response(input_signal: DiscreteSignal) -> DiscreteSignal:
    """
    Computes y[n] = 0.25*x[n] + 0.5*x[n-1] + 0.25*x[n-2]
    using basic sample loops to simulate the analytical math definition.
    """
    # Accommodate the expansion from the delay of 2 samples
    start = input_signal.start_time
    end = input_signal.end_time + 2
    
    analytical_sig = DiscreteSignal(start, end)
    
    for n in range(start, end + 1):
        x_n   = input_signal.get_value_at_time(n)
        x_n1  = input_signal.get_value_at_time(n - 1)
        x_n2  = input_signal.get_value_at_time(n - 2)
        
        # Calculate combined weighted output sample
        val = 0.25 * x_n + 0.5 * x_n1 + 0.25 * x_n2
        analytical_sig.set_value_at_time(n, val)
        
    return analytical_sig


def main_problem_1():
    print("=== Problem 1: Moving Average Cascade ===")
    obs_start, obs_end = 0, 6

    # 1. Build input signal x[n] = [4.0, 2.0, -2.0, 0.0, 4.0] for n=0...4
    x = DiscreteSignal(0, 4)
    x_vals = [4.0, 2.0, -2.0, 0.0, 4.0]
    for i, v in enumerate(x_vals):
        x.set_value_at_time(0 + i, v)

    # 2. Build filter stages h1[n] and h2[n]
    h1 = DiscreteSignal(0, 1)
    h1.set_value_at_time(0, 0.5)
    h1.set_value_at_time(1, 0.5)
    
    h2 = DiscreteSignal(0, 1)
    h2.set_value_at_time(0, 0.5)
    h2.set_value_at_time(1, 0.5)

    # Wrap them as LTI System instances
    stage1 = LTISystem(h1)
    stage2 = LTISystem(h2)

    # 3. Cascaded execution: Pass x sequentially through stage1 then stage2
    v = stage1.output(x)
    y_experimental = stage2.output(v)

    # 4. Generate analytical check response
    y_analytic = compute_analytical_cascade_response(x)

    # 5. Measure absolute deviation across the observation range
    diffs = []
    for n in range(obs_start, obs_end + 1):
        d = abs(y_experimental.get_value_at_time(n) - y_analytic.get_value_at_time(n))
        diffs.append(d)
    max_diff = max(diffs)

    print(f"Experimental response at n=2: {y_experimental.get_value_at_time(2)}")
    print(f"Analytical response at n=2:   {y_analytic.get_value_at_time(2)}")
    print(f"Maximum absolute mismatch:    {max_diff}")
    print("Verification passed:", max_diff < 1e-9)
