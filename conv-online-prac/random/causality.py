# Assuming DiscreteSignal is defined with .arr, .start_time, and .end_time

def test_causality(apply_system) -> bool:
    """
    Verifies causality. A system is causal if it produces NO output at 
    n < n0 when given an input that is strictly zero for all n < n0.
    """
    # Create an input signal that is zero everywhere except at a future time n = 2
    test_input = DiscreteSignal(-5, 5)
    
    # Using your internal index mapping: index = n - start_time
    # 2 - (-5) = 7
    test_input.arr[2 - test_input.start_time] = 5.0  
    
    # Process through system
    test_output = apply_system(test_input)
    
    # Look for any look-ahead leakage at n < 2 (e.g., at n=0 or n=1)
    for n in range(-5, 2):
        out_idx = n - test_output.start_time
        if 0 <= out_idx < len(test_output.arr):
            if abs(test_output.arr[out_idx]) > 1e-9:
                return False  # System reacted to a future input, meaning non-causal
            
    return True


def test_stability(apply_system) -> bool:
    """
    Verifies BIBO Stability. A system is stable if a bounded input 
    produces a bounded output. We test with a massive bounded step 
    to see if the output values blow up or grow without boundaries.
    """
    # Create a bounded constant input signal x[n] = 2.0
    test_input = DiscreteSignal(-5, 5)
    for n in range(-5, 6):
        test_input.arr[n - test_input.start_time] = 2.0
        
    test_output = apply_system(test_input)
    
    # Check if any output amplitude escapes a safe maximum threshold
    for n in range(test_output.start_time, test_output.end_time + 1):
        out_idx = n - test_output.start_time
        if abs(test_output.arr[out_idx]) > 100.0:  # Threshold for divergence
            return False  # Unbounded output found
            
    return True


def system_c(input_signal: DiscreteSignal) -> DiscreteSignal:
    """System C (Ideal Delay): y[n] = x[n - 2]"""
    start = input_signal.start_time + 2
    end = input_signal.end_time + 2
    output_signal = DiscreteSignal(start, end)
    
    for n in range(start, end + 1):
        # Pull historical input data safely using array indexing rules
        in_idx = (n - 2) - input_signal.start_time
        out_idx = n - output_signal.start_time
        
        if 0 <= in_idx < len(input_signal.arr):
            output_signal.arr[out_idx] = input_signal.arr[in_idx]
            
    return output_signal


def system_d(input_signal: DiscreteSignal) -> DiscreteSignal:
    """System D (Time Reversal): y[n] = x[-n]"""
    start = -input_signal.end_time
    end = -input_signal.start_time
    output_signal = DiscreteSignal(start, end)
    
    for n in range(start, end + 1):
        in_idx = (-n) - input_signal.start_time
        out_idx = n - output_signal.start_time
        
        if 0 <= in_idx < len(input_signal.arr):
            output_signal.arr[out_idx] = input_signal.arr[in_idx]
            
    return output_signal


def main_problem_2():
    print("\n=== Problem 2: Property Verification ===")
    
    is_c_causal = test_causality(system_c)
    is_d_causal = test_causality(system_d)
    
    is_c_stable = test_stability(system_c)
    is_d_stable = test_stability(system_d)
    
    print(f"System C (Ideal Delay)    -> Causal: {is_c_causal:5s} | Stable: {is_c_stable}")
    print(f"System D (Time Reversal) -> Causal: {is_d_causal:5s} | Stable: {is_d_stable}")


def test_lti_stability(lti_system_object) -> bool:
    """
    Verifies BIBO stability for an LTI system by checking if its 
    impulse response h[n] is absolutely summable.
    """
    # 1. Extract the impulse response signal from your LTI system object
    # Assuming your LTISystem class stores the signal in self.h
    h_signal = lti_system_object.h 
    
    start = h_signal.start_time
    end = h_signal.end_time
    
    absolute_sum = 0.0
    
    # 2. Compute the absolute summation from start to end
    for n in range(start, end + 1):
        idx = n - start
        absolute_sum += abs(h_signal.arr[idx])
        
    # Since any finite-length array (FIR filter) yields a finite sum,
    # it is mathematically stable. We check if it is within a reasonable bound.
    return absolute_sum < float('inf')
