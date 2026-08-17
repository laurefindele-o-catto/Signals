import numpy as np
import matplotlib.pyplot as plt

def readable_time_ticks(time_values, max_labels=18):
    if len(time_values) <= max_labels:
        return time_values

    step = int(np.ceil(len(time_values) / max_labels))
    ticks = list(time_values[::step])

    if ticks[-1] != time_values[-1]:
        ticks.append(time_values[-1])

    return ticks


class Signal:
    """Finite discrete-time signal with integer indices."""

    # Arguments: start_time and end_time are integers with start_time <= end_time.
    # Output: None; initialize start_time, end_time, and zero-valued stored samples.
    # Example: Signal(-2, 3) represents samples for n = -2, -1, ..., 3.
    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time
        self.values = np.zeros(end_time-start_time+1)

    # Arguments: none.
    # Returns: int, the number of stored samples in this finite signal.
    # Example: len(Signal(-2, 3)) should be 6.
    def __len__(self):
        return (self.end_time - self.start_time + 1);

    # Arguments: none.
    # Returns: range of integer time indices covered by the signal.
    # Example: Signal(-1, 2).times() should cover -1, 0, 1, 2.
    def times(self):
        return range(self.start_time, self.end_time+1)

    # Arguments: t is an integer time index.
    # Returns: float, the signal value at t; return 0.0 if t is outside the range.
    # Example: if x[2] = 5, then x.get_value_at_time(2) should return 5.0.
    def get_value_at_time(self, t):
        out_of_range = (t < self.start_time) | (t>self.end_time)
        
        if out_of_range:
            return 0.0
        
        return self.values[t-self.start_time]

    # Arguments: t is an integer time index, value is the sample value to store.
    # Output: None; update the stored sample at t, or raise an error if t is outside.
    # Example: x.set_value_at_time(2, 5) makes x[2] equal to 5.
    def set_value_at_time(self, t, value):
        out_of_range = (t < self.start_time) | (t>self.end_time)
                
        if out_of_range:
            raise AssertionError("t is out of range")
        
        self.values[t-self.start_time] = value

    # Arguments: k is an integer shift amount.
    # Returns: Signal, a copy with indices shifted so y[n] = x[n - k].
    # Example: shifting a signal over 0..2 by 3 returns a signal over 3..5.
    def shift(self, k):
        newSignal = Signal(self.start_time + k, self.end_time + k)
        newSignal.values = self.values.copy()
        return newSignal

    # Arguments: other is another Signal.
    # Returns: Signal over the combined range with sample-wise sums.
    # Example: if x[0] = 2 and z[0] = 3, then x.add(z)[0] should be 5.
    def add(self, other):
        newStart = min(self.start_time, other.start_time)
        newEnd = max(self.end_time, other.end_time)
        
        newSignal = Signal(newStart, newEnd)
        
        for i in range(newStart, newEnd+1):
            val = self.get_value_at_time(i) + other.get_value_at_time(i)
            newSignal.set_value_at_time(i, val)
        
        return newSignal

    # Arguments: scalar is a number used to multiply every stored sample.
    # Returns: Signal with the same time range and scaled sample values.
    # Example: if x[1] = 4, then x.multiply(0.5)[1] should be 2.
    def multiply(self, scalar):
        newSignal = Signal(self.start_time, self.end_time)
        newSignal.values = scalar * (self.values)
        return newSignal

    # Arguments: tolerance is the threshold below which values are treated as zero.
    # Returns: list of (time_index, value) tuples for samples with abs(value) > tolerance.
    # Example: values [1, 0, 3] starting at n = 0 should return [(0, 1), (2, 3)].
    def nonzero_samples(self, tolerance=1e-12):
        idx = np.where(np.abs(self.values) > tolerance)[0]
        times = idx + self.start_time
        vals = self.values[idx]
        
        return list(zip(times, vals))

    def plot(self, title, save_path=None, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        time_values = list(self.times())
        markerline, stemlines, baseline = ax.stem(time_values, self.values)
        markerline.set_markersize(6)
        baseline.set_color("black")
        baseline.set_linewidth(1)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("n")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.35)
        ax.set_xticks(readable_time_ticks(time_values))
        ax.tick_params(axis="x", labelsize=9)

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)

        return ax
    
    def flip(self):
        flipped = Signal(self.start_time, self.end_time)
        flipped.values = -self.values
        return flipped


class LTISystem:
    """Discrete-time LTI system described by a finite impulse response."""

    # Arguments: impulse_response is a Signal representing h[n].
    # Output: None; store the impulse response that defines this LTI system.
    # Example: LTISystem(impulse_identity()) creates the identity system.
    def __init__(self, impulse_response):
        self.impulse_response = impulse_response

    # Arguments: input_signal is a Signal representing x[n].
    # Returns: (start, end) tuple for the convolution output y[n].
    # Example: x over 0..4 and h over 0..2 produce output range (0, 6).
    def output_range(self, input_signal):
        start = input_signal.start_time + self.impulse_response.start_time
        end = input_signal.end_time + self.impulse_response.end_time
        
        return (start, end)

    # Arguments: input_signal is a Signal representing x[n].
    # Returns: list of (k, component_signal) for each nonzero input sample x[k].
    # Example: x[2] = 3 contributes the component 3*h[n - 2].
    def get_response_components(self, input_signal):
        components = []
        for k, val in input_signal.nonzero_samples():
            shiftedSignal = self.impulse_response.shift(k)
            componentSignal = shiftedSignal.multiply(val)
            components.append((k, componentSignal))
        return components

    # Arguments: input_signal is a Signal representing x[n].
    # Returns: Signal y[n], computed by adding all response components.
    # Example: for the identity impulse, the output should match the input signal.
    def output_by_superposition(self, input_signal):
        start, end = self.output_range(input_signal)
        y = Signal(start, end)
        
        componentSignals = self.get_response_components(input_signal)
        
        for k, component_signal in componentSignals:
            y = y.add(component_signal)
            
        return y

    # Arguments: input_signal is a Signal and n is one output time index.
    # Returns: list of (k, x_k, h_n_minus_k, product) nonzero contribution tuples.
    # Example: a term may look like (2, 3.0, 0.5, 1.5) for x[2]h[n - 2].
    def get_contributions_at_time(self, input_signal, n):
        #too loazy to calculate k, just start ar end use kortesi
        contributions = []
        for k in range(input_signal.start_time, input_signal.end_time+1):
            x_k = input_signal.get_value_at_time(k)
            h_n_minus_k = self.impulse_response.get_value_at_time(n - k)
            product = x_k * h_n_minus_k
            
            if(abs(product)> 1e-12):
                contributions.append((k, x_k, h_n_minus_k, product))
            
        return contributions

    # Arguments: input_signal is a Signal and n is one output time index.
    # Returns: float, the convolution-sum value y[n].
    # Example: output_at_time(x, 4) returns the scalar sample y[4].
    def output_at_time(self, input_signal, n):
        contributions = self.get_contributions_at_time(input_signal, n)
        
        products = [item[3] for item in contributions]
        
        conv_sum = float(np.sum(products))
        return conv_sum
            

    # Arguments: input_signal is a Signal representing x[n].
    # Returns: Signal containing every output sample y[n].
    # Example: system.output(x) returns the full convolution result x[n] * h[n].
    def output(self, input_signal):
        start, end = self.output_range(input_signal)
        y = Signal(start, end)
        
        for i in range(start, end+1):
            output_i = self.output_at_time(input_signal, i)
            y.set_value_at_time(i, output_i)
            
        return y



if __name__ == "__main__":
    INF = 10

    x = Signal(0, 10)
    x.set_value_at_time(0, 1)
    x.set_value_at_time(2, -1)
    x.plot("Input x(n)")

    h1 = Signal(0, 10)
    h1.set_value_at_time(0, 1)

    h2 = Signal(0, 10)
    h2.set_value_at_time(1, 0.5)

    h3 = Signal(0, 10)
    h3.set_value_at_time(0, 1)
    h3.set_value_at_time(1, 1)

    sys1 = LTISystem(h1)
    sys2 = LTISystem(h2)
    sys3 = LTISystem(h3)
    
    # Todo: Determine output block by block

    xh1 = sys1.output(x)
    xh2 = sys2.output(x)
    x_mid = xh1.add(xh2)
    y_final_1 = sys3.output(x_mid) 
    y_final_1.plot("Output via block-by-block system")

    # Todo: Determine h_combined
    
    h12 = h1.add(h2)
    h_combined = sys3.output(h12)
    sys_combined = LTISystem(h_combined)

    y_final_2 = sys_combined.output(x)
    y_final_2.plot("Output via combined impulse response")

    print("Outputs are equal:",
          np.allclose(y_final_1.values, y_final_2.values))
    
    plt.show()
