import numpy as np

from svg_utils import load_svg_path
from epicycle_animation import save_outputs


class FourierEpicycles:
    def __init__(self, t, signal, n_harmonics):
        """
        Step 1: Store the sampled signal and set up everything the other
        methods will need.

        Parameters
        ----------
        t : 1D numpy array, shape (M,)
            Uniformly spaced sample times covering ONE FULL PERIOD of the
            signal, as a *closed* interval: t[0] == 0 and t[-1] == T (the
            period). This is exactly what svg_utils.load_svg_path(...)
            returns.
        signal : 1D complex numpy array, shape (M,)
            signal[i] = f(t[i]) = x(t[i]) + 1j * y(t[i]). Periodic, so
            signal[-1] == signal[0].
        n_harmonics : int (call it N)
            The series will use every integer harmonic n with
            -N <= n <= N (i.e. 2N+1 terms in total -- do not forget the
            negative harmonics).

        You must set at least the following attributes, since the rest of
        this class (and the provided plotting/animation code) expects
        them to exist:
            self.t, self.signal, self.N
            self.T      -- the period (a float)
            self.omega  -- the fundamental angular frequency, 2*pi/T
            self.coeffs -- an (initially empty) dict that will map
                           n -> c_n once calculate_all_coefficients() has
                           been called
        """
        # TODO: implement this method
        self.t = t
        self.signal = signal
        self.N = n_harmonics
        self.T = t[-1]
        self.omega = (2*np.pi)/self.T
        self.coeffs = {}

    def calculate_cn(self, n):
        """
        Step 2: Compute a single complex Fourier coefficient c_n using
        numerical integration (np.trapezoid) over the stored samples
        self.t, self.signal.

            c_n = (1/T) * integral_0^T  f(t) * exp(-j*n*omega*t)  dt

        n may be zero, positive, or negative.
        """
        # TODO: implement this method
        exp_func = np.exp(-1j*n*self.t*self.omega)
        
        c_n = (1/self.T) * np.trapezoid(self.signal*exp_func, self.t)
        
        return c_n

    def calculate_all_coefficients(self):
        """
        Step 3: Populate self.coeffs with c_n for every harmonic
        n = -N, ..., -1, 0, 1, ..., N by repeatedly calling calculate_cn(n).
        """
        # TODO: implement this method
        for n in range(-self.N, self.N+1):
            self.coeffs[n] = self.calculate_cn(n)
        

    def approximate(self, t):
        """
        Step 4: Reconstruct (an approximation of) the signal at time(s) t
        from the coefficients already stored in self.coeffs:

            f_hat(t) = sum_{n=-N}^{N} c_n * exp(j*n*omega*t)

        t may be a single number or a numpy array of times -- your
        implementation must support both, since the provided
        plotting/animation code calls this both ways.
        """
        # TODO: implement this method
        f_hat = 0.0 + 0.0j
        
        for n in range(-self.N, self.N+1):
            f_hat += self.coeffs[n] * np.exp(1j*n*self.omega*t)
            
        return f_hat
    
    def calculate_total_energy(self):
        self.calculate_all_coefficients()
        total_energy = 0.0
        
        for n in range(-self.N, self.N+1):
            total_energy += np.abs(self.coeffs[n]) ** 2
            #np.angle gives the angle in radian, use deg = True for degree
            
        return total_energy
    
    def prune_harmonics_by_energy(self, r):
        total_energy = self.calculate_total_energy()
        req_energy = total_energy * r
        
        sorted_ns = sorted(self.coeffs.keys(), key = lambda n:abs(self.coeffs[n])**2, reverse = True)
        
        retained = []
        energy_sum = 0.0
        
        for n in sorted_ns:
            part = np.abs(self.coeffs[n])**2
            energy_sum += part
            retained.append(n)
            if energy_sum >= req_energy:
                break
            
        retained_set = set(retained)
        for n in self.coeffs:
            if n not in retained_set:
                self.coeffs[n] = 0.0 + 0.0j

        return len(retained), energy_sum / total_energy
    
    
    def evaluate_reconstruction_error(self):
        f_hat = self.approximate(self.t)
        mse = 0.0
        for i in range(0, len(self.t)):
            part = np.abs(self.signal[i] - f_hat[i])**2
            mse += part
        
        mse = mse/len(self.t)
        
        #or
        #mse = np.mean(np.abs(self.signal - f_hat) ** 2)
        
        return mse
        

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Usage: python3 assignment.py <path_to_svg> [n_harmonics] [comparison_png_path] [gif_path]
    if len(sys.argv) < 2:
        print("Usage: python3 assignment.py <path_to_svg> [n_harmonics] [comparison_png_path] [gif_path]")
        print("Example: python3 assignment.py svgs/heart.svg 150 heart_comparison.png heart_epicycles.gif")
        sys.exit(1)

    svg_path = sys.argv[1]
    N_HARMONICS = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    stem = Path(svg_path).stem
    comparison_path = sys.argv[3] if len(sys.argv) > 3 else f"{stem}_comparison.png"
    gif_path = sys.argv[4] if len(sys.argv) > 4 else f"{stem}_epicycles.gif"

    t, z = load_svg_path(svg_path, num_points=1000)
    fs = FourierEpicycles(t, z, n_harmonics=N_HARMONICS)
    fs.calculate_all_coefficients()
    
    print("Target Ratio | Harmonics Retained | Actual Energy Ratio | MSE")
    print("-" * 66)

    for r in [0.96, 0.98, 0.99, 1.00]:
        retained_count, actual_ratio = fs.prune_harmonics_by_energy(r)
        mse = fs.evaluate_reconstruction_error()
        print(f"{r:.2f}         | {retained_count:19d} | {actual_ratio:.4f}               | {mse:.6f}")
        save_outputs(fs, z, f"heart_pruned_{r}.png", gif_path, num_frames=240)

    save_outputs(fs, z, comparison_path, gif_path, num_frames=240)


#python fs_redrawer.py svgs/heart.svg 150 outputs/heart_comparison.png outputs/heart_epicycles.gif
#python fs_redrawer.py svgs/circle.svg 150 outputs/circle_comparison.png outputs/circle_epicycles.gif
#python fs_redrawer.py svgs/star.svg 150 outputs/star_comparison.png outputs/star_epicycles.gif
#python fs_redrawer.py svgs/infinity.svg 150 outputs/infinity_comparison.png outputs/infinity_epicycles.gif

#python fs_redrawer.py svgs/flower.svg 150 outputs/flower_comparison.png outputs/flower_epicycles.gif
