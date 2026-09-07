"""
transforms.py  --  YOUR CODE GOES HERE.

The shared transform core used by BOTH tasks. Write it once; bigmul.py
(Task A) and image_conv.py (Task B) import it.

Nothing in this file may call numpy.fft, scipy.fft, numpy.convolve,
scipy.signal, or any other library routine that performs a Fourier
transform, a convolution or a correlation for you. NumPy is for array
arithmetic only.

A quick self-test you should run before touching either application:

    import numpy as np
    from transforms import DFTAnalyzer, FFTTransformer
    x = np.random.randn(64) + 1j * np.random.randn(64)
    d, f = DFTAnalyzer(), FFTTransformer()
    assert np.max(np.abs(d.transform(x) - f.transform(x))) < 1e-9
    assert np.max(np.abs(d.inverse(d.transform(x)) - x)) < 1e-9
"""

import numpy as np
import math


def next_power_of_two(n):
    """
    Return the smallest power of two that is >= ``n`` (and at least 1).

    Both tasks need this to choose a transform length for the radix-2 FFT.
    """
    # TODO: implement this function
    # 2^x >= n => x >= log2(n)
    
    x = 2 ** (math.ceil(math.log2(max(1, n))))
    
    return x

def bit_reverse_array(x):
    """
    assume length = perfect power of 2
    bit reversed indexed x
    """
    x = np.asarray(x)
    N = len(x)
    num_bits = int(np.log2(N))
    
    indices = np.arange(N, dtype=np.uint32)
    reversed_indices = np.zeros(N, dtype=np.uint32)
    
    for i in range(num_bits):
        #extract ith bit and shifts it to its reverse
        bit = (indices >> i) & 1
        reversed_indices |= (bit << (num_bits - i -1))
        
    return x[reversed_indices]
    

class DFTAnalyzer:
    """
    The Discrete Fourier Transform, computed straight from its definition.

        Analysis:   X[k] = sum_{n=0}^{N-1} x[n] * exp(-2j*pi*k*n/N)
        Synthesis:  x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(+2j*pi*k*n/N)

    How you write it is up to you -- a literal double loop, a precomputed
    table of twiddle factors indexed by (k*n) % N, or a NumPy expression --
    as long as it computes these sums directly and is not secretly an FFT.
    """

    name = "dft"

    def transform(self, x):
        """
        Forward DFT.

        Parameters
        ----------
        x : 1D array_like, length N (real or complex)

        Returns
        -------
        numpy.ndarray of complex128, shape (N,)
        """
        # TODO: implement this method
        coeffs = []
        N = len(x)
        for k in range(0, N):
            Xk = 0.0 + 0.0j
            
            for n in range(0, N):
                Xk += x[n]*np.exp(-2j*np.pi*k*n/N)
            coeffs.append(Xk)
            
        return np.array(coeffs, dtype=np.complex128)

    def inverse(self, spectrum):
        """
        Inverse DFT, including the 1/N factor.

        Parameters
        ----------
        spectrum : 1D array_like, length N (complex)

        Returns
        -------
        numpy.ndarray of complex128, shape (N,)
            Do NOT discard the imaginary part here -- the caller decides when
            it is safe to take .real.
        """
        # TODO: implement this method
        #x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(+2j*pi*k*n/N)
        
        N = len(spectrum)
        
        x = []
        
        for n in range(0, N):
            x_n = 0.0+0.0j
            for k in range(0, N):
                x_n += spectrum[k] * np.exp(+2j*np.pi*k*n/N) 
            x.append(x_n/N)
            
        return np.array(x, dtype=np.complex128)


class FFTTransformer(DFTAnalyzer):
    """
    Radix-2 decimation-in-time (Cooley-Tukey) FFT, in O(N log N).

    It inherits from DFTAnalyzer so that both applications can treat the two
    interchangeably: they call ``engine.transform(...)`` and
    ``engine.inverse(...)`` without caring which engine they hold.

    Requirements:
      * Recursive or iterative (with bit-reversal permutation) -- your choice.
      * N must be a power of two; raise ValueError for any other length.
        The caller is responsible for zero-padding up to next_power_of_two.
      * The inverse must reuse the same butterfly machinery (conjugated
        twiddles, or conjugate-transform-conjugate), not a second copy of it.
      * Twiddle factors for a stage are computed once per stage, never once
        per butterfly.
    """

    name = "fft"

    def transform(self, x):
        """Forward FFT. Same contract as DFTAnalyzer.transform."""
        # TODO: implement this method
        #input hocche x
        
        N = len(x)
        x = np.asarray(x, dtype=np.complex128)
        
        if N == 0 or (N & (N - 1)) != 0:
            raise ValueError(f"FFT length must be a perfect power of two, got N={N}")
            
        x_bit_reversed = bit_reverse_array(x)

        
        for s in range(1, int(np.log2(N))+1):
            M = 2 ** s
            
            k_arr = np.arange(M // 2)
            W_arr = np.exp(-2j * np.pi * k_arr / M)
            
            for l in range(0, N, M):
                idx_1 = l + k_arr
                idx_2 = idx_1 + M//2
                
                g = x_bit_reversed[idx_1]
                h = W_arr * x_bit_reversed[idx_2]
                
                x_bit_reversed[idx_1] = g + h
                x_bit_reversed[idx_2] = g - h
                
                
        # for s in range(1, int(np.log2(N))+1):
        #     M = 2 ** s
        #     W_M = np.exp(np.pi * (-2) * 1j*(1/M))
            
        #     for l in range(0, N-M +1 , M):
        #         twiddle = 1 + 0j
        #         for k in range(0, (M//2)):
        #             g = x_bit_reversed[l + k]
        #             h = twiddle * x_bit_reversed[l + k + (M//2)]
        #             x_bit_reversed[l+k]= g+ h
        #             x_bit_reversed[l+k + (M//2)]= g-h
        #             twiddle = twiddle * W_M
                
        return x_bit_reversed
        
    def inverse(self, spectrum):
        """Inverse FFT, including the 1/N factor."""
        # TODO: implement this method
        N = len(spectrum)
        return np.conjugate(self.transform(np.conjugate(spectrum))) / N


# ---------------------------------------------------------------------------
# BONUS (optional) -- arbitrary-length FFT.
#
# Delete this class if you are not attempting the bonus. If you do attempt it,
# run both tasks with --engine arbitrary and leave those output directories in
# your submission as the evidence.
# ---------------------------------------------------------------------------
class ArbitraryLengthFFT(FFTTransformer):
    """
    Bonus: an O(N log N) transform for ANY length N, not just powers of two.

    Bluestein's chirp-z algorithm is the usual route: rewrite the DFT as a
    convolution of two chirp sequences, and evaluate that convolution with a
    radix-2 FFT of length >= 2N-1. A mixed-radix Cooley-Tukey that factorises
    N is equally acceptable.

    With this engine, Task A no longer has to pad the digit arrays up to a
    power of two, and Task B no longer has to pad the image up to one.
    """

    name = "arbitrary"
    
    def __init__(self, fft_engine=None):
        self.fft_engine = fft_engine if fft_engine is not None else FFTTransformer()


    def transform(self, x):
        # TODO (bonus): implement this method
        x = np.asarray(x, dtype=np.complex128)
        N = len(x)
        
        if N > 0 and (N & (N - 1)) == 0:
            return self.fft_engine.transform(x)
        
        n = np.arange(N)
        chirp_n = np.exp(-1j * np.pi * (n**2) / N)
        a = x*chirp_n
        
        pad_length = next_power_of_two(2*N-1)
        a_padded = np.zeros(pad_length, dtype=np.complex128)
        a_padded[:N] = a

        b_padded = np.zeros(pad_length, dtype=np.complex128)
        b_padded[:N] = np.conj(chirp_n) 
        #pos indices
        
        for n_idx in range(1, N):
            target_index = pad_length - n_idx
            b_padded[target_index] = np.conj(chirp_n[n_idx])
            
        spectrum_a = self.fft_engine.transform(a_padded)
        spectrum_b = self.fft_engine.transform(b_padded)
        
        product_spectrum = spectrum_a * spectrum_b
        conv = self.fft_engine.inverse(product_spectrum)
        
        X = conv[:N] * chirp_n
        #takes first n
        
        return X


    def inverse(self, spectrum):
        # TODO (bonus): implement this method
        spectrum = np.asarray(spectrum, dtype=np.complex128)
        N = len(spectrum)
        
        return np.conjugate(self.transform(np.conjugate(spectrum))) / N
