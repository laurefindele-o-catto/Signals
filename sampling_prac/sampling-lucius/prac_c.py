"""Where Does the Tone Land? Aliasing of an arbitrary frequency.

Complete the two functions marked TODO. Do not modify main().
Run with:  python fold.py
"""

import numpy as np


def apparent_frequency(f, fs):
    """Frequency (in Hz) at which a tone of frequency f appears after sampling at fs.

    The result always lies in [0, fs/2]. Unlike the alias problem, f may be
    ANY positive frequency, including f > fs/2.
    """
    f_wrapped = f%fs
    
    if f_wrapped > fs/2:
        return fs - f_wrapped
    
    return f_wrapped
    raise NotImplementedError


def measured_peak_frequency(f, fs, duration):
    """Frequency of the strongest DFT bin of the sampled cosine.

    Samples cos(2*pi*f*t) at rate fs using t = n/fs for n = 0 ... N-1,
    N = int(duration * fs). Takes the real DFT (np.fft.rfft) and returns the
    frequency (in Hz) of the bin with the largest magnitude.

    Returns a single float.
    """
    N = int(duration * fs)
    t = np.arange(N)
    t = t/fs
    
    cos_ft = np.cos(2*np.pi*f*t)
    
    real = np.abs(np.fft.rfft(cos_ft))
    mag_index = np.argmax(real)
    freqs = np.fft.rfftfreq(N, 1/fs)
    
    
    # delta_f = fs / N
    # frequency = mag_index * delta_f
    # return float(frequency)
    
    #first frequency in a bin is equal to del f = fs/N
    
    
    return float(freqs[mag_index])
    raise NotImplementedError


# --------------------------------------------------------------------------
# Everything below is provided. Do not modify.
# --------------------------------------------------------------------------

TEST_CASES = [
    (300, 1000),
    (700, 1000),
    (1300, 1000),
    (2200, 1000),
    (5000, 8000),
    (9000, 8000),
    (1000, 1000),
]

DURATION = 0.1
TOLERANCE = 1e-6


def main():
    print(f"{'f (Hz)':>8} {'fs (Hz)':>8} {'predicted':>11} {'measured':>10}   result")
    print("-" * 56)

    failures = 0
    for f, fs in TEST_CASES:
        predicted = apparent_frequency(f, fs)
        measured = measured_peak_frequency(f, fs, DURATION)

        ok = abs(predicted - measured) < TOLERANCE
        failures += not ok
        print(f"{f:>8} {fs:>8} {predicted:>11.1f} {measured:>10.1f}   "
              f"{'match' if ok else 'MISMATCH'}")

    print("-" * 56)
    if failures:
        print(f"{failures} of {len(TEST_CASES)} case(s) did not match.")
    else:
        print("Every prediction matches the measured peak.")


if __name__ == "__main__":
    main()
