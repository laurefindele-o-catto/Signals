"""The Ramp Between the Steps: linear-interpolation gain.

Complete the two functions marked TODO. Do not modify anything below
the divider. Run with:  python lininterp.py
"""

import numpy as np


def sinc(x):
    """Normalised sinc, sin(pi x) / (pi x), with sinc(0) = 1."""
    return np.sinc(x)


def linear_interp_gain(f, fs):
    """Predicted gain of linear interpolation at frequency f, sampling at rate fs.

    gain = sinc(f / fs) ** 2

    """
    return np.sinc(f/fs) ** 2
    raise NotImplementedError


def measured_linear_interp_gain(f, fs, upsample, duration):
    """Gain of linear interpolation, measured rather than predicted.

    Samples cos(2*pi*f*t) at rate fs at times t = n/fs for n = 0 ... N
    (that is N+1 samples, endpoint included) with N = int(duration * fs).
    Then reconstructs the signal by linear interpolation (np.interp) onto a
    fine grid t_fine = m / (upsample * fs) for m = 0 ... N*upsample - 1, and
    returns the amplitude of the f component of the result, measured with
    tone_amplitude(y, upsample * fs, f).

    The input amplitude is 1, so this amplitude is the gain.
    """
    N = int(duration*fs)
    t = np.arange(N+1)
    t = t/fs
    
    samples = np.cos(2*np.pi*f*t)
    
    m = np.arange(N*upsample)
    t_fine = m/ (upsample * fs)
    
    y = np.interp(t_fine, t, samples)
    
    
    return tone_amplitude(y, upsample*fs, f)
    raise NotImplementedError


# --------------------------------------------------------------------------
# Everything below is provided. Do not modify.
# --------------------------------------------------------------------------

def tone_amplitude(x, fs_fine, f):
    """Amplitude of the component of x at frequency f, via the DFT.

    x is real and sampled at rate fs_fine. Assumes f falls on a DFT bin,
    which the parameters in main() guarantee.
    """
    n = len(x)
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1 / fs_fine)
    bin_index = int(np.argmin(np.abs(freqs - f)))
    return 2 * np.abs(spectrum[bin_index]) / n


SAMPLE_RATE = 1000
UPSAMPLE = 100
DURATION = 0.1
TOLERANCE = 1e-3

TEST_FREQS = [50, 100, 200, 300, 450]


def main():
    print(f"{'f (Hz)':>8} {'f/fs':>7} {'predicted':>11} {'measured':>10} "
          f"{'|error|':>10} {'droop':>10}")
    print("-" * 60)

    failures = 0
    for f in TEST_FREQS:
        predicted = linear_interp_gain(f, SAMPLE_RATE)
        measured = measured_linear_interp_gain(f, SAMPLE_RATE, UPSAMPLE, DURATION)
        error = abs(predicted - measured)

        failures += error >= TOLERANCE
        droop_db = 20 * np.log10(predicted)
        print(f"{f:>8} {f / SAMPLE_RATE:>7.2f} {predicted:>11.4f} "
              f"{measured:>10.4f} {error:>10.2e} {droop_db:>7.2f} dB")

    print("-" * 60)
    if failures:
        print(f"{failures} of {len(TEST_FREQS)} case(s) disagree by more than "
              f"{TOLERANCE:g}. Check your formula and your interpolation.")
    else:
        print("Prediction matches measurement at every frequency.")


if __name__ == "__main__":
    main()
