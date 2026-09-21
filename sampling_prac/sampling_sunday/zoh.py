"""The Staircase Droops: zero-order hold gain.

Complete the two functions marked TODO. Do not modify anything below
the divider. Run with:  python zoh.py
"""

import numpy as np


def zoh_gain(f, fs):
    """Predicted zero-order-hold gain at frequency f, sampling at rate fs.

    gain = |sinc(f / fs)|

    """
    return abs(np.sinc(f/fs))
    raise NotImplementedError


def measured_zoh_gain(f, fs, upsample, duration):
    """Gain of the zero-order hold, measured rather than predicted.

    Samples cos(2*pi*f*t) at rate fs over [0, duration), holds each
    sample for `upsample` steps of a grid running at upsample*fs (use np.repeat(samples, upsample)), then
    returns the amplitude of the f component of the resulting staircase (using the tone amplitude method below, second parameter should be upsample * fs).

    The input amplitude is 1, so this amplitude is the gain.
    """
    
    N = int(duration*fs)
    t = np.arange(N) /fs
    cos_ft = np.cos(2*np.pi*f*t)
    
    upsample_grid = np.repeat(cos_ft, upsample)
    fs_fine = upsample * fs
    amplitude = tone_amplitude(upsample_grid, fs_fine, f)
    
    return amplitude
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


SAMPLE_RATE = 1000     # Hz
UPSAMPLE = 100         # fine-grid steps per held sample
DURATION = 0.1         # seconds
TOLERANCE = 1e-3

TEST_FREQS = [50, 100, 200, 300, 450]   # Hz


def main():
    print(f"{'f (Hz)':>8} {'f/fs':>7} {'predicted':>11} {'measured':>10} "
          f"{'|error|':>10} {'droop':>10}")
    print("-" * 60)

    failures = 0
    for f in TEST_FREQS:
        predicted = zoh_gain(f, SAMPLE_RATE)
        measured = measured_zoh_gain(f, SAMPLE_RATE, UPSAMPLE, DURATION)
        error = abs(predicted - measured)

        failures += error >= TOLERANCE
        droop_db = 20 * np.log10(predicted)
        print(f"{f:>8} {f / SAMPLE_RATE:>7.2f} {predicted:>11.4f} "
              f"{measured:>10.4f} {error:>10.2e} {droop_db:>7.2f} dB")

    print("-" * 60)
    if failures:
        print(f"{failures} of {len(TEST_FREQS)} case(s) disagree by more than "
              f"{TOLERANCE:g}. Check your formula and your staircase.")
    else:
        print("Prediction matches measurement at every frequency.")


if __name__ == "__main__":
    main()

