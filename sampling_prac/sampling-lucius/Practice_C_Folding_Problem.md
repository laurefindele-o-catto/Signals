# Assignment: Where Does the Tone Land?


## Background

Sampling a sinusoid at rate `fs` cannot tell `f` apart from `f + k·fs`, nor from `−f + k·fs`, for any integer `k`. After sampling, the tone therefore *appears* at a single frequency inside the band `[0, fs/2]`, no matter how high the original frequency was. This is the **apparent frequency** (also called the alias or folded frequency).

This assignment asks you to predict where a tone lands, then confirm it by locating the strongest bin in the DFT of the samples. Unlike the two-tones assignment, `f` may now be **any** positive frequency, including one far above `fs/2`.

## Your task

The starter file contains two functions with their bodies removed. Fill in both. `main()` is already written; do not modify it.

### Part 1 — `apparent_frequency(f, fs)`
"""The Staircase Droops: zero-order hold gain.

Complete the two functions marked TODO. Do not modify anything below
the divider. Run with:  python zoh.py
"""

import numpy as np


def zoh_gain(f, fs):
    """Predicted zero-order-hold gain at frequency f, sampling at rate fs.

    gain = |sinc(f / fs)|

    """
    raise NotImplementedError


def measured_zoh_gain(f, fs, upsample, duration):
    """Gain of the zero-order hold, measured rather than predicted.

    Samples cos(2*pi*f*t) at rate fs over [0, duration), holds each
    sample for `upsample` steps of a grid running at upsample*fs (use np.repeat(samples, upsample)), then
    returns the amplitude of the f component of the resulting staircase (using the tone amplitude method below, second parameter should be upsample * fs).

    The input amplitude is 1, so this amplitude is the gain.
    """
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



Return the frequency in `[0, fs/2]` at which a tone of frequency `f` appears after sampling at rate `fs`. Assume `f > 0`.

### Part 2 — `measured_peak_frequency(f, fs, duration)`

Empirical verification: sample `cos(2πft)` at times `n/fs` for `n = 0 … N−1`, where `N = int(duration * fs)`. Take the real DFT with `np.fft.rfft`, find the bin with the largest magnitude, and return its frequency in Hz. Use `np.fft.rfftfreq(N, 1/fs)` for the frequency axis.

Running `python fold.py` should print a table of seven cases in which the predicted and measured frequencies agree exactly.

(One case is deliberately awkward: a tone at exactly `fs`. Think about what its samples look like.)


## Starter file — `fold.py`

```python
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
    raise NotImplementedError


def measured_peak_frequency(f, fs, duration):
    """Frequency of the strongest DFT bin of the sampled cosine.

    Samples cos(2*pi*f*t) at rate fs using t = n/fs for n = 0 ... N-1,
    N = int(duration * fs). Takes the real DFT (np.fft.rfft) and returns the
    frequency (in Hz) of the bin with the largest magnitude.

    Returns a single float.
    """
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
```

## Expected output

```
  f (Hz)  fs (Hz)   predicted   measured   result
--------------------------------------------------------
     300     1000       300.0      300.0   match
     700     1000       300.0      300.0   match
    1300     1000       300.0      300.0   match
    2200     1000       200.0      200.0   match
    5000     8000      3000.0     3000.0   match
    9000     8000      1000.0     1000.0   match
    1000     1000         0.0        0.0   match
--------------------------------------------------------
Every prediction matches the measured peak.
```
