# Assignment: The Ramp Between the Steps


## Background

The zero-order hold turns samples into a staircase. A gentler reconstruction simply **connects the dots**: draw a straight line from each sample to the next. This is **linear interpolation** (a first-order hold). Because it follows the signal more closely than a staircase does, it distorts less, but it is still not free: it also attenuates the signal, and the attenuation gets worse as frequency rises.

The gain is exactly

```
gain(f) = sinc(f/fs)²
```

where `sinc(x) = sin(πx) / (πx)`. Compare the zero-order hold, whose gain is `|sinc(f/fs)|`. Linear interpolation squares it, so the droop is roughly twice as many decibels.

Your job is to predict the gain from the formula, measure it from an actual interpolated signal, and confirm the two agree.

## Your task

The starter file contains two functions with their bodies removed. Fill in both. `tone_amplitude()` and `main()` are provided; do not modify them.

### Part 1 — `linear_interp_gain(f, fs)`

Return the predicted gain from the formula above. A helper `sinc(x)` (numpy's normalised sinc) is provided at the top of the file.

### Part 2 — `measured_linear_interp_gain(f, fs, upsample, duration)`

Sample `cos(2πft)` at rate `fs`, rebuild it by linear interpolation onto a finer grid, then measure the amplitude of the `f` component of the result. Since the input cosine has amplitude 1, that measured amplitude *is* the gain.

Details, all also given in the docstring:
- Sample times are `n/fs` for `n = 0 … N`, **including the endpoint** (so `N+1` samples), where `N = int(duration * fs)`.
- The fine grid is `m / (upsample * fs)` for `m = 0 … N*upsample − 1`.
- Interpolate with `np.interp(t_fine, t_coarse, samples)`.
- Measure with `tone_amplitude(y, upsample * fs, f)`.


## Starter file — `lininterp.py`

```python
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
```

## Expected output

```
  f (Hz)    f/fs   predicted   measured    |error|      droop
------------------------------------------------------------
      50    0.05      0.9918     0.9918   8.16e-07   -0.07 dB
     100    0.10      0.9675     0.9675   3.18e-06   -0.29 dB
     200    0.20      0.8751     0.8752   1.15e-05   -1.16 dB
     300    0.30      0.7368     0.7369   2.18e-05   -2.65 dB
     450    0.45      0.4881     0.4881   3.25e-05   -6.23 dB
------------------------------------------------------------
Prediction matches measurement at every frequency.
```
