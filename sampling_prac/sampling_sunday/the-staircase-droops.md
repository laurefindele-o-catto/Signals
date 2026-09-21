# Assignment: The Staircase Droops


## Background

A real digital-to-analog converter does not emit isolated sample values. It holds each sample steady until the next one arrives, producing a staircase. This is the **zero-order hold**, and it is not a neutral operation: holding a value for a full sample period `T = 1/fs` attenuates the signal, and the attenuation gets worse as the frequency rises.

The amount is exactly

```
gain(f) = |sinc(f/fs)|
```

At low frequencies this is close to 1 and mostly unnoticeable. Near Nyquist rate it is not — which is why converter datasheets quote a "sinc droop" figure, and why some systems compensate for it in advance.

Your job is to predict the droop from the formula, measure it from an actual staircase, and confirm the two agree.

## Your task

The starter file contains two functions with their bodies removed. Fill in both. `tone_amplitude()` and `main()` are provided; do not modify them.

### Part 1 — `zoh_gain(f, fs)`

Return the predicted gain from the formula above.

### Part 2 — `measured_zoh_gain(f, fs, upsample, duration)`

Sample `cos(2πft)` at rate `fs`, build the staircase by holding each sample for `upsample` steps of a finer grid, then measure the amplitude of the `f` component of the staircase. Since the input cosine has amplitude 1, that measured amplitude *is* the gain.

Use the provided `tone_amplitude()` helper for the measurement. The fine grid runs at `upsample * fs`.



## Starter file — `zoh.py`

```python
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


```

## Expected output

```
  f (Hz)    f/fs   predicted   measured    |error|      droop
------------------------------------------------------------
      50    0.05      0.9959     0.9959   4.10e-07   -0.04 dB
     100    0.10      0.9836     0.9836   1.62e-06   -0.14 dB
     200    0.20      0.9355     0.9355   6.16e-06   -0.58 dB
     300    0.30      0.8584     0.8584   1.27e-05   -1.33 dB
     450    0.45      0.6986     0.6987   2.33e-05   -3.11 dB
------------------------------------------------------------
Prediction matches measurement at every frequency.
```

