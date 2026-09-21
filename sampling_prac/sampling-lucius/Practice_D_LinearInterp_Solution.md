# Solution: The Ramp Between the Steps

Try the problem on your own first. This walks through the reasoning in the order you would work it out.

---

## Part 1: `linear_interp_gain(f, fs)`

**Step 1: read the formula.** `gain = sinc(f/fs)²`, with `sinc(x) = sin(πx)/(πx)`.

**Step 2: use the provided helper.** `np.sinc` is already the *normalised* sinc (it includes the `π`), so no extra `π` is needed. Do **not** write `np.sinc(np.pi * f / fs)`.

**Step 3: square it.** Since `sinc` can be negative, and squaring makes it non-negative, no `abs` is needed.

```python
def linear_interp_gain(f, fs):
    return float(sinc(f / fs) ** 2)
```

**Check by hand at `f/fs = 0.45`:**
`sin(0.45π) = 0.9877`, `0.45π = 1.4137`, so `sinc = 0.6986`. Squared: **0.4881**, or −6.23 dB.

**Why the extra square?** Zero-order hold convolves the samples with a rectangle, whose transform is `sinc`. Linear interpolation convolves with a *triangle*, which is a rectangle convolved with itself, so its transform is `sinc` squared. Notice the droop in dB is about twice the ZOH figure (−3.11 dB vs −6.23 dB at `f/fs = 0.45`).

**Common mistake:** using `sinc(f/fs)` without squaring. That predicts 0.6986 at 450 Hz, the ZOH answer, and the table would show a large error.

---

## Part 2: `measured_linear_interp_gain(f, fs, upsample, duration)`

**Step 1: choose the sample times.** `N = int(duration * fs)` and `n = np.arange(N + 1)`. This is `N+1` points because linear interpolation needs a sample on **both** ends of the last interval.

**Step 2: sample.** `samples = np.cos(2π f n / fs)` and `t_coarse = n / fs`.

**Step 3: build the fine grid.** `t_fine = np.arange(N * upsample) / (upsample * fs)`. It covers `[0, duration)` at `upsample` times the sample rate.

**Step 4: connect the dots.** `y = np.interp(t_fine, t_coarse, samples)`. `np.interp` requires `t_coarse` to be increasing, which it is here.

**Step 5: measure.** `tone_amplitude(y, upsample * fs, f)`. The second argument is the *fine* rate, exactly as in the zero-order-hold assignment.

```python
def measured_linear_interp_gain(f, fs, upsample, duration):
    N = int(duration * fs)
    n = np.arange(N + 1)
    samples = np.cos(2 * np.pi * f * n / fs)
    t_coarse = n / fs
    t_fine = np.arange(N * upsample) / (upsample * fs)
    y = np.interp(t_fine, t_coarse, samples)
    return tone_amplitude(y, upsample * fs, f)
```

**Why the endpoint matters.** Each test frequency fits a whole number of cycles into `duration`, so the true signal is periodic over the record and the DFT expects the interpolated signal to be too. If you sample only `N` points, `np.interp` has nothing to interpolate towards past the last sample and simply holds the last value flat for the final interval. That small flat stretch breaks the periodicity and leaks error into the measurement. With `N` samples instead of `N+1` the error at 450 Hz is about 9e-3, well over the 1e-3 tolerance; with the endpoint it drops to about 3e-5.

**Why the small leftover error?** The measurement is a DFT on a finite fine grid, so a little of the image content near multiples of `upsample·fs` folds back onto the measured bin. It shrinks as `upsample` grows, and it is why `|error|` is around 1e-5, not zero.

---

## Full solution

```python
def linear_interp_gain(f, fs):
    return float(sinc(f / fs) ** 2)


def measured_linear_interp_gain(f, fs, upsample, duration):
    N = int(duration * fs)
    n = np.arange(N + 1)
    samples = np.cos(2 * np.pi * f * n / fs)
    t_coarse = n / fs
    t_fine = np.arange(N * upsample) / (upsample * fs)
    y = np.interp(t_fine, t_coarse, samples)
    return tone_amplitude(y, upsample * fs, f)
```

Running the file reproduces the expected table, and the ZOH comparison above (−3.11 dB vs −6.23 dB at 450 Hz) is a good sanity check to be able to state on paper.
