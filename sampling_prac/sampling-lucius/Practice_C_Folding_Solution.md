# Solution: Where Does the Tone Land?

Try the problem on your own first. This walks through the reasoning in the order you would work it out.

---

## Part 1: `apparent_frequency(f, fs)`

**Step 1: what does the sampler actually see?**
The samples are `cos(2π f n / fs)`. Replace `f` with `f + k·fs`:

```
cos(2π (f + k·fs) n / fs) = cos(2π f n / fs + 2π k n) = cos(2π f n / fs)
```

`k·n` is an integer, so the extra `2π k n` does nothing. Because cosine is even, `−f + k·fs` gives the same samples too. So the whole family `±f + k·fs` is indistinguishable.

**Step 2: which member of the family do we report?**
The one that lies in `[0, fs/2]`. Every `f` has exactly one.

**Step 3: turn that into arithmetic.**
Find the nearest multiple of `fs` and subtract it. `round(f / fs)` gives that multiple's index `k`, so `f − k·fs` lands in `[−fs/2, fs/2]`. The `−` side is covered by the `±` symmetry from Step 1, so take `abs`.

```python
def apparent_frequency(f, fs):
    return abs(f - fs * round(f / fs))
```

**Check by hand**

| f | fs | f/fs | round | f − round·fs | abs |
|---|---|---|---|---|---|
| 700 | 1000 | 0.7 | 1 | −300 | **300** |
| 1300 | 1000 | 1.3 | 1 | 300 | **300** |
| 2200 | 1000 | 2.2 | 2 | 200 | **200** |
| 9000 | 8000 | 1.125 | 1 | 1000 | **1000** |
| 1000 | 1000 | 1.0 | 1 | 0 | **0** |

**Equivalent modulo form** (some people find this cleaner):

```python
r = f % fs
return min(r, fs - r)
```

`f % fs` wraps `f` into `[0, fs)`, then `min(r, fs − r)` folds the upper half of that range back down.

**Common mistakes**
- Returning only `f % fs`. For `f = 700, fs = 1000` this gives 700, which is above Nyquist. You forgot to fold.
- Forgetting `abs`, which gives −300 for the 700 Hz case.

---

## Part 2: `measured_peak_frequency(f, fs, duration)`

**Step 1: build the samples.** Follow the spec literally: `N = int(duration * fs)`, `t = np.arange(N) / fs`, `x = np.cos(2π f t)`.

**Step 2: transform.** `np.fft.rfft(x)` gives the non-negative-frequency half of the spectrum. Take `np.abs` of it.

**Step 3: build the matching frequency axis.** `np.fft.rfftfreq(N, 1/fs)`. Bin `k` sits at `k · fs / N`. With `duration = 0.1` the spacing is `fs/N = 10 Hz`, and every test frequency is a multiple of 10, so the tone lands exactly on a bin.

**Step 4: pick the strongest bin.** `np.argmax` returns the *index*; use it to read the frequency out of the frequency axis.

```python
def measured_peak_frequency(f, fs, duration):
    N = int(duration * fs)
    t = np.arange(N) / fs
    x = np.cos(2 * np.pi * f * t)
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(N, 1 / fs)
    return float(freqs[np.argmax(spectrum)])
```

**About the awkward case, `f = fs = 1000`.** The samples are `cos(2π n)` = 1 for every `n`, a constant. A constant is DC, so the peak is at bin 0, i.e. 0 Hz, which is exactly what Part 1 predicts.

---

## Full solution

```python
import numpy as np


def apparent_frequency(f, fs):
    return abs(f - fs * round(f / fs))


def measured_peak_frequency(f, fs, duration):
    N = int(duration * fs)
    t = np.arange(N) / fs
    x = np.cos(2 * np.pi * f * t)
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(N, 1 / fs)
    return float(freqs[np.argmax(spectrum)])
```

Everything else in the file is unchanged. Running it should reproduce the expected output table exactly.
