# Signals Lab (Section C) — Exam Prep

Built from the Section A (2D CFT / `FrequencyFilter`) and Section B (Fourier series / `FourierEpicycles`) papers you shared. The pattern in both: take the class from your offline, bolt on 2–3 new methods, timed. Section C will very likely follow the same shape — one CFT-image task, one Fourier-series task.

---

## Part 1 — Predicted Task List

### 1A. 2D CFT / `FrequencyFilter` — likely variants

| Likely task | What it tests |
|---|---|
| Low-pass / high-pass filter (single threshold, not band) | Same `d(i,j)` machinery as band-pass, just one bound instead of two |
| Notch / square (non-radial) mask | Whether you understand the mask is *any* boolean array, not just radial |
| Magnitude–phase swap between two images | `np.abs` / `np.angle` decomposition, reconstructing with `np.exp(1j*phase)` |
| Phase-only or magnitude-only reconstruction | Same decomposition, one factor set to 1 or constant |
| Parseval / energy check (2D analogue of Task 2 in A) | `sum(|f|^2)` in spatial domain ≈ `sum(|F|^2)/(H*W)` in frequency domain |
| Shift theorem verification | Translating image ⇔ multiplying spectrum by linear phase ramp |
| Conjugate symmetry check for real images | `F(-u,-v) = conj(F(u,v))` — testable with `np.allclose` on flipped array |
| DRPE encrypt/decrypt (ties directly to your offline) | Apply random phase mask spatially, FFT, apply 2nd random phase mask, IFFT; decrypt reverses it; wrong-key decrypt should look like noise |
| Extend `shift_amount` DC task to scaling or zeroing the DC term | Same single-pixel indexing, different operation |

### 1B. Fourier Series / `FourierEpicycles` — likely variants

| Likely task | What it tests |
|---|---|
| Prune by **top-K magnitude** instead of energy ratio | `np.argsort(np.abs(coeffs))[::-1]`, same idea as B's energy pruning but ranked differently |
| Low-pass truncation: keep only `\|n\| <= cutoff` | Simple index slicing instead of energy-ranked selection |
| Parseval's theorem check for Fourier series | `sum(|c_n|^2)` over retained harmonics vs `(1/T)∫|f(t)|^2 dt` |
| MSE vs N sweep / convergence plot | Same `evaluate_reconstruction_error` from B, called in a loop |
| Compute `c_n` numerically from a parametrized curve via **trapezoidal integration** | This is the piece most likely to trip people up — see Part 3 |
| Drawing-direction reversal via conjugate coefficients | `c_{-n} <-> conj(c_n)` swap, tests whether you get what negative frequency *means* |
| Gibbs phenomenon commentary (short-answer, not code) | Conceptual: sharp corners in the SVG need disproportionately many harmonics |

**Bottom line prediction:** expect one filter/mask-construction task (image side) and one coefficient-ranking-and-truncation task (epicycle side), each paired with a numeric validation check (MSE, Parseval, or a max-abs-difference threshold like `< 1e-9` from A's Task 2). That validation-check pairing showed up in *both* A and B — treat it as near-certain, not just likely.

---

## Part 2 — Core Theory, Compressed

### 2D CFT / DFT basics
- For an `H x W` image, `np.fft.fft2` gives a complex spectrum. The **center** after `fftshift` is `(H//2, W//2)` — this is your `(c_i, c_j)`.
- Center = **low frequency** = slowly-varying brightness/color (the "shape" of the image). Edges of the spectrum = **high frequency** = fine detail, sharp edges, noise.
- `d(i,j) = sqrt((i-c_i)^2 + (j-c_j)^2)` is just Euclidean distance from center — a radial mask is "keep/discard everything at this distance range."
- **Band-pass + band-stop = identity** because the two masks are exact complements (every pixel is in exactly one of the two regions) and FFT/IFFT are linear: `IFFT(mask) + IFFT(1-mask) = IFFT(mask + (1-mask)) = IFFT(1) = original`.
- The **DC component** `F(c_i, c_j)` (zero frequency) equals the sum of all pixel values — i.e., proportional to the average brightness. Shifting it shifts the whole image's average brightness without touching any spatial structure.
- **Parseval's theorem (2D)**: energy is conserved between domains, up to a normalization constant that depends on your FFT convention (numpy is unnormalized forward / `1/(H*W)` inverse, so watch the scaling factor when you check equality).
- **Shift theorem**: shifting an image in space multiplies its spectrum by a complex exponential (linear phase ramp) — magnitude is unchanged, only phase rotates.
- **Magnitude vs phase**: magnitude carries "how much" of each frequency; phase carries "where." A classic demo (and plausible exam task) is swapping phase between two images — the result looks like the image whose *phase* it took, because phase carries most of the structural information.

### Fourier series / epicycles basics
- `c_n = (1/T) ∫ f(t) e^{-i n ω t} dt` over one period. Each `c_n` is a rotating vector (epicycle) of magnitude `|c_n|`, phase `angle(c_n)`, spinning at frequency `n` cycles per period.
- Reconstructing with all `2N+1` harmonics (`n = -N` to `N`) reproduces the shape; **pruning** means keeping only the harmonics that matter and zeroing the rest.
- **Energy** of the signal: `E_total = sum_n |c_n|^2` (Parseval for Fourier series). Energy-based pruning (Task 1 in B) ranks harmonics by `|c_n|^2` and keeps the smallest set that covers a target fraction of `E_total` — this is different from magnitude-based or frequency-based (low-pass) pruning, and it's worth being able to implement all three fast, since the exam swaps which one it asks for.
- **MSE**: straightforward — sample both signals at the same `t_i`, average squared magnitude of the difference. For complex-valued epicycle output, `|f - f_hat|^2` is `(f-f_hat) * conj(f-f_hat)`, which `np.abs(...)**2` already handles correctly.
- Sharp corners / discontinuities in the curve need many high harmonics to resolve (Gibbs phenomenon) — this is why energy ratio 1.00 in B's target list requires close to all `N=150` harmonics for a shape like a heart with a sharp cusp, while 0.96–0.99 can prune substantially. If asked to explain a result qualitatively, this is the reasoning.

---

## Part 3 — NumPy/Python Machinery You'll Actually Type

This is the part worth drilling tonight — the theory above is quick to state, but under time pressure the vectorized numpy is where marks get lost.

### Building a radial distance grid (for masks)
```python
H, W = spectrum.shape
ci, cj = H // 2, W // 2
i_idx, j_idx = np.indices((H, W))          # or np.meshgrid with indexing='ij'
d = np.sqrt((i_idx - ci)**2 + (j_idx - cj)**2)
```
`np.indices` is faster to type correctly under pressure than `meshgrid` because you don't have to remember `indexing='ij'` vs `'xy'`.

### Boolean masks, applied to real & imaginary parts separately
```python
mask = (d > r_low) & (d <= r_high)          # band-pass condition
filtered_real = np.where(mask, spectrum.real, 0)
filtered_imag = np.where(mask, spectrum.imag, 0)
# or, since mask is real-valued 0/1, simply:
filtered = spectrum * mask
```
Note the last line: multiplying a complex array by a boolean/0-1 mask zeroes real and imaginary parts simultaneously — usually simpler than splitting `.real`/`.imag` unless the task explicitly asks you to touch them separately.

### Magnitude / phase decomposition and reassembly
```python
mag = np.abs(spectrum)
phase = np.angle(spectrum)
reconstructed = mag * np.exp(1j * phase)     # identity check
swapped = mag_A * np.exp(1j * phase_B)       # classic magnitude/phase swap
```

### Single-pixel DC manipulation
```python
spectrum[ci, cj] += shift_amount             # or *= , or = 0, depending on the ask
```

### Energy-ranked pruning (2D or 1D, same idea)
```python
energy = np.abs(coeffs)**2
order = np.argsort(energy)[::-1]             # indices, most energetic first
cumulative = np.cumsum(energy[order])
total = energy.sum()
k = np.searchsorted(cumulative, r * total) + 1   # min harmonics for ratio r
keep_idx = order[:k]
pruned = np.zeros_like(coeffs)
pruned[keep_idx] = coeffs[keep_idx]
actual_ratio = cumulative[k-1] / total
```
`np.searchsorted` on a cumulative sum is the fast, correct way to find "minimal count reaching threshold r" — much less error-prone than a manual `for` loop with a running total, and it's exactly the kind of one-liner that saves time in a lab.

### Top-K magnitude pruning (likely variant of the above)
```python
order = np.argsort(np.abs(coeffs))[::-1]
keep_idx = order[:k]        # k given directly instead of derived from a ratio
```

### Low-pass truncation by frequency index (not magnitude/energy)
```python
n = np.arange(-N, N+1)
pruned = np.where(np.abs(n) <= cutoff, coeffs, 0)
```

### Numerical Fourier coefficients via trapezoidal integration
If a task gives you sampled `(t_i, f(t_i))` from an SVG-derived parametrized curve and asks you to compute `c_n` numerically rather than analytically:
```python
def compute_cn(t, f_vals, n, T):
    integrand = f_vals * np.exp(-1j * n * 2*np.pi/T * t)
    return np.trapz(integrand, t) / T
```
Vectorize over all `n` at once if performance matters:
```python
n_vals = np.arange(-N, N+1)
# shape (2N+1, M) via broadcasting
integrand = f_vals[None, :] * np.exp(-1j * np.outer(n_vals, t) * 2*np.pi/T)
c = np.trapz(integrand, t, axis=1) / T
```

### MSE
```python
mse = np.mean(np.abs(f_vals - f_hat_vals)**2)
```

### Complementarity / reconstruction validation pattern (from A's Task 2 — expect this shape again)
```python
delta = np.max(np.abs(I_bp + I_bs - I_recon))
passed = delta < 1e-9
```
This exact pattern — max absolute difference against a `1e-9` threshold, returning `(bool, delta)` — is a strong candidate to reappear verbatim in Section C for whatever pairing of methods it tests. Practice writing it fast.

---

## Part 4 — Quick Self-Check Before the Exam

Try to answer these from memory, then verify:
1. Why does `fftshift` matter for defining `(c_i, c_j)` — what would the "center" be without it?
2. If `mask` zeroes the band `r_low < d <= r_high`, what's the one-line mask for its complement?
3. What numpy call turns a complex spectrum into (magnitude, phase) and back?
4. Why is DC energy proportional to average brightness, in one sentence?
5. What's the difference between pruning by energy ratio, top-K magnitude, and frequency cutoff — write one line each.
6. Why does a shape with sharp corners need more harmonics to reach a given MSE than a smooth shape?

If any of these are shaky, that's where to spend your remaining review time rather than re-reading the whole offline PDF.
