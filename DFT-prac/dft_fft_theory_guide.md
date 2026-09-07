# DFT / FFT Study Guide — theory behind transforms.py, bigmul.py, image_conv.py

This assignment is one idea applied twice: **turn a hard O(N²)-or-worse
problem into a cheap frequency-domain multiplication using the DFT, then
speed the DFT itself up with an FFT.** Everything below builds toward that.

---

## 1. The DFT itself

For a length-N sequence `x[0..N-1]` (real or complex):

```
Forward (analysis):   X[k] = Σ_{n=0}^{N-1} x[n] · e^(-2πi·kn/N),   k = 0..N-1
Inverse (synthesis):  x[n] = (1/N) Σ_{k=0}^{N-1} X[k] · e^(+2πi·kn/N)
```

**How to read it:** `X[k]` measures how much the complex sinusoid of
frequency `k/N` (one full cycle every `N/k` samples) is present in `x`.
`e^(-2πi·kn/N)` is often written `W_N^{kn}`, the **N-th root of unity**
raised to the `kn` power — this is the "twiddle factor" you'll see
everywhere.

Properties worth having memorized for a test:

- **Linearity**: DFT(ax + by) = a·DFT(x) + b·DFT(y).
- **Periodicity**: both `X[k]` and the basis functions are periodic in `N`
  — `X[k] = X[k+N]`. This is *why* circular convolution exists (§3):
  everything the DFT touches is implicitly wrapped mod N.
- **Conjugate symmetry for real input**: if `x` is real, `X[N-k] = conj(X[k])`.
  Half the spectrum is redundant — real-FFT libraries exploit this, though
  this assignment's engines don't bother (they keep everything complex).
- **Cost**: the naive double sum is `O(N)` work per output sample, `N`
  samples → **O(N²)**. That's exactly what `DFTAnalyzer.transform` does —
  two nested `for` loops, no shortcuts. It exists in this codebase purely as
  a slow-but-obviously-correct reference to test the FFT against.

`DFTAnalyzer.inverse` is the same double loop with the sign of the exponent
flipped and a `1/N` at the end — a direct transcription of the synthesis
formula, nothing clever.

---

## 2. The FFT: Cooley–Tukey radix-2, decimation-in-time

Computing the DFT directly is O(N²). The **Cooley–Tukey** algorithm exploits
a symmetry in the twiddle factors to do it in **O(N log N)**, but only when
`N` is a power of two (radix-2) — hence `next_power_of_two()` appearing
everywhere in this codebase; every caller pads up to a legal FFT length
before transforming.

### The recursive idea
Split `x` into its even- and odd-indexed samples, `x_even` and `x_odd`
(each length N/2). Their two DFTs, `E[k]` and `O[k]`, combine into the full
DFT of `x` via the **butterfly**:

```
X[k]       = E[k] + W_N^k · O[k]      for k = 0 .. N/2 - 1
X[k+N/2]   = E[k] - W_N^k · O[k]
```

Apply this recursively (each half splits again) and you get `log2(N)`
levels of splitting, each level doing O(N) butterfly work → **O(N log N)**
total. This identity (sometimes called the Danielson–Lanczos lemma) is *the*
one fact that makes the FFT possible — if you remember one formula for the
test, remember this pair of lines.

### The iterative version actually in the code
`FFTTransformer.transform` doesn't recurse — it does the equivalent thing
bottom-up, which is faster in practice (no call-stack/array-copy overhead):

1. **Bit-reversal permutation first** (`bit_reverse_array`). Recursive
   splitting-by-parity, if you unrolled it all the way down to
   single-element "DFTs", would land element `x[n]` at the position whose
   binary representation is the *bit-reversal* of `n`. So the iterative
   version front-loads that reordering once, then never has to recurse —
   it just runs the same reorder as a permutation up front.
   `bit_reverse_array` builds this by taking each index, extracting each
   bit with `(indices >> i) & 1`, and writing it back in the mirrored
   position `num_bits - i - 1`.

2. **`log2(N)` stages of butterflies**, stage size `M = 2, 4, 8, ..., N`.
   Within a stage, `N/M` independent butterfly groups of size `M` run in
   parallel (the `for l in range(0, N, M)` loop); each group combines pairs
   `M/2` apart using the formula above.

3. **Twiddle factors computed once per stage, not once per butterfly** —
   `W_arr = np.exp(-2j*np.pi*k_arr/M)` is built once outside the inner
   loop and reused for every group in that stage. This is a deliberate
   efficiency requirement in the docstring: computing `exp()` per-butterfly
   would still be correct but needlessly slow (exp() is expensive, and the
   same `M/2` twiddle values repeat across every group in the stage).

Net result: same answer as `DFTAnalyzer`, in `O(N log N)` instead of
`O(N²)` — verified directly in the module's own self-test (`np.max(np.abs(d.transform(x) - f.transform(x))) < 1e-9`).

### The inverse FFT — the conjugation trick
`FFTTransformer.inverse` does **not** re-implement a second butterfly
network with flipped-sign twiddles. It reuses the forward transform via a
classic identity:

```
IDFT(X) = conj( DFT( conj(X) ) ) / N
```

**Why this works:** the IDFT sum is `Σ X[k]·e^{+2πi kn/N}`. Conjugate `X`
first, then run the *forward* DFT on it: `Σ conj(X[k])·e^{-2πi kn/N}`.
Conjugating that whole expression flips every sign back:
`Σ X[k]·e^{+2πi kn/N}` — exactly the IDFT sum, missing only the `1/N`. So
`conjugate → forward FFT → conjugate → divide by N` gives the inverse for
free, with **one** piece of butterfly code instead of two. This is exactly
what the code does:
`np.conjugate(self.transform(np.conjugate(spectrum))) / N`.

---

## 3. The convolution theorem (the actual point of the assignment)

This is the idea both applications hinge on:

> **Multiplying two spectra pointwise = convolving the two original
> sequences.** Convolution in the "signal" domain becomes cheap
> multiplication in the frequency domain.

But the DFT is inherently *periodic* (§1), so what you get from
`IDFT( DFT(a) · DFT(b) )` is **circular convolution of period N**, not the
ordinary "line them up and slide" **linear convolution** you actually want
for polynomial/integer multiplication or image filtering:

```
circular:  c[k] = Σ_{i=0}^{N-1} a[i]·b[(k-i) mod N]      -- wraps around
linear:    c[k] = Σ_i a[i]·b[k-i]                         -- no wraparound
```

**The fix — zero-padding.** If `a` has length `Na` and `b` has length `Nb`,
the true linear convolution has length `Na+Nb-1`. Zero-pad both arrays out
to at least that length before transforming; then there's nowhere for the
tail of one signal to wrap into the head of the other, so circular = linear
on that padded length. If you're using the radix-2 FFT you must then round
*up again* to the next power of two — extra zeros never hurt correctness,
they just add unused frequency bins.

This single idea (pad → transform both → multiply spectra → inverse
transform → crop) is reused three times in this codebase:

| Where | Sequences being convolved |
|---|---|
| Polynomial product (`bigmul.py`'s design) | polynomial coefficients |
| `bigmul.py::multiply_transform` | big-integer limb arrays |
| `image_conv.py::convolve_plane` | image rows/cols × kernel (2D version) |

`image_conv.py` also deliberately shows what happens **without** the
padding — see §6 (`circular=True`), because seeing the wraparound artifact
is the fastest way to actually understand why the padding rule exists.

---

## 4. Bluestein's algorithm — `ArbitraryLengthFFT`

Radix-2 needs a power-of-two length. Bluestein's **chirp-z transform**
gets O(N log N) for *any* N by turning the DFT itself into a convolution
(so you can lean on the machinery from §3, using a power-of-two FFT
underneath).

**The algebraic trick.** DFT needs `e^{-2πi·nk/N}`, i.e. it needs the
product `nk`. Complete the square:

```
nk = ( n² + k² − (k−n)² ) / 2
```

so

```
e^{-2πi nk/N} = e^{-iπn²/N} · e^{-iπk²/N} · e^{+iπ(k-n)²/N}
```

Define the **chirp** `c[n] = e^{-iπn²/N}` (this is exactly `chirp_n` in the
code). Then:

```
X[k] = c[k] · Σ_n ( x[n]·c[n] ) · conj(c[k-n])
     = c[k] · ( a ⋆ b )[k],   where a[n] = x[n]·c[n],  b[m] = conj(c[m])
```

That inner sum is a **linear convolution** of `a` and `b` — computable with
the padding trick from §3, using the power-of-two `FFTTransformer` the
class holds as `self.fft_engine`. That's the whole method, line for line:

- `a = x * chirp_n` — the signal, pre-multiplied by the chirp.
- `b_padded` holds `conj(chirp_n)` at **both** positive indices `0..N-1`
  and negative indices, stored at the *end* of the padded array (index
  `pad_length - n`) because `b[m] = conj(c[m])` needs to be evaluated for
  `m` ranging over `-(N-1)..(N-1)`, and negative indices in a circular
  buffer live at the high end — this is just "negative index = wrap to the
  back," the same convention NumPy uses for `arr[-1]`.
- `pad_length = next_power_of_two(2N-1)`: the convolution support needs to
  cover the full `-(N-1)..(N-1)` range, i.e. `2N-1` values, then rounded up
  for the internal radix-2 FFT — same padding logic as §3, just applied to
  a length that's now `2N-1` instead of `Na+Nb-1`.
- Multiply spectra, inverse-transform, keep the first `N` values, multiply
  by `chirp_n` again (the leftover `c[k]` factor) → done.

If `N` is *already* a power of two, `ArbitraryLengthFFT.transform` just
delegates straight to `self.fft_engine.transform(x)` — no reason to pay
Bluestein's larger constant factor when radix-2 already applies directly.
Its `inverse` reuses the exact same conjugation trick as §2.

---

## 5. Task A theory — big-integer multiplication (`bigmul.py`)

**Core idea:** a big integer *is* a polynomial. Writing a number in base
`B` (`BASE = 10**BASE_DIGITS`) as `Σ limb[i]·B^i` is literally evaluating a
polynomial with coefficients `limb[i]` at `x = B`. Multiplying two
integers `A·B` is therefore the same operation as multiplying two
polynomials: **convolve the coefficient (limb) arrays, then handle carries.**

```
limbs(A) = [a0, a1, ..., am]      (little-endian, a_i is the coeff of B^i)
limbs(B) = [b0, b1, ..., bn]
product coefficient c_k = Σ_{i+j=k} a_i · b_j     <-- exactly linear convolution
```

Grade-school long multiplication computes exactly the same `c_k` sums —
FFT convolution just computes *all* of them at once in `O(N log N)`
instead of one at a time.

### Why limbs of `BASE_DIGITS = 4`, not one digit at a time?
Two reasons, both worth stating on a test:

1. **Speed** — packing 4 decimal digits per limb makes the coefficient
   arrays ~4× shorter, so the FFT is faster (fewer points to transform).
2. **Numerical precision** — the FFT/IFFT round-trip is done in
   double-precision floats, which carry ~15-16 significant decimal digits.
   Each convolution output `c_k` is a sum of up to `min(len(a), len(b))`
   products of two limb values (each limb `< BASE = 10^4`, so each product
   is `< 10^8`), plus the FFT's own rounding error accumulated over
   `log2(N)` stages. Keep `BASE_DIGITS` small enough and the true integer
   result stays *comfortably* far from the point where float rounding
   would flip `np.round()` to the wrong integer. Push `BASE_DIGITS` too
   high and the un-rounded convolution values get large enough, and
   accumulate enough FFT rounding error, that `np.round()` can silently
   pick the wrong limb value — a MISMATCH that isn't a bug in your FFT,
   just a base chosen too large for float64.

### Walking the pipeline
- **`to_limbs`**: strip the sign, chop the decimal string into groups of
  `base_digits` from the *right* (least-significant end), producing a
  little-endian `int64` array. The sign is carried separately — the
  transform never sees it, because a DFT has no concept of "sign of the
  whole number," only of the coefficient values.
- **`multiply_transform`**: choose `N` — the exact linear-convolution
  length `len(a)+len(b)-1` for `DFTAnalyzer` (no power-of-two constraint),
  or `next_power_of_two(...)` for the FFT-based engines (§3's padding
  rule); zero-pad both limb arrays to `N`; `transform → multiply → inverse
  → round to nearest int64`. The rounding is needed because the "true"
  answer is an exact integer, but floats leave residual error on the order
  of `1e-9`–`1e-6`.
- **`from_limbs`**: the convolution output is *not* base-`B` yet — a limb
  can be up to `(B-1)²·N`, way over `B`. This function sweeps
  least-significant-limb → most-significant, doing exactly the carry
  propagation you'd do multiplying two numbers by hand
  (`limb[i] % base`, push `limb[i] // base` into `limb[i+1]`), then strips
  leading zero limbs and reattaches the sign.
- **`multiply_schoolbook`**: the textbook `O(n²)` convolution, written with
  NumPy's vectorized slicing instead of a true FFT — this exists purely as
  a comparison baseline for the runtime plot; it has a much smaller
  constant factor than the naive DFT so it stays competitive up to
  surprisingly large sizes before `O(n log n)` wins out.
- **Complexity takeaway**: multiplying via `DFTAnalyzer` is *still* `O(n²)`
  overall (dominated by the O(n²) transform itself) — it has no asymptotic
  advantage over schoolbook, just a different constant factor. Only the
  radix-2/Bluestein engines turn the whole multiplication into `O(n log n)`,
  which is the entire point of `run_benchmark`'s `n²` vs `n log n` reference
  curves.
- **Verification**: Python's arbitrary-precision `int` multiplication is
  the *only* place ordinary big-int multiply is allowed, used purely to
  check `product_string == str(int(text_a) * int(text_b))` — a MATCH/MISMATCH
  oracle, not part of the algorithm itself.

---

## 6. Task B theory — 2D image convolution (`image_conv.py`)

### The 2D DFT is separable
```
X[kx, ky] = Σ_x Σ_y  x[x, y] · e^{-2πi(kx·x/W + ky·y/H)}
          = Σ_x  e^{-2πi kx·x/W} · ( Σ_y  x[x, y] · e^{-2πi ky·y/H} )
```
Because the 2D exponential factors into a product of two 1D exponentials,
the whole 2D sum factors into **"1D-transform every row, then 1D-transform
every column of the result (or vice versa — order doesn't matter)."** This
is the *only* reason a 2D transform is affordable: computing the 2D
definition directly for a `P×Q` image is `O(P²Q²)` (every one of `P·Q`
output bins needs a sum over all `P·Q` input pixels); doing it as `P` (or
`Q`) row-FFTs of length `Q`, then `Q` column-FFTs of length `P`, costs
`O(PQ·log Q + PQ·log P)`, i.e. roughly `O(N² log N)` for a square `N×N`
image versus `O(N⁴)` direct — an enormous win.

That's exactly `transform_2d`/`inverse_2d` in the code: loop over rows,
call the 1D `engine.transform`/`inverse` on each; transpose (so "columns"
become "rows"); loop again; transpose back. `inverse_2d` is the identical
structure with `engine.inverse` in place of `engine.transform`.

### 2D convolution theorem, with the padding gotcha made visible
Exactly §3's rule, in two dimensions at once: to get the true **linear**
2D convolution of an `(H,W)` image with a `(kh,kw)` kernel (output size
`(H+kh-1, W+kw-1)`), zero-pad *both* dimensions of *both* arrays out to at
least that size (further, to a power of two per axis, for the FFT engine)
before transforming, multiplying spectra, and inverse-transforming.

**The half-kernel shift.** The padded kernel sits with its `[0,0]` entry
at the array's origin, not centered — so the raw convolution output is
shifted by `(kh//2, kw//2)` relative to where you'd naturally expect the
filtered image to line up with the original. `convolve_plane` corrects
this by cropping the window `[kh//2 : kh//2+H, kw//2 : kw//2+W]` out of
the padded result — miss this step and the blurred image comes out offset
diagonally from the original, a classic FFT-convolution bug.

**`circular=True` — the deliberate mistake, made concrete.** This path
transforms at *exactly* `(H, W)` with **no padding at all**, and instead
uses `np.roll` to wrap the kernel around the origin (`np.roll` is plain
array indexing, not a transform, so it's allowed). Multiplying unpadded
spectra necessarily produces the *circular* 2D convolution: whatever
content the blur would push off one edge reappears on the opposite edge,
because the DFT treats the image as periodic (§1's periodicity property,
now visible as a picture instead of an equation). Comparing `blurred.png`
against `wraparound.png` is the single clearest illustration in this whole
assignment of *why* the zero-padding rule in §3 exists.

### `convolve_plane_direct` — the correctness oracle
A literal 4-nested-loop spatial convolution,
`out[r,c] = Σ_i Σ_j plane[r+kh//2-i, c+kw//2-j]·kernel[i,j]` with
out-of-range pixels treated as zero — `O(H·W·kh·kw)`, deliberately slow,
deliberately obvious, used only to sanity-check the FFT path on a small
64×64 crop (`max |spectral − direct| ≤ 1e-9` is the pass bar; this is pure
floating-point rounding, not a modeling difference — anything larger means
a bug, e.g. a missing crop offset or wrong padding size).

### Colour images
An `(H,W,3)` image is just three independent `(H,W)` planes; `convolve_image`
runs `convolve_plane` on each channel separately and re-stacks them —
there's no cross-channel mixing, so no new theory here, just a loop.

### Why the benchmark reference curves are what they are
- **Study 1 (fixed kernel, growing image `N×N`)**: naive row-column DFT does
  `N` row-DFTs of `O(N²)` each *twice* (rows then columns) → `O(N³)`
  — hence the `"n3"` reference curve. Direct spatial convolution with a
  **fixed-size** kernel costs `O(N²·k²)` = `O(N²)` for constant `k` — hence
  `"n2"`. The FFT row-column path is `O(N²·log N)`, which looks close to
  `N²` over the plotted range, dramatically beating both.
- **Study 2 (fixed image, growing kernel `K×K`)**: direct spatial
  convolution cost scales with kernel area, `O(K²)` for fixed image size —
  hence `"n2"`. The FFT path's cost is dominated by the (fixed) image size
  until the kernel grows large enough to force bigger padding, so its curve
  stays nearly flat — the practical payoff of frequency-domain filtering:
  **cost stops depending on kernel size** almost entirely, unlike spatial
  convolution.

---

## 7. Quick-reference complexity table

| Operation | Naive | FFT-based |
|---|---|---|
| 1D DFT of length N | O(N²) | O(N log N) (radix-2, N a power of 2) |
| 1D DFT, arbitrary N | O(N²) | O(N log N) (Bluestein, via 3 power-of-two FFTs) |
| Linear convolution of two length-N sequences | O(N²) | O(N log N) (pad → FFT → multiply → IFFT) |
| N-digit integer multiplication | O(N²) (schoolbook) | O(N log N) (limb convolution) |
| 2D transform of an N×N image | O(N⁴) (direct 2D sum) | O(N² log N) (row-column separable FFT) |
| N×N image convolved with fixed k×k kernel | O(N²k²) (direct) | O(N² log N) (FFT, ~independent of k) |

---

## 8. Things a DFT/FFT test is likely to probe

- Derive or recognize the butterfly identity `X[k]=E[k]+W_N^k·O[k]`,
  `X[k+N/2]=E[k]-W_N^k·O[k]` and why it halves the problem.
- Explain **why radix-2 needs a power-of-two N** (the split-in-half
  recursion needs an even length at every level, all the way down).
- State the **conjugation identity** for getting the inverse FFT for free,
  and be able to derive it (§2).
- Explain **circular vs linear convolution** and the exact minimum padding
  length needed to avoid wraparound (`Na+Nb-1`, or per-axis in 2D).
- Explain **why 2D DFTs are computed as 1D row transforms + 1D column
  transforms** (separability of the exponential kernel) and the resulting
  complexity (`O(N² log N)` vs `O(N⁴)`).
- Bluestein: know that it exists to escape the power-of-two restriction,
  and roughly *how* — turning a DFT into a convolution via the
  `nk=(n²+k²-(k-n)²)/2` identity — even if you can't rederive every line.
- Be able to explain, in your own words, why big-integer multiplication and
  polynomial multiplication and image blurring are "the same problem" —
  they're all instances of the convolution theorem.
