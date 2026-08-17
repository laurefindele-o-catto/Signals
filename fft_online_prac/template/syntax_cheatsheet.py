"""
Syntax cheat sheet for 2D CFT / FFT / harmonic-series problems.

Run directly to see every snippet's output:
    python3 syntax_cheatsheet.py
"""

import numpy as np


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# =====================================================================
# 1. Complex numbers: literals, exp, real/imag, abs, angle, conjugate
# =====================================================================
def demo_complex_numbers():
    section("1. Complex numbers")

    z = 3 + 4j                      # complex literal
    z2 = np.exp(1j * np.pi / 2)     # e^(j*pi/2) = 0 + 1j
    print("z  =", z)
    print("z2 =", z2)

    print("z.real =", z.real)       # real part of a scalar
    print("z.imag =", z.imag)       # imaginary part of a scalar
    print("abs(z) =", np.abs(z))    # magnitude |z|
    print("angle(z) [rad] =", np.angle(z))          # phase in radians
    print("angle(z) [deg] =", np.angle(z, deg=True))  # phase in degrees

    z_conj = np.conj(z)             # complex conjugate: a+bj -> a-bj
    print("conj(z) =", z_conj)
    print("z * conj(z) == |z|^2 ->", z * z_conj, "vs", np.abs(z) ** 2)

    # Elementwise on arrays -- same functions, no loop needed
    arr = np.array([1 + 2j, 3 - 1j, 0 + 5j])
    print("arr           =", arr)
    print("arr.real      =", arr.real)
    print("arr.imag      =", arr.imag)
    print("abs(arr)      =", np.abs(arr))
    print("angle(arr)    =", np.angle(arr))
    print("conj(arr)     =", np.conj(arr))


# =====================================================================
# 2. np.trapezoid: scalar integral, and batched with axis=
# =====================================================================
def demo_trapezoid():
    section("2. np.trapezoid (scalar and batched)")

    x = np.linspace(0, 1, 5)
    y = x ** 2
    print("scalar integral of x^2 over [0,1] ~=", np.trapezoid(y, x))

    # Batched: integrate each ROW of a 2D array over the SAME x-axis.
    # Row length must equal len(x).
    M = np.array([x, 2 * x, x ** 2])          # shape (3, 5): 3 signals, 5 samples each
    row_integrals = np.trapezoid(M, x, axis=1)  # -> shape (3,), one integral per row
    print("row-wise integrals (axis=1) =", row_integrals)

    # To integrate over axis=0 instead, the x-array must match that axis's length.
    x0 = np.linspace(0, 1, 3)
    col_integrals = np.trapezoid(M, x0, axis=0)  # -> shape (5,), one integral per column
    print("column-wise integrals (axis=0) =", col_integrals)


# =====================================================================
# 3. Vectorizing double loops: np.indices / meshgrid + boolean masks
# =====================================================================
def demo_vectorized_masking():
    section("3. Vectorized masking instead of nested for i / for j")

    rows, cols = 6, 8
    cx, cy = rows // 2, cols // 2

    # np.indices gives two 2D arrays: i[i,j] = i, j[i,j] = j
    i, j = np.indices((rows, cols))
    d = np.sqrt((i - cx) ** 2 + (j - cy) ** 2)

    r_low, r_high = 1, 3
    # NOTE: parentheses required around each comparison before combining with & / |
    mask = (d > r_low) & (d <= r_high)

    real = np.round(np.random.rand(rows, cols), 2)
    imag = np.round(np.random.rand(rows, cols), 2)

    # Method A: np.where(mask, keep_value, other_value)
    real_bp_a = np.where(mask, real, 0.0)
    imag_bp_a = np.where(mask, imag, 0.0)

    # Method B: boolean indexing with ~ (elementwise NOT), in place
    real_bp_b = real.copy()
    imag_bp_b = imag.copy()
    real_bp_b[~mask] = 0.0
    imag_bp_b[~mask] = 0.0

    print("mask:\n", mask.astype(int))
    print("Method A == Method B ?", np.allclose(real_bp_a, real_bp_b))


# =====================================================================
# 4. Outer products via broadcasting -- avoids Python-level loops
# =====================================================================
def demo_broadcasting_outer_product():
    section("4. Broadcasting to build 2D grids (outer products)")

    x = np.array([0.0, 0.5, 1.0])       # shape (3,)
    u = np.array([1.0, 2.0])            # shape (2,)

    # outer[k, n] = u[k] * x[n], shape (2, 3), no explicit loop
    outer = u[:, None] * x[None, :]
    print("outer product u*x:\n", outer)

    # Same idea for a full cos(2*pi*u*x) grid in one shot
    grid = np.cos(2 * np.pi * u[:, None] * x[None, :])
    print("cos(2*pi*u*x) grid shape:", grid.shape)
    print(grid)


# =====================================================================
# 5. np.fft.fft2 / fftshift / ifft2 -- for comparing against manual CFT
# =====================================================================
def demo_fft2():
    section("5. np.fft.fft2 / fftshift / ifft2")

    img = np.random.rand(8, 8)

    F = np.fft.fft2(img)                # 2D DFT, DC term at [0, 0]
    F_shifted = np.fft.fftshift(F)      # shift DC term to the center

    magnitude = np.abs(F_shifted)
    phase = np.angle(F_shifted)
    print("magnitude shape:", magnitude.shape, "| phase shape:", phase.shape)

    # Must un-shift before taking the inverse transform
    recon = np.fft.ifft2(np.fft.ifftshift(F_shifted))
    print("reconstruction matches original ->", np.allclose(recon.real, img, atol=1e-10))


# =====================================================================
# 6. Dict of Fourier coefficients: sort by magnitude, prune, sum energy
# =====================================================================
def demo_coefficient_dict():
    section("6. Coefficient dict: sort / prune / energy sum")

    rng = np.random.default_rng(0)
    coeffs = {n: complex(rng.standard_normal(), rng.standard_normal()) for n in range(-3, 4)}
    print("coeffs:", coeffs)

    # sort keys by |c_n|^2 descending (used in harmonic pruning)
    ranked = sorted(coeffs.keys(), key=lambda n: abs(coeffs[n]) ** 2, reverse=True)
    print("ranked by energy:", ranked)

    # dict comprehension to zero out everything not in a keep-set
    keep = set(ranked[:3])
    pruned = {n: (c if n in keep else 0j) for n, c in coeffs.items()}
    print("pruned:", pruned)

    # sum over a dict's values -- e.g. total energy
    total_energy = sum(abs(c) ** 2 for c in coeffs.values())
    print("total energy:", total_energy)


# =====================================================================
# 7. Fourier series reconstruction as a one-liner (generator + sum)
# =====================================================================
def demo_fourier_series_sum():
    section("7. Fourier series sum via generator expression")

    N = 3
    omega = 2 * np.pi
    rng = np.random.default_rng(1)
    coeffs = {n: complex(rng.standard_normal(), rng.standard_normal()) for n in range(-N, N + 1)}

    t = np.linspace(0, 1, 5)   # works for an array of times...
    f_hat = sum(coeffs[n] * np.exp(1j * n * omega * t) for n in range(-N, N + 1))
    print("f_hat(t array) shape/dtype:", f_hat.shape, f_hat.dtype)

    t_scalar = 0.3              # ...and for a single scalar time, unchanged
    f_hat_scalar = sum(coeffs[n] * np.exp(1j * n * omega * t_scalar) for n in range(-N, N + 1))
    print("f_hat(scalar t):", f_hat_scalar)


# =====================================================================
# 8. Formatting a results table (f-strings with alignment/precision)
# =====================================================================
def demo_table_formatting():
    section("8. f-string table formatting")

    rows = [(0.96, 210, 0.9601, 0.000123), (1.00, 301, 1.0000, 0.0)]
    print(f"{'Target Ratio':<13}| {'Harmonics Retained':<19}| {'Actual Energy Ratio':<20}| MSE")
    for r, count, ratio, mse in rows:
        print(f"{r:<13.2f}| {count:<19d}| {ratio:<20.4f}| {mse:.6e}")


# =====================================================================
# 9. Class boilerplate pattern used across these assignments
# =====================================================================
class Something:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.cache = {}          # empty dict to be filled in later

    def method(self, x):
        return self.a * x        # self. prefix on every attribute access

    @staticmethod
    def helper(x):
        return x ** 2


def demo_class_boilerplate():
    section("9. Class boilerplate pattern")

    obj = Something(a=2, b=5)
    
    print("obj.method(10) =", obj.method(10))
    print("Something.helper(4) =", Something.helper(4))
    

# =====================================================================
# 10. [:, None] / [None, :] and slice syntax, explained
# =====================================================================
def demo_none_indexing():
    section("10. [:, None] and [None, :] broadcasting indexing")

    a = np.array([10, 20, 30])          # shape (3,)
    print("a shape:", a.shape)

    # None (equivalently np.newaxis) inserts a new axis of length 1
    # at that position. It does NOT change the data, only the shape.
    col = a[:, None]                    # ':' keeps the existing axis, None adds a new one AFTER it
    print("a[:, None] shape:", col.shape)   # (3, 1) -- a column vector
    print(col)

    row = a[None, :]                    # None adds a new axis BEFORE the existing one
    print("a[None, :] shape:", row.shape)   # (1, 3) -- a row vector
    print(row)

    # These are equivalent to np.newaxis and to reshape:
    print(np.array_equal(a[:, None], a.reshape(3, 1)))   # True
    print(np.array_equal(a[None, :], a.reshape(1, 3)))   # True

    # Why it matters: multiplying (3,1) by (1,4) broadcasts into a (3,4) grid
    b = np.array([1, 2, 3, 4])          # shape (4,)
    grid = a[:, None] * b[None, :]      # (3,1) * (1,4) -> (3,4)
    print("outer-product grid shape:", grid.shape)
    print(grid)

    # General slice syntax reminder: start:stop:step
    x = np.arange(10)
    print("x[2:7]    =", x[2:7])        # indices 2,3,4,5,6 (stop excluded)
    print("x[:5]     =", x[:5])         # start defaults to 0
    print("x[5:]     =", x[5:])         # stop defaults to end
    print("x[::2]    =", x[::2])        # every 2nd element
    print("x[::-1]   =", x[::-1])       # reversed
    print("x[-3:]    =", x[-3:])        # last 3 elements

    # ':' alone in a multi-dim index means "take everything along this axis"
    M = np.arange(12).reshape(3, 4)
    print("M[:, 0]   =", M[:, 0])       # all rows, column 0 -> shape (3,)
    print("M[1, :]   =", M[1, :])       # row 1, all columns  -> shape (4,)
    print("M[:, 0:2] =\n", M[:, 0:2])   # all rows, columns 0-1 -> shape (3,2)
    
    
# =====================================================================
# 11. sorted(): single key, multi-key, mixed ascending/descending
# =====================================================================
def demo_sorted_multikey():
    section("11. sorted() with one, two, and three keys")

    data = [
        {"n": 3,  "energy": 5.0, "phase": 1.2},
        {"n": -1, "energy": 5.0, "phase": 0.3},
        {"n": 2,  "energy": 8.0, "phase": 2.1},
        {"n": 0,  "energy": 5.0, "phase": 0.3},
    ]

    # --- sorted() basics ---
    # sorted(iterable, key=..., reverse=...) returns a NEW sorted list;
    # it never mutates the original. list.sort() is the in-place version
    # with the same key/reverse arguments but no return value.
    nums = [3, 1, 4, 1, 5, 9, 2]
    print("sorted(nums)          =", sorted(nums))
    print("sorted(nums, reverse) =", sorted(nums, reverse=True))

    # --- one key ---
    # key=... takes a function applied to each element; sorting compares
    # the function's OUTPUT, not the elements themselves.
    by_energy = sorted(data, key=lambda d: d["energy"])
    print("sorted by energy asc:", [(d["n"], d["energy"]) for d in by_energy])

    by_energy_desc = sorted(data, key=lambda d: d["energy"], reverse=True)
    print("sorted by energy desc:", [(d["n"], d["energy"]) for d in by_energy_desc])

    # --- two keys ---
    # Return a TUPLE from the key function. Python compares tuples
    # lexicographically: first element decides, ties broken by the second.
    # Here: energy descending (primary), n ascending (secondary, tie-break)
    by_energy_then_n = sorted(data, key=lambda d: (-d["energy"], d["n"]))
    print("sorted by (-energy, n):", [(d["n"], d["energy"]) for d in by_energy_then_n])
    # Note the "-d['energy']" trick: negating a numeric key flips its
    # sort direction to descending WITHOUT needing reverse=True (which
    # would flip BOTH keys' direction, not just one).

    # --- three keys, mixed directions ---
    # energy descending, then phase ascending, then n descending.
    # Since reverse=True would reverse ALL three, negate the ones you
    # want descending and leave ascending ones alone.
    by_three_keys = sorted(
        data,
        key=lambda d: (-d["energy"], d["phase"], -d["n"])
    )
    print("sorted by (-energy, phase, -n):",
          [(d["n"], d["energy"], d["phase"]) for d in by_three_keys])

    # --- sorting a dict's keys by a computed value (like harmonic pruning) ---
    coeffs = {-2: 1 + 1j, -1: 3 + 0j, 0: 0.5j, 1: 2 - 2j, 2: 0.1 + 0.1j}
    ranked = sorted(coeffs.keys(), key=lambda n: abs(coeffs[n]) ** 2, reverse=True)
    print("harmonics ranked by energy:", ranked)

    # If you need a secondary tie-break on a dict-key sort, same tuple trick:
    # e.g. energy descending, then |n| ascending (favor low-order harmonics on ties)
    ranked_tiebreak = sorted(
        coeffs.keys(),
        key=lambda n: (-abs(coeffs[n]) ** 2, abs(n))
    )
    print("ranked with tie-break on |n|:", ranked_tiebreak)
    
# =====================================================================
# 12. Sorting a dict by value, keeping the top-k keys/indices
# =====================================================================
import heapq
from operator import itemgetter


def demo_topk_from_dict():
    section("12. Sorting a dict and keeping the k highest-value entries")

    coeffs = {-3: 0.2, -2: 1.5, -1: 4.0, 0: 9.0, 1: 3.5, 2: 6.0, 3: 0.1}
    k = 3

    # --- Approach A: sorted() on dict.items(), slice the top k ---
    # dict.items() gives (key, value) pairs; sort by the value (index 1).
    items_sorted = sorted(coeffs.items(), key=lambda kv: kv[1], reverse=True)
    top_k_items = items_sorted[:k]
    print("A) top-k (key, value) pairs:", top_k_items)
    top_k_keys_a = [key for key, val in top_k_items]
    print("A) top-k keys only:", top_k_keys_a)

    # --- Approach B: sorted() on dict.keys(), using the dict for lookup ---
    # Useful when you only care about keys/indices, not the paired values.
    keys_sorted = sorted(coeffs.keys(), key=lambda n: coeffs[n], reverse=True)
    top_k_keys_b = keys_sorted[:k]
    print("B) top-k keys:", top_k_keys_b)

    # --- Approach C: operator.itemgetter instead of a lambda ---
    # itemgetter(1) is a small speed/readability win over lambda kv: kv[1]
    # for large dicts, since it avoids a Python-level function call per item.
    items_sorted_c = sorted(coeffs.items(), key=itemgetter(1), reverse=True)
    print("C) top-k via itemgetter:", items_sorted_c[:k])

    # --- Approach D: heapq.nlargest -- best when k << len(dict) ---
    # sorted() sorts the WHOLE dict (O(n log n)); nlargest only needs
    # O(n log k), which matters if the dict is huge and k is small.
    top_k_items_d = heapq.nlargest(k, coeffs.items(), key=itemgetter(1))
    print("D) top-k via heapq.nlargest:", top_k_items_d)

    # nlargest also works directly on keys with a key= into the dict:
    top_k_keys_d = heapq.nlargest(k, coeffs, key=lambda n: coeffs[n])
    print("D) top-k keys via heapq.nlargest:", top_k_keys_d)

    # --- Approach E: numpy argsort -- when values live in a numpy array ---
    # If coeffs came from arrays instead of a dict (e.g. energies per index),
    # argsort gives the INDICES that would sort the array, ascending.
    energies = np.array([0.2, 1.5, 4.0, 9.0, 3.5, 6.0, 0.1])   # aligned with some index list
    order_desc = np.argsort(energies)[::-1]     # ascending indices, then reverse -> descending
    top_k_idx = order_desc[:k]
    print("E) top-k indices (numpy):", top_k_idx)
    print("E) top-k values (numpy):", energies[top_k_idx])

    # --- Rebuilding a "pruned" dict from a top-k key list (as in harmonic pruning) ---
    keep = set(top_k_keys_a)
    pruned = {key: (val if key in keep else 0.0) for key, val in coeffs.items()}
    print("pruned dict (non-top-k zeroed):", pruned)
    
    
# Key things to remember: dict.items() gives (key, value) tuples,
# so key=lambda kv: kv[1] (or itemgetter(1)) sorts by value while
# keeping the key attached — sorting dict.keys() alone needs the lambda
# to look the value back up in the dict (key=lambda n: coeffs[n]).
# sorted()[:k] is simplest and fine for small dicts;
# heapq.nlargest(k, ...) is the better choice when the dict is large
# and k is small, since it avoids sorting entries you're going to throw away.


if __name__ == "__main__":
    demo_complex_numbers()
    demo_trapezoid()
    demo_vectorized_masking()
    demo_broadcasting_outer_product()
    demo_fft2()
    demo_coefficient_dict()
    demo_fourier_series_sum()
    demo_table_formatting()
    demo_class_boilerplate()
