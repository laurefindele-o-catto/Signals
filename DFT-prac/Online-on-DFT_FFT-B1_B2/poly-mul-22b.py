"""
bigmul.py -- Task A: weighted polynomial product via FFT-based convolution.

Given
    P(x) = p_m x^m + ... + p_1 x + p_0
    Q(x) = q_n x^n + ... + q_1 x + q_0
    W    = [w_m, ..., w_0]              (one weight per coefficient of P)

compute
    R[k] = sum_{i=0}^{k} w_i * p_i * q_{k-i},   k = 0, 1, ..., m + n

All the actual transform work (DFT / radix-2 FFT / Bluestein) lives in
transforms.py. This file only reshapes the problem into a convolution
and drives that engine -- it never calls numpy.fft, scipy.fft,
numpy.convolve, or scipy.signal.
"""

import argparse
import numpy as np

from transforms import DFTAnalyzer, FFTTransformer, ArbitraryLengthFFT, next_power_of_two


ENGINES = {
    "dft": DFTAnalyzer,
    "fft": FFTTransformer,
    "arbitrary": ArbitraryLengthFFT,
}


# ---------------------------------------------------------------------------
# Core: FFT-based linear convolution
# ---------------------------------------------------------------------------
def fft_convolve(a, b, engine):
    """
    Linear convolution of two 1D sequences a and b, computed as a circular
    convolution after zero-padding far enough to avoid wraparound:

        c[k] = sum_{i+j=k} a[i] * b[j],   k = 0 .. len(a)+len(b)-2

    This is the standard "pad, transform, multiply pointwise, inverse
    transform" trick: convolution in the time domain is multiplication in
    the frequency domain.
    """
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)

    linear_len = len(a) + len(b) - 1

    # Padding to a power of two keeps DFTAnalyzer, FFTTransformer and
    # ArbitraryLengthFFT interchangeable (the plain FFT requires it; the
    # others don't mind it either).
    N = next_power_of_two(linear_len)

    a_pad = np.zeros(N, dtype=np.complex128)
    a_pad[: len(a)] = a
    b_pad = np.zeros(N, dtype=np.complex128)
    b_pad[: len(b)] = b

    A = engine.transform(a_pad)
    B = engine.transform(b_pad)
    C = A * B                     # pointwise multiply in the frequency domain
    c = engine.inverse(C)         # back to the time domain

    return c[:linear_len]         # drop the extra zero-padded tail


# ---------------------------------------------------------------------------
# Problem-specific reshaping: weighted product -> plain convolution
# ---------------------------------------------------------------------------
def weighted_polynomial_product(P_desc, Q_desc, W_desc, engine):
    """
    P_desc, Q_desc : coefficients in DESCENDING powers, e.g. [p_m, ..., p_0]
    W_desc         : weights aligned with P_desc, [w_m, ..., w_0]

    Returns R_desc : coefficients of R(x) in descending powers, length m+n+1,
    where
        R[k] = sum_{i=0}^{k} w_i * p_i * q_{k-i}

    Key observation (from the assignment hint): R[k] is exactly the linear
    convolution of the two coefficient sequences, PROVIDED we fold the
    weight into P first. Define, using ascending index i (i.e. coefficient
    of x^i):
        a_i = w_i * p_i          for i = 0 .. m
        b_j = q_j                for j = 0 .. n
    Then
        R[k] = sum_{i+j=k} a_i * b_j
    which is a single ordinary convolution of a and b -- one FFT-based
    convolution call, no per-k loop.
    """
    if len(P_desc) != len(W_desc):
        raise ValueError("W must have exactly one weight per coefficient of P")

    # descending powers -> ascending index (index i <-> x^i)
    P = np.asarray(P_desc, dtype=np.complex128)[::-1]
    Q = np.asarray(Q_desc, dtype=np.complex128)[::-1]
    W = np.asarray(W_desc, dtype=np.complex128)[::-1]

    a = W * P     # a_i = w_i * p_i
    b = Q         # b_j = q_j

    R = fft_convolve(a, b, engine)   # R[k], ascending index, k = 0 .. m+n

    # A real-valued round trip through FFT/IFFT leaves a tiny numerical
    # imaginary residue (~1e-14). Drop it once, at the very end -- never
    # discard .imag inside transforms.py itself.
    if np.allclose(R.imag, 0, atol=1e-6):
        R = R.real

    return R[::-1]      # back to descending powers, matching the inputs


# ---------------------------------------------------------------------------
# Brute-force reference (O(N^2), no FFT at all) -- used only for self-testing
# ---------------------------------------------------------------------------
def weighted_polynomial_product_bruteforce(P_desc, Q_desc, W_desc):
    P = np.asarray(P_desc, dtype=np.complex128)[::-1]
    Q = np.asarray(Q_desc, dtype=np.complex128)[::-1]
    W = np.asarray(W_desc, dtype=np.complex128)[::-1]
    m, n = len(P) - 1, len(Q) - 1

    R = np.zeros(m + n + 1, dtype=np.complex128)
    for k in range(m + n + 1):
        total = 0.0 + 0.0j
        for i in range(0, k + 1):
            if i <= m and (k - i) <= n:
                total += W[i] * P[i] * Q[k - i]
        R[k] = total
    return R.real[::-1]


def _self_test():
    rng = np.random.default_rng(0)
    for m, n in [(0, 0), (1, 2), (5, 3), (7, 9)]:
        P = rng.standard_normal(m + 1)
        Q = rng.standard_normal(n + 1)
        W = rng.standard_normal(m + 1)
        for name, cls in ENGINES.items():
            R_fast = weighted_polynomial_product(P, Q, W, cls())
            R_ref = weighted_polynomial_product_bruteforce(P, Q, W)
            assert np.max(np.abs(R_fast - R_ref)) < 1e-6, (name, m, n)
    print("self-test passed: dft / fft / arbitrary all match brute force")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def format_polynomial(coeffs_desc):
    m = len(coeffs_desc) - 1
    terms = []
    for power, c in zip(range(m, -1, -1), coeffs_desc):
        if power == 0:
            terms.append(f"{c:.6g}")
        elif power == 1:
            terms.append(f"{c:.6g}x")
        else:
            terms.append(f"{c:.6g}x^{power}")
    return " + ".join(terms)


def parse_coeffs(text):
    return [float(t) for t in text.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--P", type=parse_coeffs,
                         help="Comma-separated coefficients of P, descending powers, e.g. '1,2,3' for x^2+2x+3")
    parser.add_argument("--Q", type=parse_coeffs,
                         help="Comma-separated coefficients of Q, descending powers")
    parser.add_argument("--W", type=parse_coeffs,
                         help="Comma-separated weights aligned with P, descending powers")
    parser.add_argument("--engine", choices=ENGINES.keys(), default="fft",
                         help="Which transform engine to use (default: fft)")
    parser.add_argument("--self-test", action="store_true",
                         help="Run the brute-force cross-check instead of a single example")
    args = parser.parse_args()

    if args.self_test or not (args.P and args.Q and args.W):
        _self_test()
        if not (args.P and args.Q and args.W):
            print("\n(no --P/--Q/--W given -- running a demo example instead)\n")
            args.P, args.Q, args.W = [1, 2, 3], [1, 0, -1], [2, 1, 1]
        else:
            return

    engine = ENGINES[args.engine]()
    R = weighted_polynomial_product(args.P, args.Q, args.W, engine)

    print(f"P(x) = {format_polynomial(args.P)}")
    print(f"Q(x) = {format_polynomial(args.Q)}")
    print(f"W    = {args.W}")
    print(f"Engine: {args.engine}")
    print()
    print(f"R(x) = {format_polynomial(list(R))}")
    print("R coefficients (descending):", list(np.round(R, 6)))


if __name__ == "__main__":
    main()
