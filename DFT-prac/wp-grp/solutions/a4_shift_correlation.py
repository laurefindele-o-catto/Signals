"""
Practice online A4 -- cross-correlation through the spectrum: find a shift.

Complete TODO 1--3. Do not modify the original offline files.

Run:

    python a4_shift_correlation.py --digits 400 --shift 7 --engine fft
"""

import argparse
import os
import sys

import numpy as np

PRACTICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFLINE_DIR = os.path.dirname(PRACTICE_DIR)
sys.path.insert(0, OFFLINE_DIR)   # practice only: a real online sits next to the offline files

from bigmul import BASE_DIGITS, to_limbs
from io_utils import random_decimal, write_report
from transforms import (ArbitraryLengthFFT, DFTAnalyzer, FFTTransformer,
                        next_power_of_two)


def choose_transform_length(len_a, len_b, engine):
    """Return a transform length in which every lag has its own slot."""
    # TODO 1 (student): the lags run from -(len_a - 1) to len_b - 1, which is
    # len_a + len_b - 1 values -- the same rule as linear convolution. Power of
    # two for the radix-2 engine, exact for the arbitrary engine.
    need = len_a+len_b-1
    if engine.name == "arbitrary":
        return need
    return next_power_of_two(need)


def cross_correlate(a, b, engine):
    """Return (c, N): c[k] = sum_i a[i] * b[i + k], lag k stored at index k mod N."""
    # TODO 2 (student): zero-pad both arrays to N and transform. Correlation is
    # convolution with a time-reversed a, and for a real sequence reversal is
    # conjugation in the frequency domain: multiply conj(A) by B, inverse-
    # transform ONCE, keep the real part and round to int64.
    N = choose_transform_length(len(a), len(b), engine)
    fa = np.zeros(N, dtype=np.complex128); fa[:len(a)] = a
    fb = np.zeros(N, dtype=np.complex128); fb[:len(b)] = b
    spectrum = np.conj(engine.transform(fa))*engine.transform(fb)
    c = engine.inverse(spectrum).real
    return np.rint(c).astype(np.int64), N


def find_shift(text_a, text_b, method):
    """Return (shift, peak, N): the lag with the largest correlation."""
    # TODO 3 (student): use the limb magnitudes only, correlate, take the
    # index of the maximum, and unwrap it: an index above N // 2 is really a
    # negative lag (index - N). The peak value is the correlation at that lag.
    _, la = to_limbs(text_a)
    _, lb = to_limbs(text_b)
    c, N = cross_correlate(la, lb, _make_engine(method))
    k = int(np.argmax(c))
    if k > N//2:
        k -= N
    return k, int(c.max()), N


def _make_engine(name):
    """Provided command-line engine selection."""
    if name == "dft":
        return DFTAnalyzer()
    if name == "fft":
        return FFTTransformer()
    if name == "arbitrary":
        return ArbitraryLengthFFT()
    raise ValueError("unknown engine: %r" % name)


def run(digits, shift, engine_name, out_dir):
    """Provided runner: b = a * BASE^shift, so b's limbs are a's moved up by shift."""
    text_a = random_decimal(digits, seed=digits)
    text_b = text_a + "0"*(BASE_DIGITS*shift)
    found, peak, N = find_shift(text_a, text_b, engine_name)

    # Independent oracle: the defining double loop over every lag.
    _, la = to_limbs(text_a)
    _, lb = to_limbs(text_b)
    spectral, _ = cross_correlate(la, lb, _make_engine(engine_name))
    error = 0
    for k in range(-(len(la)-1), len(lb)):
        direct = 0
        for i in range(len(la)):
            j = i + k
            if 0 <= j < len(lb):
                direct += int(la[i])*int(lb[j])
        error = max(error, abs(int(spectral[k % N]) - direct))
    energy = int(np.sum(la.astype(np.int64)**2))
    verdict = "MATCH" if error == 0 and found == shift and peak == energy else "MISMATCH"

    os.makedirs(out_dir, exist_ok=True)
    write_report(os.path.join(out_dir, "report.txt"), [
        "Practice A4 -- shift detection by spectral cross-correlation",
        "digits of A          : %d" % digits,
        "engine               : %s" % engine_name,
        "limbs of A / B       : %d / %d" % (len(la), len(lb)),
        "transform length N   : %d" % N,
        "true shift (limbs)   : %d" % shift,
        "detected shift       : %d" % found,
        "peak correlation     : %d  (energy of A = %d)" % (peak, energy),
        "max |spectral - direct| over all lags : %d" % error,
        "verification         : %s" % verdict,
    ])
    print("verification:", verdict, "(detected shift %d, expected %d)" % (found, shift))
    print("wrote outputs to", out_dir)
    if verdict != "MATCH":
        raise RuntimeError("cross-correlation did not match the reference")


def main():
    parser = argparse.ArgumentParser(description="Shift detection by cross-correlation")
    parser.add_argument("--digits", type=int, default=400)
    parser.add_argument("--shift", type=int, default=7)
    parser.add_argument("--engine", choices=["dft", "fft", "arbitrary"], default="fft")
    parser.add_argument("--out-dir", default=os.path.join(PRACTICE_DIR, "outputs", "a4"))
    args = parser.parse_args()
    run(args.digits, args.shift, args.engine, args.out_dir)


if __name__ == "__main__":
    main()
