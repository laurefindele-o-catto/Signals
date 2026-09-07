"""
Practice online A2 -- a*b + c*d with exactly ONE inverse transform (linearity).

Complete TODO 1--3. Do not modify the original offline files.

Run:

    python a2_sum_of_products.py --digits 1500 --seed 11 --engine fft
"""

import argparse
import os
import sys

import numpy as np

PRACTICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFLINE_DIR = os.path.dirname(PRACTICE_DIR)
sys.path.insert(0, OFFLINE_DIR)   # practice only: a real online sits next to the offline files

from bigmul import BASE_DIGITS, from_limbs, to_limbs
from io_utils import random_decimal, write_report, write_text
from transforms import (ArbitraryLengthFFT, DFTAnalyzer, FFTTransformer,
                        next_power_of_two)


def choose_transform_length(len_a, len_b, len_c, len_d, engine):
    """Return one transform length that holds BOTH linear products."""
    # TODO 1 (student): a*b needs len_a + len_b - 1 slots and c*d needs
    # len_c + len_d - 1; one N must hold the longer of the two. Power of two
    # for the radix-2 engine, exact length for the arbitrary engine.
    need = max(len_a+len_b-1, len_c+len_d-1)
    if engine.name == "arbitrary":
        return need
    return next_power_of_two(need)


def sum_of_products_transform(a, b, c, d, engine):
    """Return (un-carried coefficients of a*b + c*d, transform length N)."""
    # TODO 2 (student): zero-pad all four limb arrays to N and transform each
    # once. Form A*B + C*D in the frequency domain (the DFT is linear, so the
    # sum of two products comes back from ONE inverse transform). Keep the
    # real part, crop to the longer linear length, round to int64.
    need = max(len(a)+len(b)-1, len(c)+len(d)-1)
    N = choose_transform_length(len(a), len(b), len(c), len(d), engine)
    spectra = []
    for limbs in (a, b, c, d):
        padded = np.zeros(N, dtype=np.complex128); padded[:len(limbs)] = limbs
        spectra.append(engine.transform(padded))
    A, B, C, D = spectra
    coeffs = engine.inverse(A*B + C*D).real[:need]
    return np.rint(coeffs).astype(np.int64), N


def sum_of_products(text_a, text_b, text_c, text_d, method):
    """Return (result string, N, limb counts) for four NON-NEGATIVE decimals."""
    # TODO 3 (student): convert all four with to_limbs; raise ValueError if any
    # sign is negative (a signed sum cannot be carried by from_limbs); convolve;
    # carry with a positive sign.
    limbs = []
    for text in (text_a, text_b, text_c, text_d):
        sign, arr = to_limbs(text)
        if sign < 0:
            raise ValueError("operands must be non-negative")
        limbs.append(arr)
    coeffs, N = sum_of_products_transform(*limbs, _make_engine(method))
    return from_limbs(1, coeffs), N, tuple(len(arr) for arr in limbs)


def _make_engine(name):
    """Provided command-line engine selection."""
    if name == "dft":
        return DFTAnalyzer()
    if name == "fft":
        return FFTTransformer()
    if name == "arbitrary":
        return ArbitraryLengthFFT()
    raise ValueError("unknown engine: %r" % name)


def run(digits, seed, engine_name, out_dir):
    """Provided runner: four reproducible operands, verify, write outputs."""
    texts = [random_decimal(digits, seed=seed+i) for i in range(4)]
    result, N, limbs = sum_of_products(*texts, engine_name)

    a, b, c, d = (int(t) for t in texts)                  # the ONLY big-int use
    expected = str(a*b + c*d)
    verdict = "MATCH" if result == expected else "MISMATCH"

    os.makedirs(out_dir, exist_ok=True)
    write_text(os.path.join(out_dir, "result.txt"), result)
    write_report(os.path.join(out_dir, "report.txt"), [
        "Practice A2 -- a*b + c*d with one inverse transform",
        "digits per operand   : %d" % digits,
        "engine               : %s" % engine_name,
        "base                 : 10^%d" % BASE_DIGITS,
        "limbs of A/B/C/D     : %d / %d / %d / %d" % limbs,
        "transform length N   : %d" % N,
        "digits of result     : %d" % len(result),
        "verification         : %s" % verdict,
    ])
    print("verification:", verdict)
    print("wrote outputs to", out_dir)
    if verdict != "MATCH":
        raise RuntimeError("sum of products did not match the reference")
    return result


def main():
    parser = argparse.ArgumentParser(description="a*b + c*d through one inverse transform")
    parser.add_argument("--digits", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--engine", choices=["dft", "fft", "arbitrary"], default="fft")
    parser.add_argument("--out-dir", default=os.path.join(PRACTICE_DIR, "outputs", "a2"))
    args = parser.parse_args()
    run(args.digits, args.seed, args.engine, args.out_dir)


if __name__ == "__main__":
    main()
