"""
Practice online A1 -- product of THREE big integers with one inverse transform.

Complete TODO 1--3. Do not modify the original offline files.

Run:

    python a1_triple_product.py --input inputs/3.txt --third-digits 2000 --engine fft
"""

import argparse
import os
import sys

import numpy as np

PRACTICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFLINE_DIR = os.path.dirname(PRACTICE_DIR)
sys.path.insert(0, OFFLINE_DIR)   # practice only: a real online sits next to the offline files

from bigmul import from_limbs, to_limbs
from io_utils import random_decimal, read_operands, write_report, write_text
from transforms import (ArbitraryLengthFFT, DFTAnalyzer, FFTTransformer,
                        next_power_of_two)

# A coefficient of a triple product is a sum of about n^2/2 products of THREE
# limbs, so it can reach (n^2/2)(B-1)^3. With B = 10^4 that leaves the 2^53
# mantissa for the provided inputs; B = 10^2 keeps every coefficient exact.
TRIPLE_BASE_DIGITS = 2


def choose_transform_length(len_a, len_b, len_c, engine):
    """Return the transform length for the linear convolution of three limb arrays."""
    # TODO 1 (student): a product of three polynomials with n, q and r
    # coefficients has n + q + r - 2 coefficients. The radix-2 engine needs a
    # power of two; the arbitrary engine can use the exact length.

    # need =
    # if engine.name == "arbitrary":
    #     return
    # return
    raise NotImplementedError("TODO 1: choose the transform length for three factors")


def multiply_three_transform(a, b, c, engine):
    """Return (un-carried coefficients of a*b*c, transform length N)."""
    # TODO 2 (student): zero-pad the three limb arrays to N, transform each one
    # once, multiply the three spectra pointwise, inverse-transform exactly
    # once, keep the real part, crop to the linear length and round to int64.

    # need =
    # N = choose_transform_length(...)
    # fa = np.zeros(N, dtype=np.complex128); fa[:len(a)] = a
    # ... same for fb, fc
    # spectrum =
    # coeffs =
    # return np.rint(coeffs).astype(np.int64), N
    raise NotImplementedError("TODO 2: multiply three limb arrays through the spectrum")


def multiply_three(text_a, text_b, text_c, method):
    """Return (product string, N, (limbs_a, limbs_b, limbs_c))."""
    # TODO 3 (student): convert every operand with TRIPLE_BASE_DIGITS, keep the
    # signs aside, convolve, carry with the SAME base and re-attach the sign.

    # sign_a, la = to_limbs(text_a, TRIPLE_BASE_DIGITS)
    # ...
    # coeffs, N = multiply_three_transform(la, lb, lc, _make_engine(method))
    # product = from_limbs(..., coeffs, TRIPLE_BASE_DIGITS)
    # return product, N, (len(la), len(lb), len(lc))
    raise NotImplementedError("TODO 3: limbs in, product string out")


def _make_engine(name):
    """Provided command-line engine selection."""
    if name == "dft":
        return DFTAnalyzer()
    if name == "fft":
        return FFTTransformer()
    if name == "arbitrary":
        return ArbitraryLengthFFT()
    raise ValueError("unknown engine: %r" % name)


def _resolve(path):
    """Provided: paths such as inputs/3.txt are relative to the offline folder."""
    return path if os.path.isabs(path) else os.path.join(OFFLINE_DIR, path)


def run(input_path, third_digits, engine_name, out_dir):
    """Provided runner: multiply, verify with Python's integers, write outputs."""
    text_a, text_b = read_operands(_resolve(input_path))
    text_c = random_decimal(third_digits, seed=third_digits)
    product, N, limbs = multiply_three(text_a, text_b, text_c, engine_name)

    expected = str(int(text_a)*int(text_b)*int(text_c))   # the ONLY big-int use
    verdict = "MATCH" if product == expected else "MISMATCH"

    os.makedirs(out_dir, exist_ok=True)
    write_text(os.path.join(out_dir, "product.txt"), product)
    digits = tuple(len(t.lstrip("+-")) for t in (text_a, text_b, text_c))
    write_report(os.path.join(out_dir, "report.txt"), [
        "Practice A1 -- triple product by spectral convolution",
        "input file           : %s" % input_path,
        "engine               : %s" % engine_name,
        "digits of A / B / C  : %d / %d / %d" % digits,
        "base                 : 10^%d" % TRIPLE_BASE_DIGITS,
        "limbs of A / B / C   : %d / %d / %d" % limbs,
        "transform length N   : %d" % N,
        "digits of product    : %d" % len(product.lstrip("-")),
        "verification         : %s" % verdict,
    ])
    print("verification:", verdict)
    print("wrote outputs to", out_dir)
    if verdict != "MATCH":
        raise RuntimeError("triple product did not match the reference")
    return product


def main():
    parser = argparse.ArgumentParser(description="Product of three big integers")
    parser.add_argument("--input", default="inputs/3.txt")
    parser.add_argument("--third-digits", type=int, default=2000)
    parser.add_argument("--engine", choices=["dft", "fft", "arbitrary"], default="fft")
    parser.add_argument("--out-dir", default=os.path.join(PRACTICE_DIR, "outputs", "a1"))
    args = parser.parse_args()
    run(args.input, args.third_digits, args.engine, args.out_dir)


if __name__ == "__main__":
    main()
