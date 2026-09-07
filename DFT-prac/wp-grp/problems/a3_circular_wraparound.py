"""
Practice online A3 -- circular convolution and the wraparound it causes.

Complete TODO 1--3. Do not modify the original offline files.

Run:

    python a3_circular_wraparound.py --digits 120 --seed 5 --engine fft
"""

import argparse
import os
import sys

import numpy as np

PRACTICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFLINE_DIR = os.path.dirname(PRACTICE_DIR)
sys.path.insert(0, OFFLINE_DIR)   # practice only: a real online sits next to the offline files

from bigmul import from_limbs, multiply_transform, to_limbs
from io_utils import random_decimal, write_report
from transforms import (ArbitraryLengthFFT, DFTAnalyzer, FFTTransformer,
                        next_power_of_two)


def short_length(len_a, len_b, engine):
    """Return a length that holds both inputs but NOT their full product."""
    # TODO 1 (student): max(len_a, len_b) is enough room for either input and
    # deliberately too little for the len_a + len_b - 1 product coefficients.
    # Round up to a power of two for the radix-2 engine; the arbitrary engine
    # can use it as is.

    # N = max(...)
    # return
    raise NotImplementedError("TODO 1: a deliberately short transform length")


def circular_convolve(a, b, N, engine):
    """Return the length-N CIRCULAR convolution of a and b as int64."""
    # TODO 2 (student): zero-pad both arrays to exactly N (no further),
    # transform, multiply pointwise, inverse-transform, keep the real part and
    # round. Nothing is cropped: all N slots are the answer.

    # fa = np.zeros(N, dtype=np.complex128); fa[:len(a)] = a
    # fb = ...
    # circular =
    # return np.rint(circular).astype(np.int64)
    raise NotImplementedError("TODO 2: circular convolution at length N")


def fold(linear, N):
    """Wrap a linear convolution onto N slots: out[m mod N] += linear[m]."""
    # TODO 3 (student): every coefficient past slot N-1 is added back onto
    # slot m mod N. Return an int64 array of length N.

    # out = np.zeros(N, dtype=np.int64)
    # for m in range(len(linear)):
    #     ...
    # return out
    raise NotImplementedError("TODO 3: fold the linear result onto N slots")


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
    """Provided runner: compare the short transform with the folded padded one."""
    engine = _make_engine(engine_name)
    text_a = random_decimal(digits, seed=seed)
    text_b = random_decimal(digits, seed=seed+1)
    _, la = to_limbs(text_a)
    _, lb = to_limbs(text_b)

    N_short = short_length(len(la), len(lb), engine)
    circular = circular_convolve(la, lb, N_short, engine)
    linear, N_full = multiply_transform(la, lb, engine)     # the offline's padded route
    folded = fold(linear, N_short)

    error = int(np.max(np.abs(circular - folded)))
    wrapped_number = from_limbs(1, circular)
    correct_number = from_limbs(1, linear)
    exact = str(int(text_a)*int(text_b))                     # the ONLY big-int use
    verdict = "MATCH" if error == 0 and correct_number == exact else "MISMATCH"

    os.makedirs(out_dir, exist_ok=True)
    write_report(os.path.join(out_dir, "report.txt"), [
        "Practice A3 -- circular convolution and wraparound",
        "digits per operand   : %d" % digits,
        "engine               : %s" % engine_name,
        "limbs of A / B       : %d / %d" % (len(la), len(lb)),
        "product needs        : %d coefficients" % len(linear),
        "short transform N    : %d  (too small -> wraps)" % N_short,
        "padded transform N   : %d" % N_full,
        "max |circular - folded linear| : %d" % error,
        "digits, correct product        : %d" % len(correct_number),
        "digits, wrapped 'product'      : %d" % len(wrapped_number),
        "wrapped 'product' starts with  : %s..." % wrapped_number[:24],
        "correct product starts with    : %s..." % correct_number[:24],
        "verification         : %s" % verdict,
    ])
    print("verification:", verdict, "(max |circular - folded| = %d)" % error)
    print("wrote outputs to", out_dir)
    if verdict != "MATCH":
        raise RuntimeError("circular convolution did not equal the folded linear one")


def main():
    parser = argparse.ArgumentParser(description="Circular convolution and wraparound")
    parser.add_argument("--digits", type=int, default=120)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--engine", choices=["dft", "fft", "arbitrary"], default="fft")
    parser.add_argument("--out-dir", default=os.path.join(PRACTICE_DIR, "outputs", "a3"))
    args = parser.parse_args()
    run(args.digits, args.seed, args.engine, args.out_dir)


if __name__ == "__main__":
    main()
