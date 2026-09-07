"""
Practice online A5 -- a^4 by squaring twice, one forward transform per square.

Complete TODO 1--3. Do not modify the original offline files.

Run:

    python a5_fourth_power.py --input inputs/3.txt --engine fft
"""

import argparse
import os
import sys

import numpy as np

PRACTICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFLINE_DIR = os.path.dirname(PRACTICE_DIR)
sys.path.insert(0, OFFLINE_DIR)   # practice only: a real online sits next to the offline files

from bigmul import BASE_DIGITS, from_limbs, to_limbs
from io_utils import read_operands, write_report, write_text
from transforms import (ArbitraryLengthFFT, DFTAnalyzer, FFTTransformer,
                        next_power_of_two)


def square_length(n, engine):
    """Return the transform length for squaring an n-limb number."""
    # TODO 1 (student): a square has 2n - 1 coefficients. Power of two for
    # the radix-2 engine, exact for the arbitrary engine.

    # need =
    # return
    raise NotImplementedError("TODO 1: transform length for a square")


def square_transform(a, engine):
    """Return (un-carried coefficients of a*a, N) using ONE forward transform."""
    # TODO 2 (student): zero-pad to N, transform once, multiply the spectrum
    # by itself, inverse-transform, keep the real part, crop to 2n - 1 and
    # round to int64. Two forward transforms here is wrong: it is one.

    # need =
    # N =
    # fa = ...
    # A = engine.transform(fa)
    # coeffs =
    # return np.rint(coeffs).astype(np.int64), N
    raise NotImplementedError("TODO 2: square through ONE forward transform")


def fourth_power(text, method):
    """Return (a^4 as a string, (N1, N2), (limbs of a, limbs of a^2))."""
    # TODO 3 (student): square the limbs, then CARRY the result into a proper
    # number (from_limbs) and split it into limbs again (to_limbs) BEFORE the
    # second squaring -- un-carried coefficients are far bigger than the base
    # and squaring them would overflow the double mantissa. An even power is
    # never negative, so the sign is +1 throughout.

    # _, limbs = to_limbs(text)
    # engine = _make_engine(method)
    # square, N1 = square_transform(limbs, engine)
    # _, reduced = to_limbs(from_limbs(1, square))
    # fourth, N2 = ...
    # return from_limbs(1, fourth), (N1, N2), (len(limbs), len(reduced))
    raise NotImplementedError("TODO 3: square, carry, square again")


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


def run(input_path, engine_name, out_dir):
    """Provided runner: fourth power of the first operand, verified exactly."""
    text, _ = read_operands(_resolve(input_path))
    result, (N1, N2), (limbs1, limbs2) = fourth_power(text, engine_name)

    expected = str(int(text)**4)                             # the ONLY big-int use
    verdict = "MATCH" if result == expected else "MISMATCH"

    os.makedirs(out_dir, exist_ok=True)
    write_text(os.path.join(out_dir, "result.txt"), result)
    write_report(os.path.join(out_dir, "report.txt"), [
        "Practice A5 -- fourth power by repeated squaring",
        "input file           : %s (first operand)" % input_path,
        "engine               : %s" % engine_name,
        "digits of A          : %d" % len(text.lstrip("+-")),
        "base                 : 10^%d" % BASE_DIGITS,
        "limbs of A / A^2     : %d / %d" % (limbs1, limbs2),
        "transform lengths    : %d then %d" % (N1, N2),
        "digits of A^4        : %d" % len(result),
        "verification         : %s" % verdict,
    ])
    print("verification:", verdict)
    print("wrote outputs to", out_dir)
    if verdict != "MATCH":
        raise RuntimeError("fourth power did not match the reference")
    return result


def main():
    parser = argparse.ArgumentParser(description="Fourth power by repeated squaring")
    parser.add_argument("--input", default="inputs/3.txt")
    parser.add_argument("--engine", choices=["dft", "fft", "arbitrary"], default="fft")
    parser.add_argument("--out-dir", default=os.path.join(PRACTICE_DIR, "outputs", "a5"))
    args = parser.parse_args()
    run(args.input, args.engine, args.out_dir)


if __name__ == "__main__":
    main()
