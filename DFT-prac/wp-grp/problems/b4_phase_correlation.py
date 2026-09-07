"""
Practice online B4 -- phase correlation: find how far one image was shifted.

Complete TODO 1--3. Do not modify the original offline files.

Run:

    python b4_phase_correlation.py --image images/skyline256.png --rows 37 --cols -21 --engine fft
"""

import argparse
import os
import sys

import numpy as np

PRACTICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFLINE_DIR = os.path.dirname(PRACTICE_DIR)
sys.path.insert(0, OFFLINE_DIR)   # practice only: a real online sits next to the offline files

from image_conv import inverse_2d, transform_2d
from image_utils import load_image, save_comparison, save_image
from io_utils import write_report
from transforms import ArbitraryLengthFFT, DFTAnalyzer, FFTTransformer


def unit_cross_power(spectrum_a, spectrum_b, epsilon=1e-12):
    """Return conj(A) * B normalised to unit magnitude, bin by bin."""
    # TODO 1 (student): cross = conj(A) * B. Where |cross| > epsilon divide by
    # |cross| so only the phase difference survives; every other bin becomes 0.
    # Never divide by a vanishing magnitude.

    # cross =
    # magnitude = np.abs(cross)
    # result = np.zeros_like(cross)
    # reliable =
    # result[reliable] =
    # return result
    raise NotImplementedError("TODO 1: unit cross-power spectrum")


def correlation_surface(plane_a, plane_b, engine):
    """Return the phase-correlation surface of two planes (a peak marks the shift)."""
    # TODO 2 (student): transform both planes (no padding: the shift wraps),
    # take the unit cross-power spectrum, inverse-transform ONCE, keep the
    # real part.

    # spectrum_a =
    # spectrum_b =
    # return inverse_2d(...).real
    raise NotImplementedError("TODO 2: correlation surface of two planes")


def find_shift(image_a, image_b, engine):
    """Return (shift_rows, shift_cols, surface) such that image_b = roll(image_a, shift)."""
    # TODO 3 (student): grayscale -> one surface; RGB -> the SUM of the three
    # channel surfaces. The peak's (row, col) is the shift; an index above
    # H//2 (or W//2) is really negative: subtract H (or W).

    # image_a = np.asarray(image_a, dtype=np.float64)
    # image_b = np.asarray(image_b, dtype=np.float64)
    # if image_a.shape != image_b.shape:
    #     raise ValueError("the two images must have the same shape")
    # if image_a.ndim == 2:
    #     surface =
    # elif image_a.ndim == 3 and image_a.shape[2] == 3:
    #     surface =
    # else:
    #     raise ValueError("images must be grayscale or RGB")
    # height, width = surface.shape
    # row, col = np.unravel_index(int(np.argmax(surface)), surface.shape)
    # ... unwrap ...
    # return row, col, surface
    raise NotImplementedError("TODO 3: dispatch, locate the peak, unwrap it")


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
    """Provided: paths such as images/skyline256.png are relative to the offline folder."""
    return path if os.path.isabs(path) else os.path.join(OFFLINE_DIR, path)


def run(image_path, shift_rows, shift_cols, engine_name, out_dir, color=False):
    """Provided runner: shift with np.roll, recover the shift, check the peak."""
    engine = _make_engine(engine_name)
    image_a = load_image(_resolve(image_path), as_gray=not color)
    image_b = np.roll(image_a, (shift_rows, shift_cols), axis=(0, 1))
    found_rows, found_cols, surface = find_shift(image_a, image_b, engine)

    planes = 1 if image_a.ndim == 2 else image_a.shape[2]
    height, width = surface.shape
    ideal = np.zeros((height, width))                       # an exact shift gives a unit impulse
    ideal[shift_rows % height, shift_cols % width] = 1.0
    maximum_error = float(np.max(np.abs(surface / planes - ideal)))
    verdict = ("MATCH" if (found_rows, found_cols) == (shift_rows, shift_cols)
               and maximum_error <= 1e-9 else "MISMATCH")

    os.makedirs(out_dir, exist_ok=True)
    peak_view = np.clip(surface / planes, 0.0, 1.0)
    save_image(peak_view, os.path.join(out_dir, "correlation_surface.png"))
    save_comparison(
        [image_a, image_b, peak_view],
        ["image A", "image B = roll(A, (%d, %d))" % (shift_rows, shift_cols),
         "phase correlation (peak at shift)"],
        os.path.join(out_dir, "comparison.png"),
        suptitle="Phase correlation, engine=%s" % engine_name,
    )
    write_report(os.path.join(out_dir, "report.txt"), [
        "Practice B4 -- shift recovery by phase correlation",
        "image                : %s" % image_path,
        "image shape          : %s" % (image_a.shape,),
        "true shift           : (%d, %d)" % (shift_rows, shift_cols),
        "detected shift       : (%d, %d)" % (found_rows, found_cols),
        "peak height          : %.6f  (1.0 for an exact circular shift)" % (float(surface.max()) / planes),
        "engine               : %s" % engine_name,
        "max |surface - unit impulse| : %.3e" % maximum_error,
        "verification         : %s" % verdict,
    ])
    print("verification:", verdict, "(detected (%d, %d), expected (%d, %d))"
          % (found_rows, found_cols, shift_rows, shift_cols))
    print("wrote outputs to", out_dir)
    if verdict != "MATCH":
        raise RuntimeError("phase correlation did not recover the shift")
    return found_rows, found_cols


def main():
    parser = argparse.ArgumentParser(description="Recover an image shift by phase correlation")
    parser.add_argument("--image", default="images/skyline256.png")
    parser.add_argument("--rows", type=int, default=37)
    parser.add_argument("--cols", type=int, default=-21)
    parser.add_argument("--engine", choices=["dft", "fft", "arbitrary"], default="fft")
    parser.add_argument("--color", action="store_true")
    parser.add_argument("--out-dir", default=os.path.join(PRACTICE_DIR, "outputs", "b4"))
    args = parser.parse_args()
    run(args.image, args.rows, args.cols, args.engine, args.out_dir, color=args.color)


if __name__ == "__main__":
    main()
