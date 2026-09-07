"""
Practice online B1 -- shift an image by multiplying its spectrum by a phase ramp.

Complete TODO 1--3. Do not modify the original offline files.

Run:

    python b1_spectral_shift.py --image images/skyline256.png --rows 37 --cols -21 --engine fft
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


def phase_ramp(shape, shift_rows, shift_cols):
    """Return the spectrum multiplier that delays a plane by (shift_rows, shift_cols)."""
    # TODO 1 (student): delaying x[n] by s multiplies X[u] by exp(-2 pi j u s / N).
    # In 2D the exponent adds: exp(-2 pi j (u*shift_rows/H + v*shift_cols/W))
    # with u = 0..H-1 down the rows and v = 0..W-1 along the columns.

    # height, width = shape
    # u = np.arange(height, dtype=np.float64)[:, np.newaxis]
    # v = np.arange(width, dtype=np.float64)[np.newaxis, :]
    # return np.exp(...)
    raise NotImplementedError("TODO 1: build the phase ramp")


def shift_plane(plane, shift_rows, shift_cols, engine):
    """Circularly shift one plane through the frequency domain."""
    # TODO 2 (student): transform the plane (no padding: the shift is meant to
    # wrap), multiply by the ramp, inverse-transform ONCE, keep the real part.

    # spectrum = transform_2d(...) * phase_ramp(...)
    # return inverse_2d(...).real
    raise NotImplementedError("TODO 2: shift one plane through its spectrum")


def shift_image(image, shift_rows, shift_cols, engine):
    """Apply shift_plane to a grayscale or RGB image, preserving its shape."""
    # TODO 3 (student): one plane for grayscale; three planes stacked back
    # along the last axis for RGB.

    # image = np.asarray(image, dtype=np.float64)
    # if image.ndim == 2:
    #     return
    # if image.ndim == 3 and image.shape[2] == 3:
    #     planes =
    #     return
    # raise ValueError("images must be grayscale or RGB")
    raise NotImplementedError("TODO 3: grayscale or RGB dispatch")


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
    """Provided runner: compare the spectral shift with np.roll."""
    engine = _make_engine(engine_name)
    image = load_image(_resolve(image_path), as_gray=not color)
    result = shift_image(image, shift_rows, shift_cols, engine)

    reference = np.roll(image, (shift_rows, shift_cols), axis=(0, 1))   # the oracle
    maximum_error = float(np.max(np.abs(result - reference)))
    verdict = "MATCH" if maximum_error <= 1e-9 else "MISMATCH"

    os.makedirs(out_dir, exist_ok=True)
    save_image(result, os.path.join(out_dir, "shifted.png"))
    save_comparison(
        [image, result, reference],
        ["original", "spectral shift (%d, %d)" % (shift_rows, shift_cols), "np.roll reference"],
        os.path.join(out_dir, "comparison.png"),
        suptitle="Shift through the frequency domain, engine=%s" % engine_name,
    )
    write_report(os.path.join(out_dir, "report.txt"), [
        "Practice B1 -- shift by a spectral phase ramp",
        "image                : %s" % image_path,
        "image shape          : %s" % (image.shape,),
        "shift (rows, cols)   : (%d, %d)" % (shift_rows, shift_cols),
        "engine               : %s" % engine_name,
        "max |spectral - np.roll| : %.3e" % maximum_error,
        "verification         : %s" % verdict,
    ])
    print("verification:", verdict, "(max error %.3e)" % maximum_error)
    print("wrote outputs to", out_dir)
    if verdict != "MATCH":
        raise RuntimeError("spectral shift did not match np.roll")
    return result


def main():
    parser = argparse.ArgumentParser(description="Shift an image through its spectrum")
    parser.add_argument("--image", default="images/skyline256.png")
    parser.add_argument("--rows", type=int, default=37)
    parser.add_argument("--cols", type=int, default=-21)
    parser.add_argument("--engine", choices=["dft", "fft", "arbitrary"], default="fft")
    parser.add_argument("--color", action="store_true")
    parser.add_argument("--out-dir", default=os.path.join(PRACTICE_DIR, "outputs", "b1"))
    args = parser.parse_args()
    run(args.image, args.rows, args.cols, args.engine, args.out_dir, color=args.color)


if __name__ == "__main__":
    main()
