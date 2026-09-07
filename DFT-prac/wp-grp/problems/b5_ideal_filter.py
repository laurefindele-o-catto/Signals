"""
Practice online B5 -- ideal low-pass / high-pass split of an image spectrum.

Complete TODO 1--3. Do not modify the original offline files.

Run:

    python b5_ideal_filter.py --image images/skyline256.png --radius 20 --engine fft
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


def radial_mask(shape, radius):
    """Return a float mask that is 1 on bins within ``radius`` of zero frequency."""
    # TODO 1 (student): bin u of an H-point DFT has frequency min(u, H - u)
    # (the top half of the array holds the negative frequencies). Use that
    # wrapped distance on both axes, take the Euclidean distance and keep the
    # bins where it is <= radius. Return float64, not bool.

    # height, width = shape
    # u = np.arange(height); v = np.arange(width)
    # freq_rows = np.minimum(u, height - u)...[:, np.newaxis]
    # freq_cols = ...[np.newaxis, :]
    # distance = np.sqrt(...)
    # return (distance <= radius).astype(np.float64)
    raise NotImplementedError("TODO 1: radial mask with wrapped frequencies")


def split_plane(plane, radius, engine):
    """Return (low, high): the ideal low-pass and high-pass parts of one plane."""
    # TODO 2 (student): transform once, multiply the spectrum by the mask and by
    # (1 - mask), inverse-transform each product, keep the real parts.

    # spectrum = transform_2d(...)
    # mask = radial_mask(...)
    # low =
    # high =
    # return low, high
    raise NotImplementedError("TODO 2: low-pass and high-pass of one plane")


def split_image(image, radius, engine):
    """Apply split_plane to a grayscale or RGB image; return (low, high) with the original shape."""
    # TODO 3 (student): grayscale -> return the pair directly. RGB -> one pair
    # per channel; stack the three lows along the last axis, and the three
    # highs likewise, then return the two stacked arrays.

    # image = np.asarray(image, dtype=np.float64)
    # if image.ndim == 2:
    #     return
    # if image.ndim == 3 and image.shape[2] == 3:
    #     pairs =
    #     low =
    #     high =
    #     return low, high
    # raise ValueError("images must be grayscale or RGB")
    raise NotImplementedError("TODO 3: grayscale or RGB dispatch returning a pair")


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


def _reference_low(plane, radius, engine):
    """Provided oracle: the same low-pass built from a centred mask that is rolled back."""
    height, width = plane.shape
    rows = (np.arange(height) - height // 2)[:, np.newaxis]
    cols = (np.arange(width) - width // 2)[np.newaxis, :]
    centred = (np.sqrt(rows.astype(np.float64)**2 + cols.astype(np.float64)**2) <= radius)
    mask = np.roll(centred.astype(np.float64), (-(height // 2), -(width // 2)), axis=(0, 1))
    return inverse_2d(transform_2d(plane, engine) * mask, engine).real


def run(image_path, radius, engine_name, out_dir, color=False):
    """Provided runner: complementarity, an independent mask, and Parseval."""
    engine = _make_engine(engine_name)
    image = load_image(_resolve(image_path), as_gray=not color)
    low, high = split_image(image, radius, engine)

    complement_error = float(np.max(np.abs(low + high - image)))
    if image.ndim == 2:
        reference = _reference_low(image, radius, engine)
    else:
        reference = np.stack([_reference_low(image[:, :, c], radius, engine)
                              for c in range(image.shape[2])], axis=-1)
    mask_error = float(np.max(np.abs(low - reference)))
    # Parseval: the energy fraction kept must be the same in both domains.
    first = image if image.ndim == 2 else image[:, :, 0]
    spectrum = transform_2d(first, engine)
    kept = radial_mask(first.shape, radius)
    fraction_space = float(np.sum(split_plane(first, radius, engine)[0]**2) / np.sum(first**2))
    fraction_freq = float(np.sum(np.abs(spectrum * kept)**2) / np.sum(np.abs(spectrum)**2))
    parseval_error = abs(fraction_space - fraction_freq)
    verdict = ("MATCH" if max(complement_error, mask_error, parseval_error) <= 1e-9
               else "MISMATCH")

    os.makedirs(out_dir, exist_ok=True)
    save_image(low, os.path.join(out_dir, "low_pass.png"))
    save_image(np.clip(0.5 + 2.0 * high, 0.0, 1.0), os.path.join(out_dir, "high_pass.png"))
    save_comparison(
        [image, low, np.clip(0.5 + 2.0 * high, 0.0, 1.0)],
        ["original", "ideal low-pass (r = %d)" % radius, "ideal high-pass (rest)"],
        os.path.join(out_dir, "comparison.png"),
        suptitle="Ideal frequency split, engine=%s" % engine_name,
    )
    write_report(os.path.join(out_dir, "report.txt"), [
        "Practice B5 -- ideal low-pass / high-pass split",
        "image                : %s" % image_path,
        "image shape          : %s" % (image.shape,),
        "radius (bins)        : %d" % radius,
        "engine               : %s" % engine_name,
        "energy kept, spatial : %.6f" % fraction_space,
        "energy kept, spectral: %.6f  (Parseval)" % fraction_freq,
        "max |low + high - image| : %.3e" % complement_error,
        "max |low - reference low| : %.3e" % mask_error,
        "verification         : %s" % verdict,
    ])
    print("verification:", verdict, "(complement %.3e, mask %.3e, parseval %.3e)"
          % (complement_error, mask_error, parseval_error))
    print("wrote outputs to", out_dir)
    if verdict != "MATCH":
        raise RuntimeError("ideal filter split failed verification")
    return low, high


def main():
    parser = argparse.ArgumentParser(description="Ideal low-pass / high-pass split")
    parser.add_argument("--image", default="images/skyline256.png")
    parser.add_argument("--radius", type=int, default=20)
    parser.add_argument("--engine", choices=["dft", "fft", "arbitrary"], default="fft")
    parser.add_argument("--color", action="store_true")
    parser.add_argument("--out-dir", default=os.path.join(PRACTICE_DIR, "outputs", "b5"))
    args = parser.parse_args()
    run(args.image, args.radius, args.engine, args.out_dir, color=args.color)


if __name__ == "__main__":
    main()
