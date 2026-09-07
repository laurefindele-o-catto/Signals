"""
Practice online B2 -- unsharp masking (sharpening) in the frequency domain.

Complete TODO 1--3. Do not modify the original offline files.

Run:

    python b2_unsharp_mask.py --image images/skyline256.png --kernel-size 21 --amount 1.5 --engine fft
"""

import argparse
import os
import sys

import numpy as np

PRACTICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFLINE_DIR = os.path.dirname(PRACTICE_DIR)
sys.path.insert(0, OFFLINE_DIR)   # practice only: a real online sits next to the offline files

from image_conv import convolve_image, inverse_2d, transform_2d
from image_utils import load_image, make_kernel, save_comparison, save_image
from io_utils import write_report
from transforms import (ArbitraryLengthFFT, DFTAnalyzer, FFTTransformer,
                        next_power_of_two)


def choose_transform_shape(image_shape, kernel_shape, engine):
    """Return the padded 2D linear-convolution transform shape."""
    # TODO 1 (student): full convolution size per axis, then a power of two
    # per axis for the radix-2 engine only.
    rows, cols = image_shape
    krows, kcols = kernel_shape
    full_height, full_width = rows+krows-1, cols+kcols-1
    if engine.name == "fft":
        full_height, full_width = next_power_of_two(full_height), next_power_of_two(full_width)
    return full_height, full_width


def centred_delta_spectrum(transform_shape, kernel_shape):
    """Provided helper: DFT of an impulse placed at the kernel's centre."""
    height, width = transform_shape
    centre_row = kernel_shape[0] // 2
    centre_column = kernel_shape[1] // 2
    vertical = np.arange(height, dtype=np.float64)[:, np.newaxis]
    horizontal = np.arange(width, dtype=np.float64)[np.newaxis, :]
    phase = vertical * centre_row / height + horizontal * centre_column / width
    return np.exp(-2j * np.pi * phase)


def _pad_top_left(array, shape):
    """Provided helper: place a 2D array at the origin of a complex array."""
    result = np.zeros(shape, dtype=np.complex128)
    result[:array.shape[0], :array.shape[1]] = array
    return result


def sharpen_plane(plane, kernel, amount, engine):
    """Return plane + amount * (plane - blur(plane)) built in the frequency domain."""
    # TODO 2 (student): pad the plane and the kernel, transform both, take the
    # centred delta, form P * (delta + amount * (delta - G)), inverse-transform
    # ONCE, keep the real part and crop the usual linear-convolution window.
    plane = np.asarray(plane, dtype=np.float64)
    shape = choose_transform_shape(plane.shape, kernel.shape, engine)
    plane_spectrum = transform_2d(_pad_top_left(plane, shape), engine)
    kernel_spectrum = transform_2d(_pad_top_left(kernel, shape), engine)
    delta_spectrum = centred_delta_spectrum(shape, kernel.shape)
    combined = plane_spectrum*(delta_spectrum + amount*(delta_spectrum - kernel_spectrum))
    full = inverse_2d(combined, engine).real
    row, column = kernel.shape[0] // 2, kernel.shape[1] // 2
    return full[row:row + plane.shape[0], column:column + plane.shape[1]]


def sharpen_image(image, kernel, amount, engine):
    """Apply sharpen_plane to a grayscale or RGB image, preserving its shape."""
    # TODO 3 (student): one plane for grayscale; three planes stacked back
    # along the last axis for RGB.
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 2:
        return sharpen_plane(image, kernel, amount, engine)
    if image.ndim == 3 and image.shape[2] == 3:
        planes = [sharpen_plane(image[:, :, c], kernel, amount, engine)
                  for c in range(image.shape[2])]
        return np.stack(planes, axis=-1)
    raise ValueError("images must be grayscale or RGB")


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


def run(image_path, kernel_size, amount, engine_name, out_dir, color=False):
    """Provided runner: compare against the spatial-domain unsharp mask."""
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel size must be a positive odd integer")
    engine = _make_engine(engine_name)
    image = load_image(_resolve(image_path), as_gray=not color)
    kernel = make_kernel("gaussian", size=kernel_size)
    result = sharpen_image(image, kernel, amount, engine)

    blurred = convolve_image(image, kernel, engine)                 # the oracle route
    reference = image + amount * (image - blurred)
    maximum_error = float(np.max(np.abs(result - reference)))
    verdict = "MATCH" if maximum_error <= 1e-9 else "MISMATCH"

    os.makedirs(out_dir, exist_ok=True)
    save_image(result, os.path.join(out_dir, "sharpened.png"))
    save_image(np.clip(0.5 + 2.0 * (image - blurred), 0.0, 1.0),
               os.path.join(out_dir, "detail.png"))
    save_comparison(
        [image, blurred, result],
        ["original", "Gaussian blur", "sharpened (amount %.2f)" % amount],
        os.path.join(out_dir, "comparison.png"),
        suptitle="Unsharp mask: Gaussian %dx%d, engine=%s" % (kernel_size, kernel_size, engine_name),
    )
    transform_shape = choose_transform_shape(image.shape[:2], kernel.shape, engine)
    write_report(os.path.join(out_dir, "report.txt"), [
        "Practice B2 -- unsharp masking in the frequency domain",
        "image                : %s" % image_path,
        "image shape          : %s" % (image.shape,),
        "kernel               : Gaussian %d x %d" % kernel.shape,
        "amount               : %.3f" % amount,
        "engine               : %s" % engine_name,
        "transform shape      : %d x %d" % transform_shape,
        "max |spectral - spatial| : %.3e" % maximum_error,
        "verification         : %s" % verdict,
    ])
    print("verification:", verdict, "(max error %.3e)" % maximum_error)
    print("wrote outputs to", out_dir)
    if verdict != "MATCH":
        raise RuntimeError("sharpened result did not match the reference")
    return result


def main():
    parser = argparse.ArgumentParser(description="Unsharp masking through the spectrum")
    parser.add_argument("--image", default="images/skyline256.png")
    parser.add_argument("--kernel-size", type=int, default=21)
    parser.add_argument("--amount", type=float, default=1.5)
    parser.add_argument("--engine", choices=["dft", "fft", "arbitrary"], default="fft")
    parser.add_argument("--color", action="store_true")
    parser.add_argument("--out-dir", default=os.path.join(PRACTICE_DIR, "outputs", "b2"))
    args = parser.parse_args()
    run(args.image, args.kernel_size, args.amount, args.engine, args.out_dir, color=args.color)


if __name__ == "__main__":
    main()
