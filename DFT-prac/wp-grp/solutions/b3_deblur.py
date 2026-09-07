"""
Practice online B3 -- deblurring by inverse filtering (spectral division).

Complete TODO 1--3. Do not modify the original offline files.

Run:

    python b3_deblur.py --image images/skyline256.png --kernel-size 5 --engine fft
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
from transforms import ArbitraryLengthFFT, DFTAnalyzer, FFTTransformer


def wrapped_kernel_spectrum(kernel, shape, engine):
    """Return the spectrum of the kernel wrapped around the origin of ``shape``."""
    # TODO 1 (student): embed the kernel at the top-left of a zero array of
    # the plane's shape, roll it by (-(kh//2), -(kw//2)) so its centre tap sits
    # at index (0, 0) -- exactly the circular path of convolve_plane -- and
    # transform it.
    krows, kcols = kernel.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:krows, :kcols] = kernel
    padded = np.roll(padded, (-(krows // 2), -(kcols // 2)), axis=(0, 1))
    return transform_2d(padded, engine)


def deblur_plane(blurred, kernel, engine, epsilon=1e-12):
    """Undo a circular blur: divide the blurred spectrum by the kernel spectrum."""
    # TODO 2 (student): Y = spectrum of the blurred plane, G = wrapped kernel
    # spectrum. Where |G| > epsilon set X = Y / G; everywhere else set X = 0
    # (never divide by a vanishing bin). Inverse-transform ONCE, keep the real part.
    blurred = np.asarray(blurred, dtype=np.float64)
    blurred_spectrum = transform_2d(blurred, engine)
    kernel_spectrum = wrapped_kernel_spectrum(kernel, blurred.shape, engine)
    restored = np.zeros_like(blurred_spectrum)
    reliable = np.abs(kernel_spectrum) > epsilon
    restored[reliable] = blurred_spectrum[reliable] / kernel_spectrum[reliable]
    return inverse_2d(restored, engine).real


def deblur_image(blurred, kernel, engine):
    """Apply deblur_plane to a grayscale or RGB image, preserving its shape."""
    # TODO 3 (student): one plane for grayscale; three planes stacked back
    # along the last axis for RGB.
    blurred = np.asarray(blurred, dtype=np.float64)
    if blurred.ndim == 2:
        return deblur_plane(blurred, kernel, engine)
    if blurred.ndim == 3 and blurred.shape[2] == 3:
        planes = [deblur_plane(blurred[:, :, c], kernel, engine)
                  for c in range(blurred.shape[2])]
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


def run(image_path, kernel_size, engine_name, out_dir, color=False):
    """Provided runner: blur circularly with the offline, deblur, compare with the original."""
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel size must be a positive odd integer")
    engine = _make_engine(engine_name)
    original = load_image(_resolve(image_path), as_gray=not color)
    kernel = make_kernel("gaussian", size=kernel_size)

    blurred = convolve_image(original, kernel, engine, circular=True)   # what the student receives
    restored = deblur_image(blurred, kernel, engine)

    maximum_error = float(np.max(np.abs(restored - original)))          # the oracle is the original
    verdict = "MATCH" if maximum_error <= 1e-9 else "MISMATCH"
    smallest_bin = float(np.min(np.abs(wrapped_kernel_spectrum(kernel, original.shape[:2], engine))))

    os.makedirs(out_dir, exist_ok=True)
    save_image(blurred, os.path.join(out_dir, "blurred.png"))
    save_image(restored, os.path.join(out_dir, "restored.png"))
    save_comparison(
        [original, blurred, restored],
        ["original", "circular Gaussian blur", "restored by spectral division"],
        os.path.join(out_dir, "comparison.png"),
        suptitle="Inverse filtering: Gaussian %dx%d, engine=%s" % (kernel_size, kernel_size, engine_name),
    )
    write_report(os.path.join(out_dir, "report.txt"), [
        "Practice B3 -- deblurring by inverse filtering",
        "image                : %s" % image_path,
        "image shape          : %s" % (original.shape,),
        "kernel               : Gaussian %d x %d" % kernel.shape,
        "engine               : %s" % engine_name,
        "smallest |G| bin     : %.3e  (rounding noise is amplified by 1/this)" % smallest_bin,
        "max |restored - original| : %.3e" % maximum_error,
        "verification         : %s" % verdict,
    ])
    print("verification:", verdict, "(max error %.3e, smallest |G| %.3e)" % (maximum_error, smallest_bin))
    print("wrote outputs to", out_dir)
    if verdict != "MATCH":
        raise RuntimeError("restored image did not match the original")
    return restored


def main():
    parser = argparse.ArgumentParser(description="Deblur by spectral division")
    parser.add_argument("--image", default="images/skyline256.png")
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--engine", choices=["dft", "fft", "arbitrary"], default="fft")
    parser.add_argument("--color", action="store_true")
    parser.add_argument("--out-dir", default=os.path.join(PRACTICE_DIR, "outputs", "b3"))
    args = parser.parse_args()
    run(args.image, args.kernel_size, args.engine, args.out_dir, color=args.color)


if __name__ == "__main__":
    main()
