import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread
import sys


# =====================================================================
# Given classes — paste your offline implementations where indicated
# =====================================================================

class ContinuousImage:
    """Represents a grayscale image as a continuous 2D spatial signal. (Given)"""

    def __init__(self, image_path):
        self.image = imread(image_path, mode='L').astype(float)
        self.image = self.image / np.max(self.image)
        self.x = np.linspace(-1, 1, self.image.shape[1])
        self.y = np.linspace(-1, 1, self.image.shape[0])


class CFT2D:
    """2D Continuous Fourier Transform. (Given — paste your offline solution)"""

    def __init__(self, image_obj: ContinuousImage):
        self.I = image_obj.image
        self.x = image_obj.x
        self.y = image_obj.y
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        self.u = np.linspace(-1 / (2 * dx), 1 / (2 * dx), self.I.shape[1])
        self.v = np.linspace(-1 / (2 * dy), 1 / (2 * dy), self.I.shape[0])

    def compute_cft(self):
        raise NotImplementedError("Paste your offline compute_cft here.")

    def plot_magnitude(self):
        raise NotImplementedError("Paste your offline plot_magnitude here.")


class InverseCFT2D:
    """Inverse 2D-CFT. (Given — paste your offline solution)"""

    def __init__(self, real, imag, u, v, x, y):
        self.real = real
        self.imag = imag
        self.u = u
        self.v = v
        self.x = x
        self.y = y

    def reconstruct(self):
        raise NotImplementedError("Paste your offline reconstruct here.")


# =====================================================================
# Task A — Magnitude/Phase decomposition & swap
# =====================================================================

class MagnitudePhaseTools:
    """Practice: decompose a spectrum into magnitude/phase, recombine, and swap."""

    def to_mag_phase(self, real, imag):
        """TODO: return (magnitude, phase) arrays from real/imag spectrum components."""
        raise NotImplementedError

    def from_mag_phase(self, magnitude, phase):
        """TODO: return (real, imag) reconstructed from magnitude/phase arrays."""
        raise NotImplementedError

    def swap_magnitude_phase(self, real_a, imag_a, real_b, imag_b):
        """
        TODO: build a hybrid spectrum using the MAGNITUDE of spectrum A
        and the PHASE of spectrum B. Return (real, imag) of the hybrid.
        """
        raise NotImplementedError

    def magnitude_only(self, real, imag):
        """TODO: zero out phase (i.e. treat as purely real, non-negative), keep magnitude only."""
        raise NotImplementedError

    def phase_only(self, real, imag):
        """TODO: set magnitude to 1 everywhere, keep phase only."""
        raise NotImplementedError


# =====================================================================
# Task B — Parseval's theorem check
# =====================================================================

class ParsevalChecker:
    """Practice: verify energy conservation between spatial and frequency domains."""

    def spatial_energy(self, image):
        """TODO: return sum(|I(x,y)|^2) over the spatial-domain image."""
        raise NotImplementedError

    def frequency_energy(self, real, imag, du, dv):
        """
        TODO: return the frequency-domain energy sum(|F(u,v)|^2) * du * dv
        (match whatever normalization your offline's compute_cft convention uses).
        """
        raise NotImplementedError

    def verify_parseval(self, image, real, imag, du, dv, tol=1e-6):
        """
        TODO: compare spatial_energy(image) to frequency_energy(real, imag, du, dv).
        Return (is_valid: bool, relative_error: float).
        """
        raise NotImplementedError


# =====================================================================
# Task C — Shift theorem
# =====================================================================

class ShiftTheorem:
    """Practice: translating an image in space <=> multiplying its spectrum by a phase ramp."""

    def shift_image_spatial(self, image, x, y, dx, dy):
        """
        TODO: produce a spatially-shifted copy of `image` by (dx, dy)
        (nearest-neighbour or interpolated shift on the x,y grid is fine).
        """
        raise NotImplementedError

    def apply_phase_ramp(self, real, imag, u, v, dx, dy):
        """
        TODO: multiply the spectrum (real + i*imag) by exp(-i*2*pi*(u*dx + v*dy))
        (broadcast u along columns, v along rows). Return (real, imag) of the result.
        """
        raise NotImplementedError

    def verify_shift_theorem(self, cft2d_obj: CFT2D, image_obj: ContinuousImage,
                              dx, dy, tol=1e-6):
        """
        TODO:
          1. Compute CFT of the original image -> (real, imag)
          2. Compute CFT of the spatially-shifted image directly -> (real_s, imag_s)
          3. Compute (real, imag) * phase ramp via apply_phase_ramp -> (real_p, imag_p)
          4. Compare (real_s, imag_s) to (real_p, imag_p)
        Return (is_valid: bool, max_abs_delta: float).
        """
        raise NotImplementedError


# =====================================================================
# Validator — checks your filled-in methods against independent ground truth
# =====================================================================

class FrequencyDomainValidator:
    """
    Runs independent checks against your implementations above.
    Does NOT reimplement your algorithms — it checks the mathematical
    invariants your methods are supposed to satisfy.
    """

    def check_mag_phase_roundtrip(self, tools: MagnitudePhaseTools, real, imag, tol=1e-9):
        mag, phase = tools.to_mag_phase(real, imag)
        r2, i2 = tools.from_mag_phase(mag, phase)
        delta = np.max(np.abs((real + 1j * imag) - (r2 + 1j * i2)))
        return delta < tol, delta

    def check_magnitude_matches_original(self, tools: MagnitudePhaseTools, real, imag, tol=1e-9):
        """Swapping A's magnitude with A's own phase should reproduce A exactly."""
        r_swap, i_swap = tools.swap_magnitude_phase(real, imag, real, imag)
        delta = np.max(np.abs((real + 1j * imag) - (r_swap + 1j * i_swap)))
        return delta < tol, delta

    def check_phase_only_unit_magnitude(self, tools: MagnitudePhaseTools, real, imag, tol=1e-9):
        r_p, i_p = tools.phase_only(real, imag)
        mags = np.abs(r_p + 1j * i_p)
        delta = np.max(np.abs(mags - 1))
        return delta < tol, delta

    def check_parseval(self, checker: ParsevalChecker, image, real, imag, du, dv, tol=1e-3):
        is_valid, rel_err = checker.verify_parseval(image, real, imag, du, dv, tol=tol)
        return is_valid, rel_err

    def check_shift_theorem(self, shifter: ShiftTheorem, cft2d_obj, image_obj, dx, dy, tol=1e-3):
        is_valid, delta = shifter.verify_shift_theorem(cft2d_obj, image_obj, dx, dy, tol=tol)
        return is_valid, delta


# =====================================================================
# Entry point — wire everything together once your TODOs are filled in
# =====================================================================
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 cft_mag_phase_shift.py <image_a> <image_b>")
        sys.exit(1)

    path_a, path_b = sys.argv[1], sys.argv[2]

    img_a = ContinuousImage(path_a)
    img_b = ContinuousImage(path_b)
    cft_a = CFT2D(img_a)
    cft_b = CFT2D(img_b)
    real_a, imag_a = cft_a.compute_cft()
    real_b, imag_b = cft_b.compute_cft()

    tools = MagnitudePhaseTools()
    validator = FrequencyDomainValidator()

    ok, d = validator.check_mag_phase_roundtrip(tools, real_a, imag_a)
    print(f"[mag/phase roundtrip] valid={ok} delta={d:.2e}")

    ok, d = validator.check_magnitude_matches_original(tools, real_a, imag_a)
    print(f"[swap A-mag with A-phase == A] valid={ok} delta={d:.2e}")

    ok, d = validator.check_phase_only_unit_magnitude(tools, real_a, imag_a)
    print(f"[phase-only has unit magnitude] valid={ok} delta={d:.2e}")

    real_hybrid, imag_hybrid = tools.swap_magnitude_phase(real_a, imag_a, real_b, imag_b)
    hybrid_img = InverseCFT2D(real_hybrid, imag_hybrid, cft_a.u, cft_a.v, img_a.x, img_a.y).reconstruct()
    plt.imsave("hybrid_magA_phaseB.png", np.clip(np.abs(hybrid_img), 0, 1), cmap='gray')
    print("Saved hybrid_magA_phaseB.png")

    checker = ParsevalChecker()
    du = cft_a.u[1] - cft_a.u[0]
    dv = cft_a.v[1] - cft_a.v[0]
    ok, rel_err = validator.check_parseval(checker, img_a.image, real_a, imag_a, du, dv)
    print(f"[Parseval] valid={ok} relative_error={rel_err:.2e}")

    shifter = ShiftTheorem()
    ok, d = validator.check_shift_theorem(shifter, cft_a, img_a, dx=0.1, dy=0.05)
    print(f"[Shift theorem] valid={ok} max_abs_delta={d:.2e}")
