import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread
from scipy.ndimage import shift as ndi_shift
from types import SimpleNamespace
import sys


# =====================================================================
# Given classes — paste your offline implementations where indicated
# =====================================================================

class ContinuousImage:
    """Represents a grayscale image as a continuous 2D spatial signal. (Given)"""

    def __init__(self, image_path):
        self.image = imread(image_path, mode='L').astype(float)
        self.image = self.image / np.max(self.image)

        # Continuous spatial coordinate vectors, both spanning [-1, 1]
        self.x = np.linspace(-1, 1, self.image.shape[1])
        self.y = np.linspace(-1, 1, self.image.shape[0])

    def show(self, title="Image"):
        plt.imshow(self.image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()


class CFT2D:
    """Computes the 2D Continuous Fourier Transform of a ContinuousImage
    using separable numerical (trapezoidal) integration. (Given — pasted
    from the offline pew_edge_detector.py solution.)"""

    def __init__(self, image_obj: ContinuousImage):
        self.I = image_obj.image
        self.x = image_obj.x
        self.y = image_obj.y

        # Frequency axes conjugate to x and y (given), spanning the full
        # Nyquist range implied by the sample spacing (dx, dy).
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        self.u = np.linspace(-1 / (2 * dx), 1 / (2 * dx), self.I.shape[1])
        self.v = np.linspace(-1 / (2 * dy), 1 / (2 * dy), self.I.shape[0])

    def compute_cft(self):
        """
        Re{F(u,v)} =  Integral Integral I(x,y) cos(2*pi*(u*x + v*y)) dx dy
        Im{F(u,v)} = -Integral Integral I(x,y) sin(2*pi*(u*x + v*y)) dx dy

        Computed via SEPARABLE trapezoidal integration (angle-sum
        identities), not np.fft: first integrate over x for every (y, u)
        pair, then integrate that result over y for every (u, v) pair.
        """
        helper = Helper(self)

        # Stage 1: integrate over x, for every u -> array over y, per u
        cos_ux = np.array([helper.compute_cos_ux_for_each_u(u) for u in self.u])
        sin_ux = np.array([helper.compute_sin_ux_for_each_u(u) for u in self.u])

        # Stage 2: integrate over y, for every v, using
        # cos(ux+vy) = cos(ux)cos(vy) - sin(ux)sin(vy)
        cos_ux_cos_vy = np.array([helper.compute_cos_vy_for_each_v(v, cos_ux) for v in self.v])
        sin_ux_sin_vy = np.array([helper.compute_sin_vy_for_each_v(v, sin_ux) for v in self.v])
        real = cos_ux_cos_vy - sin_ux_sin_vy

        # Stage 2 (again): using sin(ux+vy) = sin(ux)cos(vy) + cos(ux)sin(vy)
        sin_ux_cos_vy = np.array([helper.compute_cos_vy_for_each_v(v, sin_ux) for v in self.v])
        cos_ux_sin_vy = np.array([helper.compute_sin_vy_for_each_v(v, cos_ux) for v in self.v])
        imag = -(sin_ux_cos_vy + cos_ux_sin_vy)

        self.real, self.imag = real, imag
        return real, imag

    def plot_magnitude(self):
        real, imaginary = self.compute_cft()
        magnitude = np.sqrt(real ** 2 + imaginary ** 2)
        plt.imshow(np.log(1 + magnitude), cmap='gray')
        plt.title("Magnitude Spectrum")
        plt.axis('off')
        plt.savefig("cft_magnitude.png")
        plt.close()


class Helper:
    """(Given — pasted from offline.) Separable-integration helper for CFT2D."""

    def __init__(self, cft: CFT2D):
        self.cft = cft

    def compute_cos_ux_for_each_u(self, u):
        c = np.cos(2 * np.pi * u * self.cft.x)
        integ = self.cft.I * c
        return np.trapezoid(integ, self.cft.x, axis=1)

    def compute_sin_ux_for_each_u(self, u):
        c = np.sin(2 * np.pi * u * self.cft.x)
        integ = self.cft.I * c
        return np.trapezoid(integ, self.cft.x, axis=1)

    def compute_cos_vy_for_each_v(self, v, I):
        c = np.cos(2 * np.pi * v * self.cft.y)
        integ = I * c
        return np.trapezoid(integ, self.cft.y, axis=1)

    def compute_sin_vy_for_each_v(self, v, I):
        c = np.sin(2 * np.pi * v * self.cft.y)
        integ = I * c
        return np.trapezoid(integ, self.cft.y, axis=1)


class InverseCFT2D:
    """Reconstructs the spatial-domain image from a (filtered) 2D frequency
    spectrum using separable numerical integration. (Given — pasted from
    the offline pew_edge_detector.py solution.)"""

    def __init__(self, real, imag, u, v, x, y):
        self.real = real
        self.imag = imag
        self.u = u
        self.v = v
        self.x = x
        self.y = y

    def reconstruct(self):
        """
        I(x,y) = Integral Integral F(u,v) exp(j*2*pi*(u*x + v*y)) du dv

        via separable trapezoidal integration: integrate over u first (for
        every (y, x) pair via cos/sin expansion), then over v.
        """
        helper = ReconstructionHelper(self)

        # Stage 1: integrate over u, for every x -> array over v, per x
        cos_xu_real = np.array([helper.compute_cos_xu_for_each_x(x, self.real) for x in self.x])
        sin_xu_real = np.array([helper.compute_sin_xu_for_each_x(x, self.real) for x in self.x])

        # Stage 2: integrate over v, for every y, using
        # real*cos(ux+vy) = real*[cos(ux)cos(vy) - sin(ux)sin(vy)]
        cos_xu_cos_yv_real = np.array([helper.compute_cos_yv_for_each_y(y, cos_xu_real) for y in self.y])
        sin_xu_sin_yv_real = np.array([helper.compute_sin_yv_for_each_y(y, sin_xu_real) for y in self.y])
        p = cos_xu_cos_yv_real - sin_xu_sin_yv_real

        # Same two stages for imag * sin(ux+vy) = imag*[sin(ux)cos(vy) + cos(ux)sin(vy)]
        sin_xu_imag = np.array([helper.compute_sin_xu_for_each_x(x, self.imag) for x in self.x])
        cos_xu_imag = np.array([helper.compute_cos_xu_for_each_x(x, self.imag) for x in self.x])

        sin_xu_cos_yv_imag = np.array([helper.compute_cos_yv_for_each_y(y, sin_xu_imag) for y in self.y])
        cos_xu_sin_yv_imag = np.array([helper.compute_sin_yv_for_each_y(y, cos_xu_imag) for y in self.y])
        q = sin_xu_cos_yv_imag + cos_xu_sin_yv_imag

        return p - q


class ReconstructionHelper:
    """(Given — pasted from offline.) Separable-integration helper for InverseCFT2D."""

    def __init__(self, inv: InverseCFT2D):
        self.inv = inv

    def compute_cos_xu_for_each_x(self, x, F):
        c = np.cos(2 * np.pi * x * self.inv.u)
        integ = F * c
        return np.trapezoid(integ, self.inv.u, axis=1)

    def compute_sin_xu_for_each_x(self, x, F):
        c = np.sin(2 * np.pi * x * self.inv.u)
        integ = F * c
        return np.trapezoid(integ, self.inv.u, axis=1)

    def compute_cos_yv_for_each_y(self, y, I):
        c = np.cos(2 * np.pi * y * self.inv.v)
        integ = I * c
        return np.trapezoid(integ, self.inv.v, axis=1)

    def compute_sin_yv_for_each_y(self, y, I):
        c = np.sin(2 * np.pi * y * self.inv.v)
        integ = I * c
        return np.trapezoid(integ, self.inv.v, axis=1)


# =====================================================================
# Task A — Magnitude/Phase decomposition & swap
# =====================================================================

class MagnitudePhaseTools:
    """Practice: decompose a spectrum into magnitude/phase, recombine, and swap."""

    def to_mag_phase(self, real, imag):
        spectrum = real + 1j * imag
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        return magnitude, phase

    def from_mag_phase(self, magnitude, phase):
        spectrum = magnitude * np.exp(1j * phase)
        return spectrum.real, spectrum.imag

    def swap_magnitude_phase(self, real_a, imag_a, real_b, imag_b):
        magnitude_a, _ = self.to_mag_phase(real_a, imag_a)
        _, phase_b = self.to_mag_phase(real_b, imag_b)
        return self.from_mag_phase(magnitude_a, phase_b)

    def magnitude_only(self, real, imag):
        magnitude, _ = self.to_mag_phase(real, imag)
        # phase = 0 everywhere -> purely real, non-negative spectrum
        return self.from_mag_phase(magnitude, np.zeros_like(magnitude))

    def phase_only(self, real, imag):
        _, phase = self.to_mag_phase(real, imag)
        # magnitude = 1 everywhere
        return self.from_mag_phase(np.ones_like(phase), phase)


# =====================================================================
# Task B — Parseval's theorem check
# =====================================================================

class ParsevalChecker:
    """Practice: verify energy conservation between spatial and frequency domains."""

    def spatial_energy(self, image, dx=1.0, dy=1.0):
        """
        sum(|I(x,y)|^2) * dx * dy — a Riemann-sum approximation of
        integral |I(x,y)|^2 dx dy. dx, dy default to 1 so the raw,
        un-weighted pixel energy is still available if needed, but
        verify_parseval always passes the true grid spacing in.
        """
        return np.sum(np.abs(image) ** 2) * dx * dy

    def frequency_energy(self, real, imag, du, dv):
        """sum(|F(u,v)|^2) * du * dv, matching CFT2D.compute_cft's dx*dy-scaled convention."""
        spectrum = real + 1j * imag
        return np.sum(np.abs(spectrum) ** 2) * du * dv

    def verify_parseval(self, image, real, imag, du, dv, tol=1e-6):
        """
        Parseval: integral |I(x,y)|^2 dx dy == integral |F(u,v)|^2 du dv.
        Both sides here are Riemann-sum approximations, so both must carry
        their respective area elements (dx*dy on the spatial side, du*dv on
        the frequency side) — dx, dy are inferred from the image's own
        pixel count assuming the same [-1, 1] domain used by ContinuousImage.
        """
        ny, nx = image.shape
        dx = 2.0 / (nx - 1)
        dy = 2.0 / (ny - 1)
        e_spatial = self.spatial_energy(image, dx, dy)
        e_freq = self.frequency_energy(real, imag, du, dv)
        relative_error = np.abs(e_spatial - e_freq) / e_spatial
        is_valid = relative_error < tol
        return is_valid, relative_error


# =====================================================================
# Task C — Shift theorem
# =====================================================================

class ShiftTheorem:
    """Practice: translating an image in space <=> multiplying its spectrum by a phase ramp."""

    def shift_image_spatial(self, image, x, y, dx, dy):
        """
        Shift `image` by (dx, dy) in the continuous (x, y) coordinate system.
        Converts the continuous offsets into fractional pixel offsets using
        the grid spacing, then uses spline interpolation (order=1, i.e.
        bilinear) with wraparound, matching the implicit periodicity assumed
        by the DFT/CFT.
        """
        pixel_dx = dx / (x[1] - x[0])
        pixel_dy = dy / (y[1] - y[0])
        # image is indexed [row=y, col=x] -> shift is (row_shift, col_shift)
        # cubic spline (order=3) gives a much more accurate sub-pixel shift
        # than nearest/linear, which matters since the shift theorem check
        # compares against an exact phase-ramp prediction.
        shifted = ndi_shift(image, shift=(pixel_dy, pixel_dx), mode="wrap", order=3)
        return shifted
    
    def shift_image_spatial_gemini(self, image, x, y, dx, dy):
        # 1. Convert physical shifts (dx, dy) into pixel-grid shifts
        pixel_dx = dx / (x[1] - x[0])
        pixel_dy = dy / (y[1] - y[0])
    
        rows, cols = image.shape
        shifted = np.zeros((rows, cols))
    
        # 2. Loop through every pixel in the new shifted image
        for r in range(rows):
            for c in range(cols):
                # Find the original source position before the shift
                src_r = r - pixel_dy
                src_c = c - pixel_dx
            
                # Get the four surrounding integer pixel coordinates
                r0 = int(np.floor(src_r))
                r1 = r0 + 1
                c0 = int(np.floor(src_c))
                c1 = c0 + 1
            
                # Calculate how far the source position is from those integers
                dr = src_r - r0
                dc = src_c - c0
            
                # Wrap the indices around the edges to match DFT periodicity
                r0_wrap = r0 % rows
                r1_wrap = r1 % rows
                c0_wrap = c0 % cols
                c1_wrap = c1 % cols
            
                # Fetch the values of the 4 neighboring pixels
                p00 = image[r0_wrap, c0_wrap]  # Top-left
                p10 = image[r1_wrap, c0_wrap]  # Bottom-left
                p01 = image[r0_wrap, c1_wrap]  # Top-right
                p11 = image[r1_wrap, c1_wrap]  # Bottom-right
            
                # Perform bilinear interpolation
                top_blend = p00 + dc * (p01 - p00)
                bottom_blend = p10 + dc * (p11 - p10)
                final_pixel = top_blend + dr * (bottom_blend - top_blend)
            
                # Assign the interpolated value to the output image
                shifted[r, c] = final_pixel
            
        return shifted


    def apply_phase_ramp(self, real, imag, u, v, dx, dy):
        U, V = np.meshgrid(u, v)  # U varies along columns, V along rows
        ramp = np.exp(-1j * 2 * np.pi * (U * dx + V * dy))
        spectrum = (real + 1j * imag) * ramp
        return spectrum.real, spectrum.imag

    def verify_shift_theorem(self, cft2d_obj: CFT2D, image_obj: ContinuousImage,
                              dx, dy, tol=1e-6):
        # 1. CFT of the original image
        real, imag = cft2d_obj.compute_cft()

        # 2. CFT of the spatially-shifted image, computed directly by
        #    reusing the actual CFT2D implementation (same x, y, and
        #    therefore the same u, v grid as cft2d_obj).
        shifted_image = self.shift_image_spatial(
            image_obj.image, image_obj.x, image_obj.y, dx, dy
        )
        shifted_obj = SimpleNamespace(image=shifted_image, x=image_obj.x, y=image_obj.y)
        cft_shifted = CFT2D(shifted_obj)
        real_s, imag_s = cft_shifted.compute_cft()

        # 3. Predicted spectrum via the shift theorem (phase ramp)
        real_p, imag_p = self.apply_phase_ramp(real, imag, cft2d_obj.u, cft2d_obj.v, dx, dy)

        # 4. Compare
        delta = np.max(np.abs((real_s + 1j * imag_s) - (real_p + 1j * imag_p)))
        is_valid = delta < tol
        return is_valid, delta


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
