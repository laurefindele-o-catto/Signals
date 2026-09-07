import numpy as np
import cv2

def detect_row_shift(original_row, shifted_row):
    """
    Find the circular shift (in samples) applied to shifted_row relative
    to original_row, using FFT-based cross-correlation.
    """
    X = np.fft.fft(original_row)
    Y = np.fft.fft(shifted_row)
    cross_corr = np.fft.ifft(Y * np.conj(X)).real
    return int(np.argmax(cross_corr))


def reconstruct_image(original, shifted):
    """
    Detect and reverse a per-row circular shift (rolling-shutter distortion).

    original, shifted : 2D arrays (grayscale), same shape (H, W)
    Returns: reconstructed image, and the array of detected shifts per row.
    """
    H, W = original.shape
    reconstructed = np.zeros_like(shifted)
    shifts = np.zeros(H, dtype=int)

    for r in range(H):
        s = detect_row_shift(original[r], shifted[r])
        shifts[r] = s
        reconstructed[r] = np.roll(shifted[r], -s)

    return reconstructed, shifts


if __name__ == "__main__":
    original = cv2.imread("original_image.png", cv2.IMREAD_GRAYSCALE).astype(np.float64)
    shifted = cv2.imread("shifted_image.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float64)

    reconstructed, shifts = reconstruct_image(original, shifted)

    mse = np.mean((original - reconstructed) ** 2)
    print("Detected shifts per row:", shifts)
    print("MSE after reconstruction:", mse)

    cv2.imwrite("reconstructed.png", np.clip(reconstructed, 0, 255).astype(np.uint8))