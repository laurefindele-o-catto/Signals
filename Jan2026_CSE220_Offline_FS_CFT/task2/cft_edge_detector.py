import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread


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
    using separable numerical (trapezoidal) integration."""

    def __init__(self, image_obj: ContinuousImage):
        self.I = image_obj.image
        self.x = image_obj.x
        self.y = image_obj.y

        # Frequency axes conjugate to x and y (given), spanning the full
        # Nyquist range implied by the sample spacing (dx, dy). This is
        # what lets the transform represent fine, edge-scale spatial
        # detail instead of only very coarse (near-DC) variation.
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        self.u = np.linspace(-1 / (2 * dx), 1 / (2 * dx), self.I.shape[1])
        self.v = np.linspace(-1 / (2 * dy), 1 / (2 * dy), self.I.shape[0])

    def compute_cft(self):
        """
        Compute the real and imaginary parts of the 2D Continuous Fourier
        Transform of self.I, using SEPARABLE trapezoidal integration:

            Re{F(u,v)} =  Integral Integral I(x,y) cos(2*pi*(u*x + v*y)) dx dy
            Im{F(u,v)} = -Integral Integral I(x,y) sin(2*pi*(u*x + v*y)) dx dy

        Do NOT evaluate this as a direct 4-nested-loop double integral over
        (x, y, u, v) -- that is O(N^4) and will not finish in reasonable
        time. Instead exploit separability: expand cos(2*pi*(ux+vy)) and
        sin(2*pi*(ux+vy)) with the angle-sum identities, first integrate
        over x for every (y, u) pair, then integrate the result over y for
        every (u, v) pair. Each of the two stages is an O(N^3) operation
        (an O(N) numerical integral, repeated over an N x N grid), which is
        what makes this tractable.

        Use self.u and self.v (NOT self.x/self.y) as the frequency axes --
        they were already computed for you in __init__.

        Use np.trapezoid(..., axis=...) for the integration -- no built-in
        FFT/DFT routine (np.fft, scipy.fft, ...) may be used anywhere in
        this method.

        Returns
        -------
        real, imag : two 2D numpy arrays, each of shape self.I.shape
        """
        # TODO: implement this method
        
        icos_list = []
        isin_list = []
        
        for i in range(0, len(self.u)):
            cos_x = np.cos(2 * np.pi * self.u[i] * self.x)
            sin_x = np.sin(2 * np.pi * self.u[i] * self.x)
            
            #image ar cos_x, sin_x multiplied
            #self.x is the 2nd parameter - mane dx er respect e integration
            int_cos = np.trapezoid(self.I * cos_x, self.x, axis=1)
            int_sin = np.trapezoid(self.I * sin_x, self.x, axis=1)
            
            icos_list.append(int_cos)
            isin_list.append(int_sin)            
            
        #list is now len(u), len(y)
        
        i_cos = np.array(icos_list)
        i_sin = np.array(isin_list)
        
        real_list = []
        imag_list = []
        
        for i in range(0, len(self.v)):
            cos_y = np.cos(2 * np.pi * self.v[i] * self.y)
            sin_y = np.sin(2 * np.pi * self.v[i] * self.y)
            
            real_part = i_cos * cos_y - i_sin * sin_y
            real_col = np.trapezoid(real_part, self.y, axis=1)
            real_list.append(real_col)  #collapsed value gula append
            
            imag_part = -(i_sin * cos_y + i_cos * sin_y)
            imag_col = np.trapezoid(imag_part, self.y, axis=1)
            imag_list.append(imag_col)
            
        F_real = np.array(real_list)
        F_imag = np.array(imag_list)
        
        return F_real, F_imag
    

    def plot_magnitude(self):
        """
        Plot the log-scaled magnitude spectrum of the 2D CFT computed by
        compute_cft(), i.e. plt.imshow(np.log(1 + magnitude), ...) where
        magnitude = sqrt(real**2 + imag**2). Purely for your own visual
        debugging -- not called by the command-line entry point below.
        """
        # TODO: implement this method
        
        real, imag = self.compute_cft()
        magnitude = np.sqrt(real**2 + imag**2)
        
        extent = [self.u[0], self.u[-1], self.v[0], self.v[-1]]
        
        plt.imshow(np.log(1 + magnitude), cmap = 'gray', extent = extent)
        plt.title("Log-scaled magnitude spectrum")
        plt.xlabel("u")
        plt.ylabel("v")
        plt.show()
        
        

class FrequencyFilter:
    """Applies frequency-domain filtering operations. (Given)"""

    def high_pass(self, real, imag, cutoff):
        rows, cols = real.shape
        cx, cy = rows // 2, cols // 2

        real = real.copy()
        imag = imag.copy()
        for i in range(rows):
            for j in range(cols):
                if np.sqrt((i - cx) ** 2 + (j - cy) ** 2) <= cutoff:
                    real[i, j] = 0
                    imag[i, j] = 0
        return real, imag


class InverseCFT2D:
    """Reconstructs the spatial-domain image from a (filtered) 2D frequency
    spectrum using separable numerical integration."""

    def __init__(self, real, imag, u, v, x, y):
        self.real = real
        self.imag = imag
        self.u = u
        self.v = v
        self.x = x
        self.y = y

    def reconstruct(self):
        """
        Perform the inverse 2D Continuous Fourier Transform:

            I(x,y) = Integral Integral F(u,v) exp(j*2*pi*(u*x + v*y)) du dv

        using the same separable-integration strategy as compute_cft():
        expand the complex exponential into cos/sin via Euler's identity,
        integrate over v first (for every (y, u) pair), then integrate
        that result over u (for every (x, y) pair). Use np.trapezoid.

        self.real, self.imag are the (possibly filtered) frequency-domain
        components; self.u, self.v are the frequency axes they were
        computed on; self.x, self.y are the spatial axes to reconstruct
        onto.

        Returns
        -------
        image : 2D numpy array of shape (len(self.y), len(self.x))
            The reconstructed real-valued spatial-domain signal. Note
            that after a high-pass filter this is NOT a valid image on
            its own (it will contain negative values, since the DC/
            low-frequency component that carried the average brightness
            has been removed) -- see the command-line entry point below
            for how it gets turned into a displayable edge map.
        """
        # TODO: implement this method
        
        r_cos_list = []
        r_sin_list = []
        i_cos_list = []
        i_sin_list = []
        
        for i in range(0, len(self.y)):
            cos_v = np.cos(2*np.pi*self.y[i]*self.v)
            sin_v = np.sin(2*np.pi*self.y[i]*self.v)
            
            #self.real er shape len(v), len(u) -> axis 0
            #cos_v er shape tahole len v, 1 type lagbe
             
            int_rcos = np.trapezoid(self.real * cos_v[:, None], self.v, axis=0)
            int_rsin = np.trapezoid(self.real * sin_v[:, None], self.v, axis=0)
            int_icos = np.trapezoid(self.imag * cos_v[:, None], self.v, axis=0)
            int_isin = np.trapezoid(self.imag * sin_v[:, None], self.v, axis=0)
            
            r_cos_list.append(int_rcos)
            r_sin_list.append(int_rsin)
            i_cos_list.append(int_icos)
            i_sin_list.append(int_isin)
            
        r_cos = np.array(r_cos_list)
        r_sin = np.array(r_sin_list)
        i_cos = np.array(i_cos_list)
        i_sin = np.array(i_sin_list)
        
        final_list = []
        
        for i in range(0, len(self.x)):
            cos_u = np.cos(2 * np.pi * self.x[i] * self.u)
            sin_u = np.sin(2 * np.pi * self.x[i] * self.u)
            
            part = cos_u * (r_cos - i_sin) - sin_u * (r_sin + i_cos)
            
            int_part = np.trapezoid(part, self.u, axis = 1)
            final_list.append(int_part)
            
        I = np.array(final_list).T
        
        return I


# =====================================================
# Command-line entry point (given -- do not modify)
# Usage: python3 cft_edge_detector.py <input_image_path> <output_image_path> [cutoff]
# =====================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python3 cft_edge_detector.py <input_image_path> <output_image_path> [cutoff]")
        print("Example: python3 cft_edge_detector.py pikachu.png pikachu_edges.png 15")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    cutoff = float(sys.argv[3]) if len(sys.argv) > 3 else 15

    img = ContinuousImage(input_path)
    cft2d = CFT2D(img)
    real, imag = cft2d.compute_cft()

    filt = FrequencyFilter()
    real_f, imag_f = filt.high_pass(real, imag, cutoff)

    icft2d = InverseCFT2D(real_f, imag_f, cft2d.u, cft2d.v, img.x, img.y)
    edges = icft2d.reconstruct()

    edge_map = np.abs(edges)
    if edge_map.max() > 0:
        edge_map = edge_map / edge_map.max()
    edge_map = 1 - edge_map  # invert: edges black, background white

    plt.imsave(output_path, edge_map, cmap='gray')
    print(f"Saved edge map to {output_path}")


#python cft_edge_detector.py pikachu.png pikachu_edges.png 15

#python cft_edge_detector.py castle.png castle_edges.png 15