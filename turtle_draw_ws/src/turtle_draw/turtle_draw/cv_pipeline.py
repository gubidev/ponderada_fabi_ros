"""
cv_pipeline.py — Computer-Vision pipeline built entirely with NumPy.

OpenCV is used ONLY to load the image (cv2.imread).
Every transformation afterwards — resize, grayscale, Gaussian blur, Sobel edge
detection and double-threshold hysteresis — is implemented as pure NumPy matrix
operations.

Pipeline stages
---------------
1. Resize          → reduce image to ≤ max_dim on the longest side (nearest-neighbour)
2. Grayscale       → luminance-weighted channel collapse (ITU-R BT.601)
3. Gaussian blur   → 2-D convolution with a separable Gaussian kernel (noise suppression)
4. Sobel edges     → two 3×3 derivative kernels → gradient magnitude + direction
5. Hysteresis threshold → strong / weak edge classification + neighbour propagation
"""

import cv2  # permitted only for imread
import numpy as np
import matplotlib.pyplot as plt


class CVPipeline:
    """Load an image and run the full edge-detection pipeline."""

    def __init__(self, image_path: str):
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        self.original = bgr          # kept as BGR uint8 for visualisation
        self.edges: np.ndarray = None

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resize(image: np.ndarray, max_dim: int) -> np.ndarray:
        """
        Nearest-neighbour downsample so the longest side ≤ max_dim.

        We compute evenly spaced source indices with np.linspace and use
        NumPy fancy indexing — no copy of the data is made until the final
        index step.  Nearest-neighbour (i.e. round-to-int) is sufficient
        here because we only need the broad structure for edge detection,
        not sub-pixel accuracy.
        """
        h, w = image.shape[:2]
        if max(h, w) <= max_dim:
            return image
        scale = max_dim / max(h, w)
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))

        # Row / column source indices
        r_idx = np.linspace(0, h - 1, new_h).astype(np.int32)
        c_idx = np.linspace(0, w - 1, new_w).astype(np.int32)

        if image.ndim == 3:
            # Fancy index: shape (new_h, new_w, channels)
            return image[r_idx[:, None], c_idx[None, :], :]
        return image[r_idx[:, None], c_idx[None, :]]

    @staticmethod
    def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        2-D discrete convolution — implemented with NumPy stride tricks.

        Instead of nested Python loops we build a *view* of the padded image
        that exposes every (kH × kW) patch centred on each output pixel as
        two extra dimensions.  Then a single np.einsum contracts the patch
        dimensions against the kernel — the entire computation runs inside
        compiled NumPy code with no Python-level loops.

        Mathematical definition
        -----------------------
          (I ★ K)[i, j] = Σ_{m=0}^{kH-1} Σ_{n=0}^{kW-1}  I_pad[i+m, j+n] · K[m, n]

        Padding strategy
        ----------------
        'reflect' mode mirrors the image content at the border instead of
        using zeros.  Zero-padding would create artificial bright/dark rings
        at the edges that confuse the Sobel detector.

        Stride-trick mechanics
        ----------------------
        padded.strides = (s0, s1) bytes/step along rows and columns.
        The 4-D view has shape (H, W, kH, kW) and strides (s0, s1, s0, s1).
        Element [i, j, m, n] is therefore padded[i+m, j+n] — exactly the
        patch pixel we need, with zero extra memory allocation.
        """
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2
        img_f64 = image.astype(np.float64)
        padded = np.pad(img_f64, ((ph, ph), (pw, pw)), mode='reflect')

        H, W = image.shape
        s0, s1 = padded.strides
        view = np.lib.stride_tricks.as_strided(
            padded,
            shape=(H, W, kh, kw),
            strides=(s0, s1, s0, s1),
        )
        # Σ over patch dims k, l against kernel[k, l]
        return np.einsum('ijkl,kl->ij', view, kernel)

    # ------------------------------------------------------------------ #
    #  Pipeline stages                                                     #
    # ------------------------------------------------------------------ #

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert BGR image to luminance (grayscale).

        Formula: Y = 0.114·B + 0.587·G + 0.299·R   (ITU-R BT.601)

        These coefficients reflect human perceptual sensitivity: the eye is
        most sensitive to green (~59 %), moderately to red (~30 %) and least
        to blue (~11 %).  OpenCV stores channels in BGR order, so index 0 is
        Blue, 1 is Green, 2 is Red.
        """
        if image.ndim == 2:
            return image  # already single-channel
        B = image[:, :, 0].astype(np.float64)
        G = image[:, :, 1].astype(np.float64)
        R = image[:, :, 2].astype(np.float64)
        Y = 0.114 * B + 0.587 * G + 0.299 * R
        return np.clip(Y, 0, 255).astype(np.uint8)

    def gaussian_blur(self, image: np.ndarray,
                      sigma: float = 1.5, ksize: int = 5) -> np.ndarray:
        """
        Low-pass filter with an isotropic Gaussian kernel.

        Continuous 2-D Gaussian:
          G(x,y) = 1/(2πσ²) · exp(-(x²+y²)/(2σ²))

        Because the 2-D Gaussian is *separable* we build a 1-D vector g(x)
        and form the 2-D kernel as the outer product:
          K = g ⊗ g  (normalised so Σ K = 1)

        Effect: the convolution acts as a weighted local average that
        attenuates high-frequency content (fur texture, image grain) while
        keeping large-scale intensity transitions — which is exactly what we
        want so the subsequent Sobel detector fires on true object borders
        rather than texture noise.

        σ controls the trade-off: larger σ → stronger smoothing → fewer
        spurious edges but potentially blurred thin features.  σ = 1.5 with
        a 5×5 kernel is a good balance for the dog photo.
        """
        half = ksize // 2
        ax = np.arange(-half, half + 1, dtype=np.float64)
        g1d = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
        g1d /= g1d.sum()                    # normalise to unit sum
        kernel = np.outer(g1d, g1d)         # separable 2-D Gaussian
        result = self._convolve2d(image, kernel)
        return np.clip(result, 0, 255).astype(np.uint8)

    def sobel_edges(self, image: np.ndarray):
        """
        Estimate the image gradient using the Sobel operator.

        Two 3×3 kernels approximate the first-order partial derivatives:

          Kx = [[-1, 0, 1],      Ky = [[-1,-2,-1],
                [-2, 0, 2],             [ 0, 0, 0],
                [-1, 0, 1]]             [ 1, 2, 1]]

        Kx detects vertical edges (horizontal intensity change ∂I/∂x).
        Ky detects horizontal edges (vertical intensity change ∂I/∂y).

        The ±2 weight in the centre row/column smooths in the perpendicular
        direction, giving Sobel better noise robustness than a plain finite
        difference [−1, 0, 1].

        Returns
        -------
        magnitude : ndarray  √(Gx² + Gy²)  — edge strength at each pixel
        direction : ndarray  atan2(Gy, Gx) — gradient direction in radians
        """
        Kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float64)
        Ky = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=np.float64)

        Gx = self._convolve2d(image.astype(np.float64), Kx)
        Gy = self._convolve2d(image.astype(np.float64), Ky)

        magnitude  = np.sqrt(Gx ** 2 + Gy ** 2)
        direction  = np.arctan2(Gy, Gx)
        return magnitude, direction

    def threshold(self, magnitude: np.ndarray,
                  low_ratio: float = 0.15,
                  high_ratio: float = 0.35) -> np.ndarray:
        """
        Double-threshold hysteresis — a simplified Canny-style approach.

        Steps
        -----
        1. Normalise magnitude to [0, 1].
        2. Pixels ≥ high_ratio  → *strong* edge (always kept).
        3. Pixels in [low_ratio, high_ratio) → *weak* edge (kept only if
           at least one 8-connected neighbour is a strong edge).
        4. Pixels < low_ratio   → discarded.

        The two-level strategy retains coherent, connected contours while
        suppressing isolated noise speckles.  A single threshold would either
        miss thin edges (too high) or keep too much texture noise (too low).

        Neighbour check implementation
        ------------------------------
        We convolve the boolean strong-edge mask with a 3×3 ones kernel.
        The result at pixel (i,j) equals the count of strong pixels in the
        3×3 neighbourhood (including itself).  If that count > 0, the pixel
        is adjacent to a strong edge and any weak edge there is promoted.
        Using _convolve2d reuses the same stride-tricks machinery and handles
        border pixels cleanly via reflect-padding.
        """
        norm = magnitude / (magnitude.max() + 1e-8)

        strong = norm >= high_ratio
        weak   = (norm >= low_ratio) & ~strong

        # Dilate strong mask by 1 pixel using 3×3 neighbourhood sum
        dilation_kernel = np.ones((3, 3), dtype=np.float64)
        strong_dilated  = self._convolve2d(strong.astype(np.float64),
                                           dilation_kernel) > 0

        edge_map = (strong | (weak & strong_dilated)).astype(np.uint8) * 255
        return edge_map

    # ------------------------------------------------------------------ #
    #  Full pipeline entry point                                           #
    # ------------------------------------------------------------------ #

    def run(self, sigma: float = 1.5, ksize: int = 5,
            low_ratio: float = 0.15, high_ratio: float = 0.35,
            max_dim: int = 400, visualize: bool = False) -> np.ndarray:
        """
        Execute all pipeline stages and return the binary edge map.

        Parameters
        ----------
        sigma      : Gaussian blur standard deviation
        ksize      : Gaussian kernel side length (must be odd)
        low_ratio  : weak-edge threshold as fraction of max gradient
        high_ratio : strong-edge threshold as fraction of max gradient
        max_dim    : longest side of the working resolution
        visualize  : if True, save a 4-panel figure to /tmp/cv_pipeline.png
        """
        small    = self._resize(self.original, max_dim)
        gray     = self.to_grayscale(small)
        blurred  = self.gaussian_blur(gray, sigma=sigma, ksize=ksize)
        mag, _   = self.sobel_edges(blurred)
        edges    = self.threshold(mag, low_ratio=low_ratio, high_ratio=high_ratio)
        self.edges = edges

        if visualize:
            fig, axes = plt.subplots(1, 4, figsize=(18, 4))
            for ax, img, title in zip(
                axes,
                [gray, blurred, mag, edges],
                ['1. Grayscale', '2. Gaussian blur',
                 '3. Sobel magnitude', '4. Edge map (hysteresis)'],
            ):
                ax.imshow(img, cmap='gray')
                ax.set_title(title)
                ax.axis('off')
            plt.tight_layout()
            out = '/tmp/cv_pipeline.png'
            plt.savefig(out, dpi=110)
            print(f'[cv_pipeline] pipeline figure saved → {out}')
            plt.show()

        return edges