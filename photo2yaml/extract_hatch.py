"""
Extract diagonal hatched (斜线阴影) regions from a floor plan image using Gabor filters.

Detects regions with diagonal line textures at 45° and -45° orientations,
which correspond to furniture/obstacle markings in the floor plan.
"""

import cv2
import numpy as np
import os


def build_gabor_kernel(ksize, sigma, theta, lambd, gamma, psi=0):
    """Create a single Gabor kernel."""
    kernel = cv2.getGaborKernel(
        (ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F
    )
    # Normalize kernel
    kernel /= kernel.sum() if kernel.sum() != 0 else 1
    return kernel


def apply_gabor_filter(img, kernel):
    """Apply Gabor filter and return magnitude response."""
    filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
    return np.abs(filtered)


def extract_hatch_mask(
    img_path,
    output_path,
    ksize=21,
    sigma=4.0,
    lambd=8.0,
    gamma=0.5,
    psi=0,
    morph_kernel_size=5,
    min_area=200,
):
    """
    Extract diagonal hatched regions from a floor plan image.

    Parameters
    ----------
    img_path : str
        Path to input grayscale image.
    output_path : str
        Path to save the resulting binary mask.
    ksize : int
        Gabor kernel size (must be odd).
    sigma : float
        Standard deviation of Gaussian envelope.
    lambd : float
        Wavelength of sinusoidal factor.
    gamma : float
        Spatial aspect ratio (ellipticity).
    psi : float
        Phase offset.
    morph_kernel_size : int
        Kernel size for morphological operations.
    min_area : int
        Minimum connected component area to keep.
    """
    # 1. Read image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    img_float = img.astype(np.float32)

    # 2. Build Gabor kernels for 45° and -45°
    theta1 = np.pi / 4   # 45 degrees
    theta2 = 3 * np.pi / 4  # -45 degrees (135 degrees)

    kernel1 = build_gabor_kernel(ksize, sigma, theta1, lambd, gamma, psi)
    kernel2 = build_gabor_kernel(ksize, sigma, theta2, lambd, gamma, psi)

    # 3. Apply filters
    resp1 = apply_gabor_filter(img_float, kernel1)
    resp2 = apply_gabor_filter(img_float, kernel2)

    # 4. Combine responses (element-wise max)
    combined = np.maximum(resp1, resp2)

    # 5. Threshold - use Otsu on the combined response
    combined_norm = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(combined_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 6. Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))

    # Close small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Open small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 7. Remove small connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            mask[labels == i] = 0

    # 8. Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, mask)

    print(f"Mask saved to: {output_path}")
    print(f"Mask shape: {mask.shape}")
    print(f"Hatched region pixels: {np.sum(mask > 0)} / {mask.size} "
          f"({100 * np.sum(mask > 0) / mask.size:.1f}%)")

    return mask


if __name__ == "__main__":
    input_path = r"e:\python\PythonProject\ymj_TS\photo2yaml\photo\room_real1.png"
    output_path = r"e:\python\PythonProject\ymj_TS\photo2yaml\output\masks\hatch_mask.png"

    mask = extract_hatch_mask(
        img_path=input_path,
        output_path=output_path,
        ksize=21,
        sigma=4.0,
        lambd=8.0,
        gamma=0.5,
        morph_kernel_size=5,
        min_area=200,
    )
