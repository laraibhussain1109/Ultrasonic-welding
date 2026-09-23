"""Non-generative frame quality gate for rotating blower photographs."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameQuality:
    valid: bool
    sharpness: float
    tenengrad: float
    blur_score: float
    saturation_ratio: float
    dark_ratio: float
    glare_ratio: float
    reasons: tuple[str, ...]


class FrameQualityAnalyzer:
    def __init__(self, minimum_sharpness: float = 60.0, max_glare_ratio: float = .20,
                 max_saturation_ratio: float = .12, max_dark_ratio: float = .55) -> None:
        self.minimum_sharpness = minimum_sharpness
        self.max_glare_ratio = max_glare_ratio
        self.max_saturation_ratio = max_saturation_ratio
        self.max_dark_ratio = max_dark_ratio

    def analyze(self, image: np.ndarray, valid_mask: np.ndarray | None = None) -> FrameQuality:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if valid_mask is None:
            valid_mask = np.ones(gray.shape, dtype=bool)
        else:
            valid_mask = np.asarray(valid_mask, dtype=bool)
            if valid_mask.shape != gray.shape:
                raise ValueError("frame-quality valid_mask must match the image shape")
        if not np.any(valid_mask):
            raise ValueError("frame-quality valid_mask contains no pixels")
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        # Derivatives are measured only over real ROI pixels. Letterbox padding
        # must not make a wide, black blower look underexposed or artificially
        # sharp at the padding boundary.
        eroded = cv2.erode(valid_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        qualified = eroded if np.any(eroded) else valid_mask
        lap_image = cv2.Laplacian(gray, cv2.CV_64F)
        lap = float(np.var(lap_image[qualified]))
        tenengrad = float(np.mean((gx * gx + gy * gy)[qualified]))
        saturation = float(np.mean((gray >= 250)[valid_mask]))
        dark = float(np.mean((gray <= 5)[valid_mask]))
        local_std = np.sqrt(np.maximum(cv2.blur(gray.astype(np.float32) ** 2, (15, 15)) -
                                       cv2.blur(gray.astype(np.float32), (15, 15)) ** 2, 0))
        glare = float(np.mean(((gray >= 225) & (local_std < 18))[valid_mask]))
        reasons = []
        if lap < self.minimum_sharpness:
            reasons.extend(("MOTION_BLUR", "LOW_SHARPNESS"))
        if saturation > self.max_saturation_ratio:
            reasons.append("OVEREXPOSED")
        if dark > self.max_dark_ratio:
            reasons.append("UNDEREXPOSED")
        if glare > self.max_glare_ratio:
            reasons.append("EXCESSIVE_GLARE")
        return FrameQuality(not reasons, lap, tenengrad, 1.0 / (1.0 + lap), saturation, dark, glare, tuple(reasons))
