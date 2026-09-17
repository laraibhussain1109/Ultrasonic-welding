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

    def analyze(self, image: np.ndarray) -> FrameQuality:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        tenengrad = float(np.mean(gx * gx + gy * gy))
        saturation = float(np.mean(gray >= 250))
        dark = float(np.mean(gray <= 5))
        local_std = np.sqrt(np.maximum(cv2.blur(gray.astype(np.float32) ** 2, (15, 15)) -
                                       cv2.blur(gray.astype(np.float32), (15, 15)) ** 2, 0))
        glare = float(np.mean((gray >= 225) & (local_std < 18)))
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

