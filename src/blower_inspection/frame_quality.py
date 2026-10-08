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
    # Keep native-resolution sharpness thresholds while bounding derivative and
    # local-variance scratch arrays, including on 4K training photographs.
    STRIP_ROWS = 256

    def __init__(self, minimum_sharpness: float = 60.0, max_glare_ratio: float = .20,
                 max_saturation_ratio: float = .12, max_dark_ratio: float = .55) -> None:
        self.minimum_sharpness = minimum_sharpness
        self.max_glare_ratio = max_glare_ratio
        self.max_saturation_ratio = max_saturation_ratio
        self.max_dark_ratio = max_dark_ratio

    def analyze(self, image: np.ndarray, valid_mask: np.ndarray | None = None) -> FrameQuality:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if not gray.size:
            raise ValueError("frame-quality valid_mask contains no pixels")
        qualified = None
        if valid_mask is not None:
            valid_mask = np.asarray(valid_mask, dtype=bool)
            if valid_mask.shape != gray.shape:
                raise ValueError("frame-quality valid_mask must match the image shape")
            if not np.any(valid_mask):
                raise ValueError("frame-quality valid_mask contains no pixels")
            # Exclude padding boundaries from derivatives, as before.
            valid_mask = valid_mask.astype(np.uint8)
            eroded = cv2.erode(valid_mask, np.ones((3, 3), np.uint8))
            qualified = eroded if cv2.countNonZero(eroded) else valid_mask
        count, lap_mean, lap_m2, gradient_sum = 0, 0.0, 0.0, 0.0
        pixel_count = saturated = dark_pixels = glare_pixels = 0
        height, width = gray.shape
        for start in range(0, height, self.STRIP_ROWS):
            end = min(height, start + self.STRIP_ROWS)
            # A seven-row halo preserves the 15x15 glare filter and derivative
            # neighborhoods at strip seams; outer borders still use OpenCV's
            # original reflection behavior.
            top, bottom = max(0, start - 7), min(height, end + 7)
            strip = gray[top:bottom]
            center = slice(start - top, end - top)
            mask = qualified[start:end] if qualified is not None else None
            n = cv2.countNonZero(mask) if mask is not None else (end - start) * width
            if n:
                lap_image = cv2.Laplacian(strip, cv2.CV_32F)[center]
                mean, std = cv2.meanStdDev(lap_image, mask=mask)
                mean, variance = float(mean[0, 0]), float(std[0, 0]) ** 2
                delta = mean - lap_mean
                lap_m2 += n * variance + delta * delta * count * n / (count + n)
                lap_mean += delta * n / (count + n)
                count += n
                del lap_image
                gx = cv2.Sobel(strip, cv2.CV_32F, 1, 0)
                gy = cv2.Sobel(strip, cv2.CV_32F, 0, 1)
                np.square(gx, out=gx)
                np.square(gy, out=gy)
                gx += gy
                gradient_sum += cv2.mean(gx[center], mask=mask)[0] * n
                del gx, gy
            mask = valid_mask[start:end] if valid_mask is not None else None
            pixels = gray[start:end]
            pixel_count += cv2.countNonZero(mask) if mask is not None else pixels.size
            saturated += np.count_nonzero((pixels >= 250) & mask) if mask is not None else np.count_nonzero(pixels >= 250)
            dark_pixels += np.count_nonzero((pixels <= 5) & mask) if mask is not None else np.count_nonzero(pixels <= 5)
            values = strip.astype(np.float32)
            local_mean = cv2.blur(values, (15, 15))
            np.square(values, out=values)
            local_std = cv2.blur(values, (15, 15))
            np.square(local_mean, out=local_mean)
            local_std -= local_mean
            np.maximum(local_std, 0, out=local_std)
            np.sqrt(local_std, out=local_std)
            glare = (pixels >= 225) & (local_std[center] < 18)
            glare_pixels += np.count_nonzero(glare & mask) if mask is not None else np.count_nonzero(glare)
            del values, local_mean, local_std, glare
        lap = lap_m2 / count
        tenengrad = gradient_sum / count
        saturation, dark, glare = saturated / pixel_count, dark_pixels / pixel_count, glare_pixels / pixel_count
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
