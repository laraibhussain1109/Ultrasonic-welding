"""Conservative translation/rotation registration with diagnostics."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RegistrationResult:
    aligned_image: np.ndarray
    success: bool
    correlation: float
    translation_x: float
    translation_y: float
    rotation_deg: float
    reference_index: int | None = None


def register_to_reference(image: np.ndarray, reference: np.ndarray, *, min_correlation: float = 0.55,
                          max_translation_ratio: float = 0.06, max_rotation_deg: float = 3.0,
                          reference_index: int | None = None) -> RegistrationResult:
    if image.shape[:2] != reference.shape[:2]:
        image = cv2.resize(image, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)
    def structural(value: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(value, cv2.COLOR_BGR2GRAY) if value.ndim == 3 else value
        gray = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32) / 255.0
        return cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    ref_s, image_s = structural(reference), structural(image)
    shift, _response = cv2.phaseCorrelate(ref_s, image_s)
    warp = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], dtype=np.float32)
    correlation = 0.0
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6)
        correlation, warp = cv2.findTransformECC(ref_s, image_s, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5)
    except cv2.error:
        pass
    rotation = float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0])))
    tx, ty = float(warp[0, 2]), float(warp[1, 2])
    h, w = reference.shape[:2]
    valid = (np.isfinite(correlation) and correlation >= min_correlation and
             abs(tx) <= w * max_translation_ratio and abs(ty) <= h * max_translation_ratio and
             abs(rotation) <= max_rotation_deg)
    aligned = cv2.warpAffine(image, warp, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REFLECT) if valid else image.copy()
    return RegistrationResult(aligned, bool(valid), float(correlation), tx, ty, rotation, reference_index)
