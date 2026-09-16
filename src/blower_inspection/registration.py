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
    failure_reason: str | None = None


def register_to_reference(image: np.ndarray, reference: np.ndarray, *, min_correlation: float = 0.55,
                          max_translation_ratio: float = 0.06, max_rotation_deg: float = 3.0,
                          reference_index: int | None = None,
                          max_working_dimension: int = 512) -> RegistrationResult:
    if image.shape[:2] != reference.shape[:2]:
        image = cv2.resize(image, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)
    height, width = reference.shape[:2]
    working_scale = min(1.0, max_working_dimension / max(height, width))
    working_size = (max(32, int(round(width * working_scale))),
                    max(32, int(round(height * working_scale))))

    def structural(value: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(value, cv2.COLOR_BGR2GRAY) if value.ndim == 3 else value
        if gray.shape[::-1] != working_size:
            gray = cv2.resize(gray, working_size, interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32) / 255.0
        gradient = cv2.magnitude(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1),
        )
        return cv2.normalize(gradient, None, 0.0, 1.0, cv2.NORM_MINMAX)
    ref_s, image_s = structural(reference), structural(image)
    shift, _response = cv2.phaseCorrelate(ref_s, image_s)
    warp = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], dtype=np.float32)
    correlation = 0.0
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-5)
        correlation, warp = cv2.findTransformECC(ref_s, image_s, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5)
    except cv2.error:
        pass
    rotation = float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0])))
    # ECC was solved on the bounded working image; translation must be restored
    # to full-resolution crop coordinates before safety checks and warping.
    tx = float(warp[0, 2]) / working_scale
    ty = float(warp[1, 2]) / working_scale
    warp[0, 2], warp[1, 2] = tx, ty
    h, w = reference.shape[:2]
    failures = []
    if not np.isfinite(correlation) or correlation < min_correlation:
        failures.append("LOW_CORRELATION")
    if abs(tx) > w * max_translation_ratio or abs(ty) > h * max_translation_ratio:
        failures.append("TRANSLATION_LIMIT")
    if abs(rotation) > max_rotation_deg:
        failures.append("ROTATION_LIMIT")
    valid = not failures
    aligned = cv2.warpAffine(image, warp, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REFLECT) if valid else image.copy()
    return RegistrationResult(aligned, bool(valid), float(correlation), tx, ty, rotation,
                              reference_index, "+".join(failures) or None)
