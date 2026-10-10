"""Hybrid nuisance filtering and tolerance in the calibrated heatmap's pixels."""
from dataclasses import dataclass

import cv2
import numpy as np

from .inspection_fusion import ProductionDecision


@dataclass(frozen=True)
class HeatmapArea:
    mask: np.ndarray
    valid_mask: np.ndarray
    area_px: int
    valid_area_px: int
    percentage: float
    sections: tuple[int, ...]
    decision: ProductionDecision


def evaluate_heatmap_area(raw: np.ndarray, valid: np.ndarray, reflection: np.ndarray,
                          geometry_mask: np.ndarray, *, patch_threshold: float,
                          image_score: float, fail_threshold: float,
                          geometry_score: float, geometry_threshold: float,
                          tolerance_percent: float, min_component_px: int,
                          section_count: int, scratch_max_width_px: float = 2.0,
                          scratch_min_aspect: float = 8.0) -> HeatmapArea:
    if raw.ndim != 2 or any(mask.shape != raw.shape for mask in (valid, reflection, geometry_mask)):
        raise ValueError("Heatmap and hybrid masks must have matching two-dimensional shapes")
    if not np.isfinite(raw).all() or not np.isfinite(patch_threshold) or patch_threshold <= 0:
        raise ValueError("PatchCore heatmap and calibrated patch threshold must be finite and positive")
    if not 0 <= tolerance_percent <= 100:
        raise ValueError("Heatmap area tolerance must be between 0 and 100 percent")
    corroborated = geometry_mask.astype(bool) & (geometry_score >= geometry_threshold)
    # Preserve reflected pixels when local fin deformation corroborates them.
    inspection = valid.astype(bool) & ~(reflection.astype(bool) & ~corroborated)
    original = (raw >= patch_threshold) & inspection
    count, labels, stats, _ = cv2.connectedComponentsWithStats(original.astype(np.uint8), 8)
    filtered = np.zeros_like(original)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(1, min_component_px):
            continue
        component = labels == label
        # minAreaRect works for diagonal as well as vertical thin scratches.
        points = np.column_stack(np.nonzero(component)[::-1]).astype(np.float32)
        _center, sides, _angle = cv2.minAreaRect(points)
        short, long = sorted(value + 1 for value in sides)
        scratch = short <= scratch_max_width_px and long / short >= scratch_min_aspect
        if scratch and not np.any(component & corroborated):
            continue
        filtered |= component
    area = int(filtered.sum())
    valid_area = int(inspection.sum())
    percentage = 100 * area / valid_area if valid_area else 0.0
    sections = tuple(index for index, mask in enumerate(np.array_split(filtered, section_count, axis=1))
                     if np.any(mask))
    if not valid_area:
        status, reasons = "VIEW INVALID", ("NO_INSPECTABLE_AREA",)
    elif area == 0 or percentage < tolerance_percent:
        status, reasons = "PASS", ("WITHIN_HEATMAP_AREA_TOLERANCE",)
    elif np.any(filtered & corroborated):
        status, reasons = "FAIL", ("HEATMAP_AREA_EXCEEDED", "GEOMETRY_DEFORMATION")
    elif image_score >= fail_threshold:
        status, reasons = "FAIL", ("HEATMAP_AREA_EXCEEDED", "STRONG_PATCHCORE_ANOMALY")
    else:
        status, reasons = "CANDIDATE", ("HEATMAP_AREA_EXCEEDED", "PATCHCORE_ANOMALY")
    decision = ProductionDecision(status, reasons, status == "FAIL", status == "CANDIDATE", filtered)
    return HeatmapArea(filtered, inspection, area, valid_area, percentage, sections, decision)
