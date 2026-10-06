"""Spatial tolerance decisions independent of model inference and image score."""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .fixed_settings import InspectionSettings, ToleranceSettings


@dataclass(frozen=True)
class AnomalyRegion:
    area_pixels: int
    bounding_box: tuple[int, int, int, int]


@dataclass(frozen=True)
class AnomalyDecision:
    candidate_mask: np.ndarray
    valid_mask: np.ndarray
    filtered_mask: np.ndarray
    raw_anomaly_pixels: int
    valid_candidate_pixels: int
    filtered_anomaly_pixels: int
    valid_roi_pixels: int
    largest_component_pixels: int
    anomaly_percentage: float
    regions: tuple[AnomalyRegion, ...]
    verdict: str

    def statistics(self) -> dict:
        return {"raw_anomaly_pixels": self.raw_anomaly_pixels, "valid_candidate_pixels": self.valid_candidate_pixels,
                "filtered_anomaly_pixels": self.filtered_anomaly_pixels, "valid_roi_pixels": self.valid_roi_pixels,
                "largest_component_pixels": self.largest_component_pixels, "anomaly_percentage": self.anomaly_percentage,
                "component_count": len(self.regions), "verdict": self.verdict,
                "regions": [{"area_pixels": r.area_pixels, "bounding_box": list(r.bounding_box),
                             "width": r.bounding_box[2], "height": r.bounding_box[3]} for r in self.regions]}


def process_heatmap(heatmap: np.ndarray, settings: ToleranceSettings,
                    content_mask: np.ndarray | None = None) -> AnomalyDecision:
    InspectionSettings(tolerance=settings).validate()
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim != 2 or heatmap.size == 0 or not np.isfinite(heatmap).all():
        raise ValueError("PatchCore heatmap must be a nonempty, finite 2-D array")
    h, w = heatmap.shape
    valid = np.ones((h, w), dtype=bool) if content_mask is None else np.asarray(content_mask, dtype=bool).copy()
    if valid.shape != heatmap.shape:
        raise ValueError("Content mask and heatmap dimensions differ")
    ys, xs = np.where(valid)
    if not len(xs):
        raise ValueError("ROI contains no valid content pixels")
    # Margins apply to the real letterboxed content, never to black padding.
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    left = x0 + math.ceil((x1 - x0) * settings.ignore_left_percent / 100)
    right = x1 - math.ceil((x1 - x0) * settings.ignore_right_percent / 100)
    top = y0 + math.ceil((y1 - y0) * settings.ignore_top_percent / 100)
    bottom = y1 - math.ceil((y1 - y0) * settings.ignore_bottom_percent / 100)
    border = np.zeros_like(valid)
    border[top:bottom, left:right] = True
    valid &= border
    count = int(valid.sum())
    if count == 0:
        raise ValueError("Ignored borders leave no valid pixels")
    candidate = heatmap >= settings.heatmap_threshold
    mask = (candidate & valid).astype(np.uint8)
    for size, op in ((settings.opening_kernel, cv2.MORPH_OPEN), (settings.closing_kernel, cv2.MORPH_CLOSE)):
        if size:
            mask = cv2.morphologyEx(mask, op, np.ones((size, size), np.uint8))
            mask &= valid.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(valid)
    regions = []
    for index in range(1, n):
        x, y, width, height, area = (int(v) for v in stats[index])
        if area >= settings.minimum_component_pixels:
            filtered |= labels == index
            regions.append(AnomalyRegion(area, (x, y, width, height)))
    pixels = int(filtered.sum())
    percent = pixels / count * 100.0
    px_fail = pixels >= settings.pixel_threshold
    percent_fail = percent >= settings.percentage_threshold
    failed = {"PIXEL": px_fail, "PERCENTAGE": percent_fail,
              "BOTH": px_fail and percent_fail, "EITHER": px_fail or percent_fail}[settings.decision_mode]
    return AnomalyDecision(candidate, valid, filtered, int(candidate.sum()), int((candidate & valid).sum()),
                           pixels, count, max((r.area_pixels for r in regions), default=0), percent,
                           tuple(regions), "FAIL" if failed else "PASS")


def heatmap_image(heatmap: np.ndarray, display_max: float) -> np.ndarray:
    """Fixed raw-distance color scale; no per-frame min/max normalization."""
    levels = np.clip(np.asarray(heatmap) / display_max * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(levels, cv2.COLORMAP_TURBO)


def marked_image(roi: np.ndarray, decision: AnomalyDecision, *, show_ignored: bool = True) -> np.ndarray:
    image = roi.copy()
    if show_ignored:
        ignored = ~decision.valid_mask
        image[ignored] = (image[ignored].astype(np.float32) * .35 + 110 * .65).astype(np.uint8)
    marked = decision.filtered_mask
    image[marked] = (image[marked].astype(np.float32) * .65 + np.array([0, 0, 255]) * .35).astype(np.uint8)
    contours, _ = cv2.findContours(marked.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
    return image
