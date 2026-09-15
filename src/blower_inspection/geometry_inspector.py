"""Calibrated structural inspection of horizontal cylindrical blower fins."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .trainer import broken_fin_mask, smooth_reflection_mask


@dataclass(frozen=True)
class GeometryEvidence:
    """Scores are anomaly severities: zero is normal and one is reject-level."""
    score: float
    orientation_score: float
    pitch_score: float
    continuity_score: float
    broken_fin_score: float
    periodicity_score: float
    support_rib_confidence: float
    defect_mask: np.ndarray
    candidate_regions: list[tuple[int, int, int, int]] = field(default_factory=list)
    valid: bool = True


@dataclass(frozen=True)
class GlareEvidence:
    score: float
    mask: np.ndarray


def inspection_band_mask(shape: tuple[int, int], top: float, bottom: float) -> np.ndarray:
    if not 0 <= top < bottom <= 1:
        raise ValueError("inspection band ratios must satisfy 0 <= top < bottom <= 1")
    mask = np.zeros(shape, bool)
    mask[int(shape[0] * top):max(int(shape[0] * bottom), int(shape[0] * top) + 1)] = True
    return mask


def support_rib_mask(image: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gx = np.abs(cv2.Sobel(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_32F, 1, 0))
    threshold = max(15.0, float(np.percentile(gx, 88)))
    persistence = np.mean(gx >= threshold, axis=0)
    columns = persistence >= max(0.16, float(np.percentile(persistence, 90)))
    width = max(3, gray.shape[1] // 120)
    columns = cv2.dilate(columns.astype(np.uint8)[None], np.ones((1, width), np.uint8))[0].astype(bool)
    return np.broadcast_to(columns, gray.shape).copy(), float(np.clip(np.max(persistence) / 0.45, 0, 1))


def _features(image: np.ndarray, top: float, bottom: float) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    band = inspection_band_mask(gray.shape, top, bottom)
    ribs, rib_confidence = support_rib_mask(gray)
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gx, gy)
    qualified = band & ~ribs
    threshold = max(12.0, float(np.percentile(magnitude[qualified], 70))) if np.any(qualified) else 12.0
    edges = (magnitude >= threshold) & qualified
    # Horizontal physical edges have a vertical gradient. Gradient orientation
    # is converted to line orientation around zero degrees.
    angle = np.degrees(np.arctan2(gy, gx))
    line_angle = ((angle + 90) % 180) - 90
    horizontal_weight = np.abs(gy) * qualified
    orientation = float(np.sum(np.abs(line_angle) * horizontal_weight) / (np.sum(horizontal_weight) + 1e-6))
    signal = np.mean(np.abs(gy) * qualified, axis=1)
    signal -= np.mean(signal)
    ac = np.correlate(signal, signal, mode="full")[len(signal)-1:]
    ac /= ac[0] + 1e-6
    min_lag = max(2, gray.shape[0] // 100)
    max_lag = max(min_lag + 1, gray.shape[0] // 4)
    region = ac[min_lag:max_lag]
    periodicity = float(np.max(region)) if region.size else 0.0
    pitch = float(min_lag + np.argmax(region)) if region.size else 0.0
    edge_rows = np.mean((np.abs(gy) >= max(10.0, np.percentile(np.abs(gy)[qualified], 70))) & qualified, axis=1)
    continuity = float(np.percentile(edge_rows[band[:, 0]], 75)) if np.any(band[:, 0]) else 0.0
    return {"orientation": orientation, "pitch": pitch, "periodicity": periodicity,
            "continuity": continuity, "rib_confidence": rib_confidence}, edges, ribs


class FinGeometryInspector:
    def __init__(self, calibration: dict | None = None, *, band_top: float = 0.18, band_bottom: float = 0.82) -> None:
        self.calibration = calibration or {}
        self.band_top, self.band_bottom = band_top, band_bottom

    @staticmethod
    def calibrate(images: list[np.ndarray], *, band_top: float = 0.18, band_bottom: float = 0.82) -> dict:
        if len(images) < 3:
            raise ValueError("Geometry calibration requires at least three known-good images")
        rows = [_features(image, band_top, band_bottom)[0] for image in images]
        output = {"version": 1, "sample_count": len(images)}
        for name in ("orientation", "pitch", "periodicity", "continuity", "rib_confidence"):
            values = np.asarray([row[name] for row in rows])
            median = float(np.median(values))
            output[name] = {"median": median, "mad": float(max(np.median(np.abs(values - median)), 1e-3)),
                            "p01": float(np.quantile(values, .01)), "p99": float(np.quantile(values, .99))}
        return output

    def inspect(self, image: np.ndarray) -> GeometryEvidence:
        features, _edges, ribs = _features(image, self.band_top, self.band_bottom)
        def high(name: str, value: float) -> float:
            item = self.calibration.get(name)
            return float(np.clip(abs(value - item["median"]) / max(6 * 1.4826 * item["mad"], 1e-3), 0, 2)) if item else 0.0
        orientation = high("orientation", features["orientation"])
        pitch = high("pitch", features["pitch"])
        periodicity = high("periodicity", features["periodicity"])
        continuity = high("continuity", features["continuity"])
        broken = broken_fin_mask(image, image.shape[:2])
        broken &= ~ribs & inspection_band_mask(image.shape[:2], self.band_top, self.band_bottom)
        broken_score = float(np.clip(np.count_nonzero(broken) / max(image.size / 3 * .001, 1), 0, 2))
        score = max(broken_score, orientation, pitch, periodicity, continuity,
                    .55 * orientation + .35 * periodicity + .3 * continuity)
        count, _labels, stats, _ = cv2.connectedComponentsWithStats(broken.astype(np.uint8), 8)
        regions = [tuple(map(int, stats[i, :4])) for i in range(1, count)]
        return GeometryEvidence(float(score), orientation, pitch, continuity, broken_score,
                                periodicity, features["rib_confidence"], broken, regions,
                                valid=bool(features["rib_confidence"] >= .15))


def glare_evidence(image: np.ndarray, output_shape: tuple[int, int], *, top: float = .18, bottom: float = .82) -> GlareEvidence:
    mask = smooth_reflection_mask(image, output_shape) & inspection_band_mask(output_shape, top, bottom)
    # Coverage is deliberately saturating: a broad smooth highlight is strong
    # optical evidence, but fusion never subtracts it from severe geometry.
    score = float(np.clip(np.count_nonzero(mask) / max(np.count_nonzero(inspection_band_mask(output_shape, top, bottom)) * .18, 1), 0, 1))
    return GlareEvidence(score, mask)
