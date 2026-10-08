"""Quality selection and strict YOLO-only ROI normalization shared by all paths."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .fixed_settings import InspectionSettings, QualitySettings
from .frame_quality import FrameQualityAnalyzer
from .roi_stabilizer import ROIStabilizer
from .yolo_tracking import YoloByteTrackDetector


class InvalidView(ValueError):
    """Retryable quality/ROI failure, distinct from a failed defect verdict."""


@dataclass(frozen=True)
class QualityEvidence:
    valid: bool
    score: float
    blur_score: float
    mean_intensity: float
    dark_percent: float
    saturated_percent: float
    reasons: tuple[str, ...]


def assess_quality(frame: np.ndarray, settings: QualitySettings, mask=None) -> QualityEvidence:
    if frame is None or frame.size == 0:
        raise InvalidView("EMPTY FRAME")
    result = FrameQualityAnalyzer(settings.minimum_blur_score, max_glare_ratio=1.0,
                                  max_saturation_ratio=settings.maximum_saturated_percent / 100,
                                  max_dark_ratio=settings.maximum_dark_percent / 100).analyze(frame, mask)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = cv2.mean(gray, mask=None if mask is None else np.asarray(mask, dtype=np.uint8))[0]
    reasons = list(result.reasons)
    if mean < settings.minimum_mean_intensity:
        reasons.append("UNDEREXPOSED MEAN")
    if mean > settings.maximum_mean_intensity:
        reasons.append("OVEREXPOSED MEAN")
    normalized_blur = result.sharpness / (result.sharpness + max(settings.minimum_blur_score, 1.0))
    score = max(0.0, min(100.0, 100 * (settings.blur_weight * normalized_blur -
                                  settings.exposure_penalty * (result.dark_ratio + result.saturation_ratio))))
    return QualityEvidence(not reasons, score, result.sharpness, mean, result.dark_ratio * 100,
                           result.saturation_ratio * 100, tuple(reasons))


def select_best_frame(frames: list[np.ndarray], settings: QualitySettings) -> tuple[np.ndarray, QualityEvidence]:
    candidates, reasons = [], set()
    for frame in frames:
        quality = assess_quality(frame, settings)
        if quality.valid:
            candidates.append((frame, quality))
        else:
            reasons.update(quality.reasons)
    if not candidates:
        raise InvalidView("IMAGE QUALITY FAILED: " + ", ".join(sorted(reasons or {"NO FRAMES"})))
    return max(candidates, key=lambda item: (item[1].score, item[1].blur_score))


@dataclass(frozen=True)
class PreparedROI:
    original: np.ndarray | None
    image: np.ndarray
    content_mask: np.ndarray
    bounds: tuple[int, int, int, int]
    yolo_confidence: float
    quality: QualityEvidence


class YoloROI:
    def __init__(self, settings: InspectionSettings, model=None) -> None:
        self.settings = settings
        self.loader = YoloByteTrackDetector(settings.yolo.model_path, settings.yolo.confidence)
        if model is not None:
            self.loader._model = model

    def ready(self) -> None:
        self.loader._load()

    def prepare(self, frame: np.ndarray, *, retain_original: bool = True) -> PreparedROI:
        s = self.settings
        quality = assess_quality(frame, s.quality)
        if not quality.valid:
            raise InvalidView("IMAGE QUALITY FAILED: " + ", ".join(quality.reasons))
        options = dict(conf=s.yolo.confidence, iou=s.yolo.iou, verbose=False)
        if s.yolo.class_id >= 0:
            options["classes"] = [s.yolo.class_id]
        results = self.loader._load().predict(frame, **options)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            raise InvalidView("ROI NOT DETECTED")
        boxes = results[0].boxes
        confidences = boxes.conf.detach().cpu().numpy()
        coordinates = boxes.xyxy.detach().cpu().numpy()
        height, width = frame.shape[:2]
        candidates = []
        for box, confidence in zip(coordinates, confidences):
            if not np.isfinite(box).all() or not np.isfinite(confidence) or confidence < s.yolo.confidence:
                continue
            x0, y0, x1, y1 = (float(v) for v in box)
            # Never silently clip a partial part into an apparently valid ROI.
            if x0 <= 0 or y0 <= 0 or x1 >= width or y1 >= height or x1 <= x0 or y1 <= y0:
                continue
            candidates.append((float(confidence), (x0, y0, x1, y1)))
        if not candidates:
            raise InvalidView("ROI PARTIAL OR CONFIDENCE TOO LOW")
        confidence, (x0, y0, x1, y1) = max(candidates, key=lambda item: item[0])
        px, py = (x1 - x0) * s.yolo.padding_x_percent / 100, (y1 - y0) * s.yolo.padding_y_percent / 100
        # Padding may be clipped; the actual detector box above may never be clipped.
        x0, y0 = max(0, int(np.floor(x0 - px))), max(0, int(np.floor(y0 - py)))
        x1, y1 = min(width, int(np.ceil(x1 + px))), min(height, int(np.ceil(y1 + py)))
        bounds = (x0, y0, x1 - x0, y1 - y0)
        canonical = ROIStabilizer((s.patchcore.input_width, s.patchcore.input_height),
                                 mode="fixed_after_detection", padding_ratio=0).canonicalize(frame, bounds)
        mask = np.zeros(canonical.image.shape[:2], bool)
        x, y, w, h = canonical.content_bounds
        mask[y:y + h, x:x + w] = True
        crop_quality = assess_quality(canonical.image, s.quality, mask)
        if not crop_quality.valid:
            raise InvalidView("ROI IMAGE QUALITY FAILED: " + ", ".join(crop_quality.reasons))
        return PreparedROI(frame.copy() if retain_original else None, canonical.image, mask, bounds, confidence, crop_quality)
