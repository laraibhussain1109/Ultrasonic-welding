"""Six stopped-position acquisition and independent per-view decisions."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone

import numpy as np

from .anomaly_tolerance import AnomalyDecision, process_heatmap
from .fixed_settings import InspectionSettings, ToleranceSettings
from .patchcore_spatial import SpatialPatchCore
from .stationary_roi import InvalidView, QualityEvidence, YoloROI, select_best_frame

ANGLES = (60, 120, 180, 240, 300, 360)


@dataclass(frozen=True)
class ViewInspectionResult:
    view_number: int
    angle: int
    timestamp: str
    original_image: np.ndarray
    roi_image: np.ndarray
    content_mask: np.ndarray
    roi_bounds: tuple[int, int, int, int]
    quality: QualityEvidence
    yolo_confidence: float
    patchcore_score: float
    heatmap: np.ndarray
    decision: AnomalyDecision
    tolerance: ToleranceSettings
    model_path: str
    inference_ms: float

    @property
    def verdict(self) -> str:
        return self.decision.verdict

    def retune(self, tolerance: ToleranceSettings) -> "ViewInspectionResult":
        return replace(self, decision=process_heatmap(self.heatmap, tolerance, self.content_mask), tolerance=tolerance)

    def metadata(self) -> dict:
        return {"view_number": self.view_number, "angle": self.angle, "timestamp": self.timestamp,
                "quality": asdict(self.quality), "quality_score": self.quality.score, "blur_score": self.quality.blur_score,
                "yolo_confidence": self.yolo_confidence, "roi_bounds": list(self.roi_bounds),
                "patchcore_score": self.patchcore_score, "model_path": self.model_path,
                "tolerance": asdict(self.tolerance), "heatmap_threshold": self.tolerance.heatmap_threshold,
                "pixel_threshold": self.tolerance.pixel_threshold, "percentage_threshold": self.tolerance.percentage_threshold,
                "inference_ms": self.inference_ms, **self.decision.statistics()}


@dataclass
class PartInspectionResult:
    part_id: str
    start_time: str
    started_monotonic: float
    views: dict[int, ViewInspectionResult] = field(default_factory=dict)
    end_time: str | None = None
    cycle_time_s: float | None = None
    fault: str | None = None

    @property
    def failed_views(self) -> list[int]:
        return sorted(angle for angle, view in self.views.items() if view.verdict == "FAIL")

    @property
    def final_verdict(self) -> str:
        if self.fault:
            return "FAULT"
        if set(self.views) != set(ANGLES):
            return "INCOMPLETE"
        return "FAIL" if self.failed_views else "PASS"

    def metadata(self) -> dict:
        return {"part_id": self.part_id, "start_time": self.start_time, "end_time": self.end_time,
                "cycle_time_s": self.cycle_time_s, "view_count": len(self.views), "final_verdict": self.final_verdict,
                "failed_views": self.failed_views, "fault": self.fault,
                "views": [self.views[a].metadata() for a in sorted(self.views)]}


class FixedViewPipeline:
    def __init__(self, settings: InspectionSettings, detector=None, model=None) -> None:
        self.settings = settings.validate()
        self.detector = detector or YoloROI(settings)
        self.model = model or SpatialPatchCore(settings)

    def ready(self) -> None:
        self.detector.ready()
        self.model.validate_ready()

    def inspect_frames(self, frames: list[np.ndarray], angle: int) -> ViewInspectionResult:
        if angle not in ANGLES:
            raise ValueError("Invalid fixed inspection angle")
        started = time.perf_counter()
        frame, _ = select_best_frame(frames, self.settings.quality)
        roi = self.detector.prepare(frame)
        heatmap, score, roi = self.model.infer(roi, angle)
        decision = process_heatmap(heatmap, self.settings.tolerance, roi.content_mask)
        return ViewInspectionResult(ANGLES.index(angle) + 1, angle, datetime.now(timezone.utc).isoformat(),
                                    roi.original, roi.image, roi.content_mask, roi.bounds, roi.quality,
                                    roi.yolo_confidence, score, heatmap.copy(), decision, self.settings.tolerance,
                                    str(self.settings.patchcore.path_for_angle(angle)),
                                    (time.perf_counter() - started) * 1000)


class FixedViewController:
    """No inferred angles: fit-check and stopped-position confirmations are mandatory.

    All methods are owned by one worker thread. Camera frames carry capture times.
    A moving event discards the entire unfinished burst; bad views never advance.
    """
    def __init__(self, settings: InspectionSettings) -> None:
        self.settings = settings.validate()
        self.part: PartInspectionResult | None = None
        self.state = "IDLE"
        self.angle: int | None = None
        self.stopped_at = 0.0
        self.burst: list[np.ndarray] = []
        self.last_frame_time = -1.0
        self.pending = False
        self.last_stop_angle: int | None = None
        self.motion_since_stop = True

    def begin(self, part_id: str, *, fit_check_complete: bool, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self.part is not None and self.part.final_verdict == "INCOMPLETE":
            raise ValueError("Finish or abort the active part before starting another")
        if fit_check_complete is not True or not part_id.strip():
            raise ValueError("A part ID and confirmed 360° PLC fit check are required")
        self.part = PartInspectionResult(part_id.strip(), datetime.now(timezone.utc).isoformat(), now)
        self.state, self.angle, self.pending = "WAITING FOR STOP", None, False
        self.burst.clear()
        self.last_stop_angle = None
        self.motion_since_stop = True

    @property
    def expected_angle(self) -> int | None:
        if self.part is None:
            return None
        return next((angle for angle in ANGLES if angle not in self.part.views), None)

    def stopped(self, angle: int, *, now: float | None = None) -> None:
        if self.part is not None and angle in self.part.views and angle == self.last_stop_angle:
            return
        if self.part is None or self.part.final_verdict != "INCOMPLETE":
            raise ValueError("No active incomplete part")
        if angle != self.expected_angle:
            raise ValueError(f"Expected stopped position {self.expected_angle}°, received {angle}°")
        if self.last_stop_angle is not None and angle != self.last_stop_angle and not self.motion_since_stop:
            raise ValueError("Motor-moving confirmation is required between distinct inspection positions")
        if self.angle == angle and self.state in {"SETTLING", "ACQUIRING", "INSPECTING"}:
            return  # A repeated PLC heartbeat must not reset acquisition.
        self.angle, self.stopped_at = angle, time.monotonic() if now is None else now
        self.last_stop_angle, self.motion_since_stop = angle, False
        self.state, self.pending = "SETTLING", False
        self.burst.clear()
        self.last_frame_time = -1.0

    def moving(self) -> None:
        self.motion_since_stop = True
        self.burst.clear()
        self.pending = False
        self.angle = None
        if self.part is not None and self.part.final_verdict == "INCOMPLETE":
            self.state = "WAITING FOR STOP"

    def frame(self, image: np.ndarray, captured_at: float) -> bool:
        if self.part is None or self.part.final_verdict != "INCOMPLETE" or self.angle is None or self.pending:
            return False
        if captured_at <= self.last_frame_time:
            return False
        start = self.stopped_at + self.settings.machine.settle_delay_s
        end = start + self.settings.machine.acquisition_time_s
        if captured_at < start:
            return False
        if captured_at > end:
            if len(self.burst) < self.settings.machine.minimum_burst_frames:
                self.invalid("INSUFFICIENT STATIONARY FRAMES")
                return False
            self.state, self.pending = "INSPECTING", True
            return True
        self.state = "ACQUIRING"
        self.last_frame_time = captured_at
        self.burst.append(image.copy())
        if len(self.burst) >= self.settings.camera.burst_frames:
            self.state, self.pending = "INSPECTING", True
            return True
        return False

    def accept(self, result: ViewInspectionResult, *, now: float | None = None) -> None:
        if not self.pending or self.part is None or result.angle != self.angle or result.angle != self.expected_angle:
            raise ValueError("Stale, duplicate, or unrequested view result")
        self.part.views[result.angle] = result
        self.burst.clear()
        self.pending = False
        self.angle = None
        if len(self.part.views) == 6:
            self.part.end_time = datetime.now(timezone.utc).isoformat()
            self.part.cycle_time_s = (time.monotonic() if now is None else now) - self.part.started_monotonic
            self.state = self.part.final_verdict
        else:
            self.state = "WAITING FOR STOP"

    def invalid(self, reason: str) -> None:
        number = ANGLES.index(self.expected_angle) + 1 if self.expected_angle else 0
        self.state = f"VIEW {number} — {reason}"
        self.burst.clear()
        self.pending = False
        self.angle = None

    def fault(self, reason: str, now: float | None = None) -> None:
        self.moving()
        if self.part is not None:
            self.part.fault = reason
            self.part.end_time = datetime.now(timezone.utc).isoformat()
            self.part.cycle_time_s = (time.monotonic() if now is None else now) - self.part.started_monotonic
        self.state = "SYSTEM FAULT: " + reason

    def tick(self, now: float) -> None:
        if self.part is None or self.part.final_verdict != "INCOMPLETE":
            return
        if now - self.part.started_monotonic > self.settings.machine.cycle_timeout_s:
            self.fault("SIX-VIEW CYCLE TIMEOUT", now)
        elif self.angle is not None and now - self.stopped_at > self.settings.machine.position_timeout_s:
            self.invalid("ACQUISITION TIMEOUT")
