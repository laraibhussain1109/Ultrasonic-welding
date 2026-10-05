"""Deterministic PLC-phase-locked structural production inspection.

This module deliberately contains no PatchCore/DINO decision path.  Learned
anomaly providers may add supporting evidence, but only repeatable structural
and geometry evidence can reject a component.
"""

from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


ANGLES = (60, 120, 180, 240, 300, 360)
PREPROCESSING_VERSION = "phase-structural-v1"
MODEL_VERSION = 1


class Verdict(str, Enum):
    PASS = "PASS"
    CANDIDATE = "CANDIDATE"
    FAIL = "FAIL"
    VIEW_INVALID = "VIEW_INVALID"
    FITMENT_FAIL = "FITMENT_FAIL"
    INSPECTION_INVALID = "INSPECTION_INVALID"


class InspectionState(str, Enum):
    IDLE = "IDLE"
    PART_PRESENT = "PART_PRESENT"
    FITMENT_ROTATION = "FITMENT_ROTATION"
    FITMENT_ANALYSIS = "FITMENT_ANALYSIS"
    FITMENT_PASS = "FITMENT_PASS"
    WAIT_PHASE = "WAIT_PHASE"
    SETTLING = "SETTLING"
    CAPTURING_BURST = "CAPTURING_BURST"
    ANALYZING_PHASE = "ANALYZING_PHASE"
    PHASE_COMPLETE = "PHASE_COMPLETE"
    FINAL_HOME_CHECK = "FINAL_HOME_CHECK"
    FINALIZING = "FINALIZING"
    PASS = "PASS"
    FAIL = "FAIL"
    INVALID = "INVALID"


@dataclass(frozen=True)
class PhaseLockedConfig:
    inspection_angles: tuple[int, ...] = ANGLES
    settle_delay_ms: int = 200
    burst_frame_count: int = 7
    minimum_sharpness: float = 60.0
    minimum_qualified_frames: int = 5
    temporal_confirmation_frames: int = 3
    registration_max_translation_ratio: float = .02
    registration_max_rotation_deg: float = 1.0
    registration_min_correlation: float = .55
    fitment_runout_threshold: float = 8.0
    fitment_axial_threshold: float = 6.0
    golden_noise_floor: float = .035
    golden_candidate_threshold: float = 4.0
    golden_fail_threshold: float = 7.0
    edge_candidate_threshold: float = .10
    edge_fail_threshold: float = .20
    geometry_candidate_threshold: float = .45
    geometry_fail_threshold: float = .85
    glare_rejection_enabled: bool = True
    efficientad_enabled: bool = False
    minimum_region_area: int = 24
    consensus_iou: float = .18
    closure_max_translation_ratio: float = .02
    closure_min_correlation: float = .55


@dataclass(frozen=True)
class RegistrationResult:
    valid: bool
    correlation: float
    translation_x: float
    translation_y: float
    rotation_deg: float
    image: np.ndarray
    reason: str = ""


@dataclass
class Defect:
    phase: int
    angle: int
    defect: str
    bbox: tuple[int, int, int, int]
    confidence: float
    evidence: list[str]
    persistence_count: int = 1
    qualified_frames: int = 1

    def to_dict(self) -> dict:
        value = asdict(self)
        value["bbox"] = list(self.bbox)
        value["persistence"] = f"{self.persistence_count}/{self.qualified_frames}"
        return value


@dataclass
class PhaseResult:
    angle: int
    verdict: Verdict
    reason: str
    defects: list[Defect] = field(default_factory=list)
    registrations: list[dict] = field(default_factory=list)
    captured_frames: int = 0
    qualified_frames: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class FitmentObservation:
    center_x: float
    center_y: float
    width: float
    height: float
    sharpness: float
    tracking_confidence: float = 1.0


@dataclass
class FitmentResult:
    verdict: Verdict
    reason: str
    statistics: dict[str, float]


def illumination_normalized(image: np.ndarray) -> np.ndarray:
    """Return bounded local-contrast gray, insensitive to broad illumination."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float32) / 255.0
    sigma = max(5.0, min(gray.shape) / 24.0)
    illumination = cv2.GaussianBlur(gray, (0, 0), sigma)
    normalized = gray / np.maximum(illumination, .08)
    return np.clip((normalized - .45) / 1.1, 0, 1).astype(np.float32)


def representations(image: np.ndarray) -> dict[str, np.ndarray]:
    gray = illumination_normalized(image)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    magnitude, orientation = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    magnitude = np.clip(magnitude / max(float(np.percentile(magnitude, 99.5)), 1e-4), 0, 1)
    edge_threshold = max(.08, float(np.percentile(magnitude, 72)))
    edges = (magnitude >= edge_threshold).astype(np.float32)
    return {"gray": gray, "gradient": magnitude, "orientation": orientation / 360., "edges": edges}


def glare_mask(image: np.ndarray) -> np.ndarray:
    bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 9)
    mean2 = cv2.GaussianBlur(gray.astype(np.float32) ** 2, (0, 0), 9)
    variance = np.maximum(mean2 - mean ** 2, 0)
    mask = (hsv[..., 2] >= 235) & (hsv[..., 1] <= 75) & (variance < 180)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(bool)


@dataclass
class GoldenPhaseModel:
    angle: int
    median_gray: np.ndarray
    scale_gray: np.ndarray
    median_gradient: np.ndarray
    scale_gradient: np.ndarray
    edge_probability: np.ndarray
    metadata: dict

    @classmethod
    def calibrate(cls, angle: int, images: list[np.ndarray], *, model_id: str,
                  roi: tuple[int, int, int, int], noise_floor: float = .035,
                  physical_part_ids: Iterable[str] | None = None) -> "GoldenPhaseModel":
        if angle not in ANGLES:
            raise ValueError(f"Unsupported inspection phase {angle}")
        ids = list(physical_part_ids or ())
        if len(images) < 3:
            raise ValueError("Golden calibration requires at least three qualified images")
        if ids and len(ids) != len(images):
            raise ValueError("physical_part_ids must identify every calibration image")
        reps = [representations(image) for image in images]
        shape = reps[0]["gray"].shape
        if any(row["gray"].shape != shape for row in reps):
            raise ValueError("All phase calibration images must have the same registered ROI shape")
        def robust(channel: str) -> tuple[np.ndarray, np.ndarray]:
            stack = np.stack([row[channel] for row in reps])
            median = np.median(stack, axis=0).astype(np.float32)
            mad = np.median(np.abs(stack - median), axis=0)
            return median, np.maximum(1.4826 * mad, noise_floor).astype(np.float32)
        gray, gray_scale = robust("gray")
        gradient, gradient_scale = robust("gradient")
        edge_probability = np.mean(np.stack([row["edges"] for row in reps]), axis=0).astype(np.float32)
        metadata = {
            "version": MODEL_VERSION, "model_id": model_id, "phase": angle,
            "camera_resolution": [int(images[0].shape[1]), int(images[0].shape[0])],
            "roi": list(roi), "preprocessing_version": PREPROCESSING_VERSION,
            "registration": "euclidean-gradient-ecc", "calibration_images": len(images),
            "physical_good_parts": len(set(ids)) if ids else None,
            "creation_timestamp": datetime.now(timezone.utc).isoformat(),
            "noise_floor": noise_floor,
            "software_model_version": MODEL_VERSION,
        }
        return cls(angle, gray, gray_scale, gradient, gradient_scale, edge_probability, metadata)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, median_gray=self.median_gray, scale_gray=self.scale_gray,
                            median_gradient=self.median_gradient, scale_gradient=self.scale_gradient,
                            edge_probability=self.edge_probability,
                            metadata=np.array(json.dumps(self.metadata)))
        return path

    @classmethod
    def load(cls, path: str | Path, *, expected_angle: int | None = None,
             expected_shape: tuple[int, int] | None = None) -> "GoldenPhaseModel":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            if metadata.get("version") != MODEL_VERSION or metadata.get("preprocessing_version") != PREPROCESSING_VERSION:
                raise ValueError("Golden model is incompatible; recalibration is required")
            angle = int(metadata["phase"])
            if expected_angle is not None and angle != expected_angle:
                raise ValueError(f"Wrong phase model: expected {expected_angle}, found {angle}")
            model = cls(angle, data["median_gray"], data["scale_gray"], data["median_gradient"],
                        data["scale_gradient"], data["edge_probability"], metadata)
        if expected_shape is not None and model.median_gray.shape != expected_shape:
            raise ValueError("Golden model ROI shape is incompatible; recalibration is required")
        return model


class GoldenBank:
    def __init__(self, root: str | Path, model_id: str) -> None:
        self.root, self.model_id = Path(root), model_id
        self._cache: dict[int, GoldenPhaseModel] = {}

    def path(self, angle: int) -> Path:
        return self.root / self.model_id / f"phase_{angle:03d}" / "golden_v1.npz"

    def get(self, angle: int, shape: tuple[int, int] | None = None) -> GoldenPhaseModel:
        if angle not in ANGLES:
            raise ValueError(f"PLC supplied invalid phase {angle}")
        if angle not in self._cache:
            path = self.path(angle)
            if not path.exists():
                raise FileNotFoundError(f"Phase {angle} golden model is missing: {path}")
            self._cache[angle] = GoldenPhaseModel.load(path, expected_angle=angle, expected_shape=shape)
        model = self._cache[angle]
        if shape is not None and model.median_gray.shape != shape:
            raise ValueError("Golden model ROI shape is incompatible; recalibration is required")
        return model


def register_to_golden(image: np.ndarray, golden: GoldenPhaseModel, config: PhaseLockedConfig) -> RegistrationResult:
    current = representations(image)["gradient"]
    reference = golden.median_gradient
    if current.shape != reference.shape:
        return RegistrationResult(False, 0, 0, 0, 0, image, "ROI_SHAPE_MISMATCH")
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        correlation, warp = cv2.findTransformECC(reference, current, warp, cv2.MOTION_EUCLIDEAN,
                                                  (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5),
                                                  None, 3)
    except cv2.error:
        return RegistrationResult(False, 0, 0, 0, 0, image, "ECC_FAILED")
    tx, ty = float(warp[0, 2]), float(warp[1, 2])
    rotation = float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0])))
    limit_x, limit_y = image.shape[1] * config.registration_max_translation_ratio, image.shape[0] * config.registration_max_translation_ratio
    valid = correlation >= config.registration_min_correlation and abs(tx) <= limit_x and abs(ty) <= limit_y and abs(rotation) <= config.registration_max_rotation_deg
    reason = "" if valid else "TRANSFORM_OUTSIDE_SAFE_LIMITS"
    if correlation < config.registration_min_correlation:
        reason = "LOW_CORRELATION"
    if not valid:
        return RegistrationResult(False, float(correlation), tx, ty, rotation, image, reason)
    aligned = cv2.warpAffine(image, warp, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    return RegistrationResult(True, float(correlation), tx, ty, rotation, aligned)


def analyze_fitment(observations: list[FitmentObservation], config: PhaseLockedConfig) -> FitmentResult:
    if len(observations) < 8:
        return FitmentResult(Verdict.INSPECTION_INVALID, "INSUFFICIENT_FITMENT_TRACKING", {"frames": float(len(observations))})
    values = np.array([[o.center_x, o.center_y, o.width, o.height, o.sharpness, o.tracking_confidence] for o in observations])
    center = np.median(values[:, :2], axis=0)
    radial = np.linalg.norm(values[:, :2] - center, axis=1)
    # A robust high quantile represents production runout without one corrupt tracking box.
    runout = float(np.quantile(radial, .99))
    x_p99 = float(np.quantile(np.abs(values[:, 0] - center[0]), .99))
    y_p99 = float(np.quantile(np.abs(values[:, 1] - center[1]), .99))
    width_mad = float(np.median(np.abs(values[:, 2] - np.median(values[:, 2]))))
    tracking_p05 = float(np.quantile(values[:, 5], .05))
    stats = {"center_x": float(center[0]), "center_y": float(center[1]), "runout_p99": runout,
             "horizontal_p99": x_p99, "vertical_p99": y_p99, "width_mad": width_mad,
             "sharpness_p05": float(np.quantile(values[:, 4], .05)), "tracking_p05": tracking_p05,
             "frames": float(len(observations))}
    if tracking_p05 < .45:
        return FitmentResult(Verdict.INSPECTION_INVALID, "UNSTABLE_TRACKING", stats)
    if runout > config.fitment_runout_threshold or y_p99 > config.fitment_axial_threshold:
        return FitmentResult(Verdict.FITMENT_FAIL, "EXCESSIVE_RUNOUT", stats)
    return FitmentResult(Verdict.PASS, "FITMENT_OK", stats)


class SupportingAnomalyModel(ABC):
    """Optional EfficientAD-compatible provider; its evidence is never authoritative."""
    @abstractmethod
    def infer(self, image: np.ndarray) -> tuple[float, np.ndarray]: ...


class DisabledAnomalyModel(SupportingAnomalyModel):
    def infer(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        return 0.0, np.zeros(image.shape[:2], np.float32)


def _boxes(mask: np.ndarray, minimum_area: int) -> list[tuple[int, int, int, int]]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [tuple(map(int, stats[i, :4])) for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= minimum_area]


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    return intersection / max(aw * ah + bw * bh - intersection, 1)


class PhaseLockedStructuralInspector:
    """Inspect stationary bursts against the PLC-selected phase only."""
    def __init__(self, bank: GoldenBank, config: PhaseLockedConfig | None = None,
                 supporting_model: SupportingAnomalyModel | None = None) -> None:
        self.bank, self.config = bank, config or PhaseLockedConfig()
        self.supporting_model = supporting_model or DisabledAnomalyModel()

    def _frame_candidates(self, angle: int, image: np.ndarray, golden: GoldenPhaseModel) -> tuple[list[Defect], RegistrationResult, dict]:
        registration = register_to_golden(image, golden, self.config)
        if not registration.valid:
            return [], registration, {}
        rep = representations(registration.image)
        z_gradient = np.abs(rep["gradient"] - golden.median_gradient) / golden.scale_gradient
        stable_expected = golden.edge_probability >= .75
        missing_edges = stable_expected & (rep["edges"] < .5)
        unexpected_edges = (golden.edge_probability <= .20) & (rep["edges"] > .5)
        structural = ((z_gradient >= self.config.golden_candidate_threshold) & (missing_edges | unexpected_edges))
        glare = glare_mask(registration.image) if self.config.glare_rejection_enabled else np.zeros(rep["gray"].shape, bool)
        # Glare suppresses intensity-only evidence, never a missing expected edge.
        structural &= (~glare | missing_edges)
        structural = cv2.morphologyEx(structural.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
        defects = []
        for bbox in _boxes(structural, self.config.minimum_region_area):
            x, y, w, h = bbox
            local_missing = float(np.mean(missing_edges[y:y+h, x:x+w]))
            local_z = float(np.percentile(z_gradient[y:y+h, x:x+w], 90))
            evidence = ["GOLDEN_STRUCTURAL_MISMATCH"]
            kind = "STRUCTURAL_DEFORMATION"
            geometry = min(1.5, local_missing * 3)
            if local_missing >= self.config.edge_candidate_threshold:
                evidence += ["EDGE_CONTINUITY_LOSS", "GEOMETRY_BREAK"]
                kind = "BROKEN_FIN"
            confidence = float(np.clip(.55 * local_z / self.config.golden_fail_threshold + .45 * geometry, 0, 1))
            defects.append(Defect(ANGLES.index(angle) + 1, angle, kind, bbox, confidence, evidence))
        ai_score, ai_map = self.supporting_model.infer(registration.image) if self.config.efficientad_enabled else (0., np.zeros(rep["gray"].shape, np.float32))
        return defects, registration, {"gradient_residual": z_gradient, "edge_mismatch": structural.astype(np.uint8) * 255,
                                        "glare_mask": glare.astype(np.uint8) * 255, "efficientad_map": ai_map,
                                        "efficientad_score": float(ai_score)}

    def inspect_phase(self, angle: int, frames: list[np.ndarray]) -> PhaseResult:
        started = time.perf_counter()
        if angle not in self.config.inspection_angles:
            return PhaseResult(angle, Verdict.VIEW_INVALID, "WRONG_PLC_PHASE", captured_frames=len(frames))
        if not frames:
            return PhaseResult(angle, Verdict.VIEW_INVALID, "NO_BURST_FRAMES")
        golden = self.bank.get(angle, frames[0].shape[:2])
        qualified = [f for f in frames if cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f, cv2.CV_64F).var() >= self.config.minimum_sharpness]
        if len(qualified) < self.config.minimum_qualified_frames:
            return PhaseResult(angle, Verdict.VIEW_INVALID, "INSUFFICIENT_QUALIFIED_FRAMES", captured_frames=len(frames), qualified_frames=len(qualified))
        candidates: list[list[Defect]] = []
        registrations, diagnostic = [], {}
        for frame in qualified:
            defects, registration, diagnostic = self._frame_candidates(angle, frame, golden)
            registrations.append({"correlation": registration.correlation, "translation_x": registration.translation_x,
                                  "translation_y": registration.translation_y, "rotation_deg": registration.rotation_deg,
                                  "valid": registration.valid, "reason": registration.reason})
            if not registration.valid:
                continue
            candidates.append(defects)
        if len(candidates) < self.config.minimum_qualified_frames:
            return PhaseResult(angle, Verdict.VIEW_INVALID, "REGISTRATION_FAILED", registrations=registrations,
                               captured_frames=len(frames), qualified_frames=len(qualified))
        tracks: list[list[Defect]] = []
        for frame_defects in candidates:
            used: set[int] = set()
            for track in tracks:
                match = next((i for i, defect in enumerate(frame_defects) if i not in used and _iou(track[-1].bbox, defect.bbox) >= self.config.consensus_iou), None)
                if match is not None:
                    track.append(frame_defects[match]); used.add(match)
            tracks.extend([[defect] for i, defect in enumerate(frame_defects) if i not in used])
        confirmed: list[Defect] = []
        for track in tracks:
            if len(track) >= self.config.temporal_confirmation_frames:
                best = max(track, key=lambda item: item.confidence)
                best.persistence_count, best.qualified_frames = len(track), len(candidates)
                best.evidence.append("STATIONARY_TEMPORAL_PERSISTENCE")
                confirmed.append(best)
        # AI-only and single-frame regions remain candidates and can never FAIL.
        unconfirmed = any(tracks)
        verdict = Verdict.FAIL if confirmed else (Verdict.CANDIDATE if unconfirmed else Verdict.PASS)
        reason = confirmed[0].defect if confirmed else ("UNCONFIRMED_STRUCTURAL_CANDIDATE" if unconfirmed else "STRUCTURE_NORMAL")
        return PhaseResult(angle, verdict, reason, confirmed, registrations, len(frames), len(qualified),
                           {"total": (time.perf_counter() - started) * 1000}, diagnostic)


def calibrate_fitment(results: list[FitmentResult]) -> dict:
    values = np.asarray([r.statistics["runout_p99"] for r in results if r.verdict == Verdict.PASS])
    if len(values) < 3:
        raise ValueError("Fitment calibration requires at least three known-good physical parts")
    median = float(np.median(values)); mad = float(np.median(np.abs(values - median)))
    return {"sample_count": len(values), "median": median, "mad": mad, "p95": float(np.quantile(values, .95)),
            "p99": float(np.quantile(values, .99)), "recommended_reject_threshold": float(max(np.quantile(values, .99), median + 6 * 1.4826 * mad))}


def calibration_report(channel_values: dict[str, list[float]], good_verdicts: list[Verdict]) -> dict:
    channels = {}
    for name, raw in channel_values.items():
        values = np.asarray(raw, dtype=float)
        median = float(np.median(values)); mad = float(np.median(np.abs(values - median)))
        channels[name] = {"median": median, "mad": mad, "p95": float(np.quantile(values, .95)),
                          "p99": float(np.quantile(values, .99)), "p995": float(np.quantile(values, .995))}
    rejects = sum(v in {Verdict.FAIL, Verdict.FITMENT_FAIL} for v in good_verdicts)
    return {"known_good_parts": len(good_verdicts), "good_false_reject_rate": rejects / max(len(good_verdicts), 1), "channels": channels}

