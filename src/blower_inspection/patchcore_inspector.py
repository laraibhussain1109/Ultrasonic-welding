"""Clean PatchCore-primary production backend.

PaDiM, template residuals and distillation remain available in
``anomaly_models`` for experiments; none participates in this decision path.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .anomaly_models import HybridPatchcorePadimInspector, HybridTrainingSettings, require_module
from .config import PartModelConfig
from .frame_quality import FrameQualityAnalyzer
from .geometry_inspector import FinGeometryInspector, glare_evidence, inspection_band_mask, support_rib_mask
from .inspection_fusion import fuse_patchcore_geometry
from .roi_stabilizer import CanonicalROI, ROIStabilizer
from .trainer import InspectionResult, inspection_overlay, list_images
from .yolo_tracking import YoloByteTrackDetector

PATCHCORE_MODEL_VERSION = 1


@dataclass(frozen=True)
class PatchCoreEvidence:
    raw_patch_distance_map: np.ndarray
    calibrated_score_map: np.ndarray
    image_score: float
    peak_score: float
    anomaly_mask: np.ndarray
    boxes: list[tuple[int, int, int, int]]
    section_scores: tuple[float, ...]
    valid: bool


def edge_authority_mask(shape: tuple[int, int], ratio: float, object_mask: np.ndarray | None = None) -> np.ndarray:
    """Continuous authority from zero at an object/ROI edge to one inside."""
    base = np.ones(shape, np.uint8) if object_mask is None else object_mask.astype(np.uint8).copy()
    # distanceTransform measures distance to an existing zero. An all-ones ROI
    # has no zero and OpenCV returns a large constant, which previously gave the
    # outer silhouette full authority. Seed the geometric border explicitly.
    base[[0, -1], :] = 0
    base[:, [0, -1]] = 0
    distance = cv2.distanceTransform(base, cv2.DIST_L2, 3)
    ramp = max(1.0, min(shape) * max(0.0, ratio))
    return np.clip(distance / ramp, 0, 1).astype(np.float32)


def filter_duplicate_images(images: list[np.ndarray], threshold: float = .99995,
                            pixel_mae_threshold: float = .002) -> list[int]:
    """Return indices after removing only genuinely redundant photographs.

    A repetitive blower legitimately produces descriptors with cosine similarity
    above .98 at different rotational phases. Cosine similarity alone therefore
    collapsed entire captures to one or two images. A frame is now redundant
    only when both its illumination-normalized descriptor *and* its actual
    thumbnail pixels are nearly identical.
    """
    kept, descriptors, thumbnails = [], [], []
    for index, image in enumerate(images):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        small = cv2.resize(gray, (64, 24), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        descriptor = (small - small.mean()).ravel()
        descriptor /= np.linalg.norm(descriptor) + 1e-6
        duplicate = any(
            float(descriptor @ old_descriptor) >= threshold
            and float(np.mean(np.abs(small - old_thumbnail))) <= pixel_mae_threshold
            for old_descriptor, old_thumbnail in zip(descriptors, thumbnails)
        )
        if not duplicate:
            kept.append(index)
            descriptors.append(descriptor)
            thumbnails.append(small)
    return kept


def evenly_limit_indices(count: int, limit: int) -> list[int]:
    """Deterministically retain coverage across a long rotating capture."""
    if count <= limit:
        return list(range(count))
    return sorted(set(int(round(value)) for value in np.linspace(0, count - 1, limit)))


class PatchCoreInspector(HybridPatchcorePadimInspector):
    def __init__(self, settings: HybridTrainingSettings | None = None, device: str | None = None) -> None:
        super().__init__(settings or HybridTrainingSettings(embedding_layers=("layer2", "layer3")), device)
        self._roi: ROIStabilizer | None = None

    @staticmethod
    def _model_path(config: PartModelConfig) -> Path:
        return config.patchcore_model_file or config.model_file

    def _configure(self, config: PartModelConfig) -> None:
        self.settings = replace(self.settings, image_size=config.image_size,
                                embedding_layers=config.patchcore_embedding_layers,
                                coreset_ratio=config.patchcore_coreset_ratio,
                                max_coreset_patches=config.patchcore_memory_bank_size,
                                # Calibration and production must query the exact
                                # same bank. The legacy 1,024-patch runtime cap
                                # raised live distances above calibrated limits.
                                runtime_memory_bank_limit=config.patchcore_memory_bank_size)
        self._roi = ROIStabilizer((config.image_size, max(128, config.image_size // 3)),
                                  mode=config.roi_mode, smoothing_frames=config.roi_smoothing_frames,
                                  padding_ratio=config.roi_padding_ratio)

    def _canonical(self, frame: np.ndarray, config: PartModelConfig,
                   detector: YoloByteTrackDetector | None = None) -> CanonicalROI:
        assert self._roi is not None
        bounds = detector.detect_best(frame).bounds if detector else (0, 0, frame.shape[1], frame.shape[0])
        return self._roi.canonicalize(frame, bounds)

    @staticmethod
    def _content_mask(roi: CanonicalROI) -> np.ndarray:
        mask = np.zeros(roi.image.shape[:2], dtype=bool)
        x, y, width, height = roi.content_bounds
        mask[y:y + height, x:x + width] = True
        return mask

    def _write_training_report(self, config: PartModelConfig, report: dict) -> Path:
        model_path = self._model_path(config)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        path = model_path.with_suffix(".training_report.json")
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return path

    def train(self, config: PartModelConfig, progress_callback=None) -> Path:
        self._configure(config)
        paths = list_images(config.normal_image_dir)
        if len(paths) < 20:
            raise ValueError("PatchCore training requires at least 20 source images")
        if config.yolo_model_path is None:
            raise ValueError("PatchCore training requires yolo_model_path for full-frame localization")
        detector = YoloByteTrackDetector(config.yolo_model_path, config.yolo_confidence)
        quality = FrameQualityAnalyzer(config.training_min_sharpness, config.max_glare_ratio,
                                       config.max_saturation_ratio)
        accepted, accepted_paths, rejection_counts = [], [], {}
        rejected_examples: dict[str, list[str]] = {}
        started = time.monotonic()
        for i, path in enumerate(paths, 1):
            frame = cv2.imread(str(path))
            try:
                canonical = self._canonical(frame, config, detector) if frame is not None else None
                result = quality.analyze(canonical.image, self._content_mask(canonical)) if canonical is not None else None
            except ValueError:
                result = None
            if result is None or not result.valid:
                reasons = result.reasons if result else ("BAD_CROP",)
                for reason in reasons:
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                    examples = rejected_examples.setdefault(reason, [])
                    if len(examples) < 10:
                        examples.append(str(path))
            else:
                accepted.append(canonical.image); accepted_paths.append(path)
            if progress_callback:
                from .training_progress import TrainingProgress
                progress_callback(TrainingProgress("Quality-gating YOLO crops", i, len(paths), time.monotonic() - started))
        qualified_count = len(accepted)
        keep = filter_duplicate_images(accepted)
        accepted = [accepted[i] for i in keep]; accepted_paths = [accepted_paths[i] for i in keep]
        nonduplicate_count = len(accepted)
        limited = evenly_limit_indices(nonduplicate_count, self.settings.max_training_images)
        accepted = [accepted[i] for i in limited]; accepted_paths = [accepted_paths[i] for i in limited]
        if len(accepted) < 10:
            report = {"status": "FAILED", "total_source_images": len(paths),
                      "quality_accepted_images": qualified_count,
                      "nonduplicate_images": nonduplicate_count,
                      "diverse_accepted_images": len(accepted),
                      "duplicates_removed": qualified_count - nonduplicate_count,
                      "training_limit_removed": nonduplicate_count - len(accepted),
                      "rejected": rejection_counts, "rejected_examples": rejected_examples,
                      "quality_settings": {"minimum_sharpness": config.training_min_sharpness,
                                           "max_glare_ratio": config.max_glare_ratio,
                                           "max_saturation_ratio": config.max_saturation_ratio}}
            report_path = self._write_training_report(config, report)
            reason_summary = ", ".join(f"{name}={count}" for name, count in sorted(rejection_counts.items())) or "none"
            raise ValueError(
                f"Only {len(accepted)} diverse, qualified images remain; at least 10 are required. "
                f"Quality accepted {qualified_count}/{len(paths)}; duplicate filter removed "
                f"{qualified_count - nonduplicate_count}; training limit removed "
                f"{nonduplicate_count - len(accepted)}. Rejections: {reason_summary}. "
                f"Details: {report_path}"
            )
        # Calibration is disjoint and never enters the memory bank.
        split = max(3, len(accepted) // 5)
        calibration_images, calibration_paths = accepted[-split:], accepted_paths[-split:]
        training_images, training_paths = accepted[:-split], accepted_paths[:-split]
        tensors = [self._preprocess_image(image) for image in training_images]
        torch = require_module("torch")
        embeddings = []
        for tensor in tensors:
            fmap, grid = self._extract_embeddings_from_tensor(tensor)
            embeddings.append(fmap.flatten(2).permute(0, 2, 1).reshape(-1, fmap.shape[1]).cpu())
        all_embeddings = torch.cat(embeddings)
        memory, candidates = self._build_patchcore_memory(all_embeddings, torch)
        bank_hash = hashlib.sha256(memory.numpy().tobytes()).hexdigest()
        model_path = self._model_path(config)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {"version": PATCHCORE_MODEL_VERSION, "algorithm": "patchcore_primary",
                      "settings": asdict(self.settings), "memory_bank": memory,
                      "memory_bank_hash": bank_hash, "memory_candidate_count": candidates,
                      "training_manifest_hash": self._manifest_hash(training_paths)}
        torch.save(checkpoint, model_path)
        raw = [self._raw_map(image, checkpoint, torch) for image in calibration_images]
        calibration = self._calibration(raw, config, bank_hash, checkpoint["training_manifest_hash"], calibration_paths)
        calibration_path = config.patchcore_calibration_file or model_path.with_suffix(".calibration.json")
        calibration_path.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
        report = {"total_source_images": len(paths), "accepted_images": len(accepted),
                  "training_images": len(training_images), "calibration_images": len(calibration_images),
                  "nonduplicate_images": nonduplicate_count,
                  "duplicates_removed": qualified_count - nonduplicate_count,
                  "training_limit_removed": nonduplicate_count - len(accepted),
                  "rejected": rejection_counts,
                  "rejected_examples": rejected_examples}
        self._write_training_report(config, report)
        self._checkpoint_cache.clear()
        return model_path

    @staticmethod
    def _manifest_hash(paths: list[Path]) -> str:
        return hashlib.sha256("\n".join(str(p.resolve()) for p in paths).encode()).hexdigest()

    def _raw_map(self, image: np.ndarray, checkpoint: dict, torch) -> np.ndarray:
        features, _ = self._extract_embeddings_from_tensor(self._preprocess_image(image))
        small = self._patchcore_score_map(features, checkpoint["memory_bank"], torch)
        return cv2.resize(small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

    def _calibration(self, maps, config, bank_hash, manifest_hash, paths) -> dict:
        image_scores = [float(np.quantile(item, .995)) for item in maps]
        patch_scores = np.concatenate([item.ravel() for item in maps])
        median, mad = float(np.median(image_scores)), float(np.median(np.abs(image_scores - np.median(image_scores))))
        candidate = max(float(np.quantile(image_scores, .99)), median + 6 * 1.4826 * mad)
        fail = max(float(np.quantile(image_scores, .999)), median + 8 * 1.4826 * mad, candidate * 1.05)
        return {"version": 1, "model_version": PATCHCORE_MODEL_VERSION, "backbone": self.settings.backbone,
                "embedding_layers": list(self.settings.embedding_layers), "memory_bank_hash": bank_hash,
                "training_manifest_hash": manifest_hash, "calibration_manifest_hash": self._manifest_hash(paths),
                "calibration_date": datetime.now(timezone.utc).isoformat(),
                "thresholds": {"candidate": candidate, "fail": fail,
                               "patch_p999": float(np.quantile(patch_scores, .999))},
                "quality_settings": {"minimum_sharpness": config.training_min_sharpness,
                                     "max_glare_ratio": config.max_glare_ratio,
                                     "max_saturation_ratio": config.max_saturation_ratio}}

    def inspect(self, config: PartModelConfig, image: np.ndarray, *, save_outputs: bool = True,
                crop_to_component: bool = True) -> InspectionResult:
        started = time.perf_counter(); self._configure(config)
        detector = YoloByteTrackDetector(config.yolo_model_path, config.yolo_confidence) if crop_to_component and config.yolo_model_path else None
        canonical = self._canonical(image, config, detector)
        roi = canonical.image
        quality = FrameQualityAnalyzer(config.inference_min_sharpness, config.max_glare_ratio,
                                       config.max_saturation_ratio).analyze(roi, self._content_mask(canonical))
        quality_ms = (time.perf_counter() - started) * 1000
        if not quality.valid:
            return InspectionResult("VIEW INVALID", 0, 0, 0, [], display_image=roi,
                                    reason_codes=quality.reasons, view_valid=False,
                                    view_quality_score=quality.sharpness,
                                    latencies_ms={"frame_quality": quality_ms})
        torch = require_module("torch")
        model_path = self._model_path(config)
        checkpoint = self._load_runtime_checkpoint(torch, model_path)
        if checkpoint.get("algorithm") != "patchcore_primary":
            raise ValueError("Model is not a PatchCore-primary checkpoint; retrain it")
        calibration_path = config.patchcore_calibration_file or model_path.with_suffix(".calibration.json")
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration.get("memory_bank_hash") != checkpoint.get("memory_bank_hash") or calibration.get("embedding_layers") != list(self.settings.embedding_layers):
            raise RuntimeError("Stale PatchCore calibration: model/settings do not match")
        patch_started = time.perf_counter(); raw = self._raw_map(roi, checkpoint, torch)
        band = inspection_band_mask(raw.shape, config.inspection_band_top_ratio, config.inspection_band_bottom_ratio)
        ribs, _ = support_rib_mask(roi)
        if config.support_rib_mask_enabled:
            margin = max(1, round(raw.shape[1] * config.support_rib_margin_ratio))
            ribs = cv2.dilate(ribs.astype(np.uint8), np.ones((1, margin * 2 + 1), np.uint8)).astype(bool)
        valid = band & ~ribs
        authority = edge_authority_mask(raw.shape, config.patchcore_edge_ignore_ratio)
        glare = glare_evidence(roi, raw.shape, top=config.inspection_band_top_ratio,
                               bottom=config.inspection_band_bottom_ratio)
        weighted = raw * authority * valid * (1.0 - .75 * glare.mask.astype(np.float32))
        thresholds = calibration["thresholds"]
        candidate, fail = config.patchcore_candidate_threshold or thresholds["candidate"], config.patchcore_fail_threshold or thresholds["fail"]
        mask = (weighted >= thresholds["patch_p999"]) & valid
        image_score = float(np.quantile(weighted[valid], .995)) if np.any(valid) else 0.0
        section_scores = tuple(float(np.quantile(part[part > 0], .995)) if np.any(part > 0) else 0.0
                               for part in np.array_split(weighted, config.patchcore_section_count, axis=1))
        candidate_sections = tuple(index for index, score in enumerate(section_scores) if score >= candidate)
        geometry_started = time.perf_counter()
        geometry = FinGeometryInspector(band_top=config.inspection_band_top_ratio,
                                        band_bottom=config.inspection_band_bottom_ratio).inspect(roi)
        decision = fuse_patchcore_geometry(image_score, mask, geometry, glare_score=glare.score,
                                           candidate_threshold=candidate, fail_threshold=fail,
                                           geometry_candidate_threshold=config.geometry_candidate_threshold,
                                           geometry_fail_threshold=config.geometry_fail_threshold,
                                           glare_threshold=config.glare_threshold)
        display, boxes = inspection_overlay(roi, weighted, decision.confirmed_mask, status=decision.status,
                                             anomaly_score=image_score, min_box_area_px=config.min_defect_area_px,
                                             score_normalizer=max(fail, 1e-6))
        total_ms = (time.perf_counter() - started) * 1000
        return InspectionResult(decision.status, image_score, int(decision.confirmed_mask.sum()), 0, [],
                                display_image=display, defect_boxes=boxes, geometry_score=geometry.score,
                                periodicity_score=geometry.periodicity_score, glare_score=glare.score,
                                hybrid_score=max(image_score / max(fail, 1e-6), geometry.score),
                                reason_codes=decision.reason_codes, view_valid=True,
                                candidate_sections=candidate_sections,
                                view_quality_score=quality.sharpness,
                                latencies_ms={"frame_quality": quality_ms,
                                              "patchcore": (geometry_started - patch_started) * 1000,
                                              "geometry": (time.perf_counter() - geometry_started) * 1000,
                                              "total": total_ms},
                                geometry_components={"orientation": geometry.orientation_score,
                                                     "continuity": geometry.continuity_score,
                                                     "pitch": geometry.pitch_score,
                                                     "broken": geometry.broken_fin_score,
                                                     "missing": geometry.missing_fin_score,
                                                     "tilted": geometry.tilted_fin_score})
