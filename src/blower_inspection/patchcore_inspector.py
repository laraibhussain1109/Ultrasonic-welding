"""Clean PatchCore-primary production backend.

PaDiM, template residuals and distillation remain available in
``anomaly_models`` for experiments; none participates in this decision path.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .anomaly_models import HybridPatchcorePadimInspector, HybridTrainingSettings, require_module
from .config import PartModelConfig
from .frame_quality import FrameQualityAnalyzer
from .geometry_inspector import GeometryEvidence, FinGeometryInspector, glare_evidence, inspection_band_mask, support_rib_mask
from .heatmap_area import evaluate_heatmap_area
from .roi_stabilizer import CanonicalROI, ROIStabilizer
from .trainer import InspectionResult, inspection_overlay, list_images
from .training_cache import DiskPatchEmbeddings
from .training_progress import TrainingProgress
from .yolo_tracking import YoloByteTrackDetector

PATCHCORE_MODEL_VERSION = 3


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


class DuplicateImageFilter:
    """Remove redundant crops while retaining only small comparison thumbnails.

    A repetitive blower legitimately produces descriptors with cosine similarity
    above .98 at different rotational phases. Cosine similarity alone therefore
    collapsed entire captures to one or two images. A frame is now redundant
    only when both its illumination-normalized descriptor *and* its actual
    thumbnail pixels are nearly identical.
    """
    def __init__(self, threshold: float = .99995, pixel_mae_threshold: float = .002) -> None:
        self.threshold = threshold
        self.pixel_mae_threshold = pixel_mae_threshold
        self.descriptors, self.thumbnails = [], []

    def accept(self, image: np.ndarray) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        small = cv2.resize(gray, (64, 24), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        descriptor = (small - small.mean()).ravel()
        descriptor /= np.linalg.norm(descriptor) + 1e-6
        duplicate = any(
            float(descriptor @ old_descriptor) >= self.threshold
            and float(np.mean(np.abs(small - old_thumbnail))) <= self.pixel_mae_threshold
            for old_descriptor, old_thumbnail in zip(self.descriptors, self.thumbnails)
        )
        if not duplicate:
            self.descriptors.append(descriptor)
            self.thumbnails.append(small)
        return not duplicate


def filter_duplicate_images(images: list[np.ndarray], threshold: float = .99995,
                            pixel_mae_threshold: float = .002) -> list[int]:
    duplicate_filter = DuplicateImageFilter(threshold, pixel_mae_threshold)
    return [index for index, image in enumerate(images) if duplicate_filter.accept(image)]


def evenly_limit_indices(count: int, limit: int) -> list[int]:
    """Deterministically retain coverage across a long rotating capture."""
    if count <= limit:
        return list(range(count))
    return sorted(set(int(round(value)) for value in np.linspace(0, count - 1, limit)))


class PatchCoreInspector(HybridPatchcorePadimInspector):
    def __init__(self, settings: HybridTrainingSettings | None = None, device: str | None = None) -> None:
        super().__init__(settings or HybridTrainingSettings(embedding_layers=("layer2", "layer3")), device)
        self._roi: ROIStabilizer | None = None
        self._feature_preprocessing: dict | None = None
        self._runtime_checkpoint = None
        self._training_roi_size: tuple[int, int] | None = None
        self._fixed_detector = None

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
        size = self._training_roi_size or (config.image_size, max(128, config.image_size // 3))
        self._roi = ROIStabilizer(size,
                                  mode=config.roi_mode, smoothing_frames=config.roi_smoothing_frames,
                                  padding_ratio=config.roi_padding_ratio)

    def _restore_runtime(self, checkpoint: dict, config: PartModelConfig, torch) -> None:
        """Restore the trained features before preparing or scoring any ROI."""
        if checkpoint is not self._runtime_checkpoint:
            previous_backbone = self._backbone_cache
            self._apply_checkpoint_settings(checkpoint)
            self._feature_preprocessing = checkpoint.get("preprocessing")
            self._fixed_detector = None
            if self._feature_preprocessing and self._feature_preprocessing.get("mode") == "fixed_spatial":
                from .patchcore_spatial import file_digest
                from .fixed_settings import InspectionSettings
                from .stationary_roi import YoloROI
                source_settings = InspectionSettings.from_dict(checkpoint["roi_settings"])
                if config.yolo_model_path is None or file_digest(config.yolo_model_path) != self._feature_preprocessing["yolo_sha256"]:
                    raise ValueError("Imported PatchCore YOLO weights changed; restore training weights or retrain")
                source_settings = replace(source_settings, yolo=replace(source_settings.yolo, model_path=str(config.yolo_model_path)))
                self._fixed_detector = YoloROI(source_settings)
            state = checkpoint.get("backbone_state")
            if state is not None:
                from .anomaly_models import _FeatureHook
                models = require_module("torchvision.models")
                backbone = getattr(models, self.settings.backbone)(weights=None)
                backbone.load_state_dict(state)
                backbone.eval().to(self._device(torch))
                if previous_backbone is not None:
                    previous_backbone[1].close()
                hook = _FeatureHook(self.settings.embedding_layers)
                hook.attach(backbone)
                self._backbone_cache = (backbone, hook, torch)
            self._runtime_checkpoint = checkpoint
        preprocessing = self._feature_preprocessing
        if preprocessing:
            if (preprocessing["input_width"] != config.image_size
                    or tuple(self.settings.embedding_layers) != config.patchcore_embedding_layers
                    or preprocessing.get("roi_padding_ratio", config.roi_padding_ratio) != config.roi_padding_ratio):
                raise ValueError("PatchCore ROI width/padding/embedding layers changed; restore training settings or retrain")
            self._roi = ROIStabilizer((preprocessing["input_width"], preprocessing["input_height"]),
                                      mode=config.roi_mode, smoothing_frames=config.roi_smoothing_frames,
                                      padding_ratio=config.roi_padding_ratio)

    def _load_runtime_checkpoint(self, torch, model_file):
        stat = model_file.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        cached = self._checkpoint_cache.get(model_file)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        checkpoint = self._load_hybrid_checkpoint(torch, model_file, map_location="cpu")
        if checkpoint.get("algorithm") == "patchcore_fixed_spatial":
            raise ValueError("Spatial checkpoint needs production calibration. Run 'blower-inspection import-fixed' before starting the desktop.")
        bank = checkpoint.get("memory_bank")
        if not isinstance(bank, torch.Tensor) or bank.ndim != 2 or not len(bank) or not torch.isfinite(bank).all():
            raise ValueError("Invalid PatchCore memory bank")
        digest = hashlib.sha256(bank.numpy().tobytes()).hexdigest()
        if digest != checkpoint.get("memory_bank_hash"):
            raise ValueError("PatchCore memory-bank digest mismatch")
        # Calibration queries the complete bank. A changed runtime limit must
        # never silently subsample that bank and change the distance scale.
        checkpoint["memory_bank"] = bank.to(self._device(torch)).float().contiguous()
        self._checkpoint_cache[model_file] = (stamp, checkpoint)
        return checkpoint

    def inspection_bounds(self, frame, config, bounds):
        """Lock the same padded full-frame crop that trained the checkpoint."""
        assert self._roi is not None
        return self._roi.stabilize(bounds, frame.shape)

    def detect_inspection_roi(self, frame, config, detector):
        from .yolo_tracking import TrackedPart
        if self._fixed_detector is not None:
            roi = self._fixed_detector.prepare(frame, retain_original=False)
            return TrackedPart(1, roi.bounds, roi.yolo_confidence)
        detected = detector.detect_best(frame)
        return replace(detected, bounds=self.inspection_bounds(frame, config, detected.bounds))

    def _preprocess_image(self, image):
        if not self._feature_preprocessing or self._feature_preprocessing.get("mode") != "fixed_spatial":
            return super()._preprocess_image(image)
        p = self._feature_preprocessing
        if image.shape[:2] != (p["input_height"], p["input_width"]):
            raise ValueError("Imported PatchCore requires its trained canonical ROI dimensions")
        torch = require_module("torch")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0)

    def _extract_embeddings_with_backbone(self, tensor, backbone, hook, torch):
        if not self._feature_preprocessing or self._feature_preprocessing.get("mode") != "fixed_spatial":
            return super()._extract_embeddings_with_backbone(tensor, backbone, hook, torch)
        # Preserve the rectangular, float32 feature grid of the imported bank.
        with torch.inference_mode():
            hook.clear()
            backbone(tensor.to(self._device(torch)))
            maps = [hook.features[layer].float() for layer in self.settings.embedding_layers]
            target = maps[0].shape[-2:]
            maps = [torch.nn.functional.interpolate(m, size=target, mode="bilinear", align_corners=False)
                    if m.shape[-2:] != target else m for m in maps]
            p = self._feature_preprocessing
            grid = (max(1, p["input_height"] // 8), max(1, p["input_width"] // 8))
            embedding = torch.nn.functional.adaptive_avg_pool2d(torch.cat(maps, dim=1), grid)
            embedding = self._project_embedding(embedding, torch)
            return torch.nn.functional.normalize(embedding, p=2, dim=1).contiguous(), grid

    def validate_ready(self, config: PartModelConfig) -> None:
        """Report missing production assets before the motor starts capturing."""
        model_path = self._model_path(config)
        calibration_path = config.patchcore_calibration_file or model_path.with_suffix(".calibration.json")
        for label, path in (("PatchCore checkpoint", model_path), ("PatchCore calibration", calibration_path)):
            if not path.is_file():
                raise FileNotFoundError(f"{label} not found: {path.resolve()}. Select/train this model before starting inspection.")
        self._configure(config)
        torch = require_module("torch")
        checkpoint = self._load_runtime_checkpoint(torch, model_path)
        if checkpoint.get("algorithm") != "patchcore_primary" or checkpoint.get("version") != PATCHCORE_MODEL_VERSION:
            raise ValueError("PatchCore checkpoint requires calibrated fin geometry; retrain the selected model")
        if len(checkpoint.get("geometry_calibrations", [])) != config.patchcore_section_count:
            raise ValueError("PatchCore geometry section calibration does not match the selected model")
        self._restore_runtime(checkpoint, config, torch)
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration.get("memory_bank_hash") != checkpoint.get("memory_bank_hash") or calibration.get("embedding_layers") != list(self.settings.embedding_layers):
            raise RuntimeError("Stale PatchCore calibration: model/settings do not match")
        for name in ("patch_p999", "candidate", "fail"):
            value = calibration.get("thresholds", {}).get(name)
            if value is None or not np.isfinite(value) or value <= 0:
                raise ValueError(f"PatchCore calibration has no valid {name} threshold")

    def _canonical(self, frame: np.ndarray, config: PartModelConfig,
                   detector: YoloByteTrackDetector | None = None) -> CanonicalROI:
        assert self._roi is not None
        if detector is not None and self._fixed_detector is not None:
            prepared = self._fixed_detector.prepare(frame, retain_original=False)
            x, y, width, height = cv2.boundingRect(prepared.content_mask.astype(np.uint8))
            return CanonicalROI(prepared.image, prepared.bounds, (x, y, width, height))
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

    @staticmethod
    def _section_bounds(width: int, count: int) -> list[tuple[int, int]]:
        edges = np.linspace(0, width, max(1, count) + 1).round().astype(int)
        return [(int(edges[index]), int(edges[index + 1])) for index in range(len(edges) - 1)]

    def _calibrate_geometry(self, images: list[np.ndarray], config: PartModelConfig) -> list[dict]:
        """Calibrate each longitudinal section so one bent fin is not averaged away."""
        calibrations = []
        for x0, x1 in self._section_bounds(images[0].shape[1], config.patchcore_section_count):
            sections = [image[:, x0:x1] for image in images if x1 > x0]
            calibrations.append(FinGeometryInspector.calibrate(
                sections, band_top=config.inspection_band_top_ratio,
                band_bottom=config.inspection_band_bottom_ratio,
            ))
        return calibrations

    def _inspect_geometry(self, image: np.ndarray, config: PartModelConfig,
                          calibrations: list[dict]) -> GeometryEvidence:
        bounds = self._section_bounds(image.shape[1], config.patchcore_section_count)
        if len(calibrations) != len(bounds):
            raise RuntimeError("Stale PatchCore model: geometry section calibration count changed; retrain")
        evidence = []
        full_mask = np.zeros(image.shape[:2], dtype=bool)
        regions: list[tuple[int, int, int, int]] = []
        for (x0, x1), calibration in zip(bounds, calibrations):
            current = FinGeometryInspector(
                calibration, band_top=config.inspection_band_top_ratio,
                band_bottom=config.inspection_band_bottom_ratio,
                candidate_threshold=config.geometry_candidate_threshold,
            ).inspect(image[:, x0:x1])
            evidence.append(current)
            full_mask[:, x0:x1] |= current.defect_mask
            regions.extend((x + x0, y, width, height) for x, y, width, height in current.candidate_regions)
            if current.score >= config.geometry_candidate_threshold and not np.any(current.defect_mask):
                y0 = int(round(image.shape[0] * config.inspection_band_top_ratio))
                y1 = int(round(image.shape[0] * config.inspection_band_bottom_ratio))
                full_mask[y0:y1, x0:x1] = True
                regions.append((x0, y0, x1 - x0, max(1, y1 - y0)))
        return GeometryEvidence(
            score=max(item.score for item in evidence),
            orientation_score=max(item.orientation_score for item in evidence),
            pitch_score=max(item.pitch_score for item in evidence),
            continuity_score=max(item.continuity_score for item in evidence),
            broken_fin_score=max(item.broken_fin_score for item in evidence),
            periodicity_score=max(item.periodicity_score for item in evidence),
            support_rib_confidence=max(item.support_rib_confidence for item in evidence),
            defect_mask=full_mask, candidate_regions=regions,
            valid=all(item.valid for item in evidence),
            missing_fin_score=max(item.missing_fin_score for item in evidence),
            tilted_fin_score=max(item.tilted_fin_score for item in evidence),
        )

    def train(self, config: PartModelConfig, progress_callback=None, *, roi_size=None) -> Path:
        self._feature_preprocessing = None
        self._fixed_detector = None
        self._runtime_checkpoint = None
        self._training_roi_size = roi_size
        self._configure(config)
        paths = list_images(config.normal_image_dir)
        if len(paths) < 20:
            raise ValueError("PatchCore training requires at least 20 source images")
        if config.yolo_model_path is None:
            raise ValueError("PatchCore training requires yolo_model_path for full-frame localization")
        detector = YoloByteTrackDetector(config.yolo_model_path, config.yolo_confidence)
        quality = FrameQualityAnalyzer(config.training_min_sharpness, config.max_glare_ratio,
                                       config.max_saturation_ratio)
        with tempfile.TemporaryDirectory(prefix="patchcore-training-") as directory:
            return self._train_cached(config, paths, detector, quality, Path(directory), progress_callback)

    def _train_cached(self, config, paths, detector, quality, cache_dir, progress_callback):
        accepted, accepted_paths, rejection_counts = [], [], {}
        duplicate_filter = DuplicateImageFilter()
        qualified_count = 0
        rejected_examples: dict[str, list[str]] = {}
        started = time.monotonic()
        def progress(stage, completed, total):
            if progress_callback:
                progress_callback(TrainingProgress(stage, completed, total, time.monotonic() - started))
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
                qualified_count += 1
                if duplicate_filter.accept(canonical.image):
                    cached = cache_dir / f"roi_{i}.npy"
                    np.save(cached, canonical.image, allow_pickle=False)
                    accepted.append(cached)
                    accepted_paths.append(path)
            progress("Quality-gating YOLO crops", i, len(paths))
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
        # Only the configured, evenly spaced subset is loaded for geometry and
        # calibration, rather than every accepted crop in the source directory.
        accepted = [np.load(path, allow_pickle=False) for path in accepted]
        del frame, canonical, duplicate_filter
        # Calibration is disjoint and never enters the memory bank.
        split = max(3, len(accepted) // 5)
        calibration_images, calibration_paths = accepted[-split:], accepted_paths[-split:]
        training_images, training_paths = accepted[:-split], accepted_paths[:-split]
        torch = require_module("torch")
        with DiskPatchEmbeddings(cache_dir / "patches.bin") as embeddings:
            for index, image in enumerate(training_images, 1):
                fmap, grid = self._extract_embeddings_from_tensor(self._preprocess_image(image))
                embeddings.append(fmap.flatten(2).permute(0, 2, 1).reshape(-1, fmap.shape[1]))
                progress("PatchCore features", index, len(training_images))
            progress("PatchCore coreset", 0, 1)
            memory, candidates = embeddings.build_memory(self, torch)
            progress("PatchCore coreset", 1, 1)
        bank_hash = hashlib.sha256(memory.numpy().tobytes()).hexdigest()
        progress("Fin geometry calibration", 0, 1)
        geometry_calibrations = self._calibrate_geometry(training_images, config)
        progress("Fin geometry calibration", 1, 1)
        model_path = self._model_path(config)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {"version": PATCHCORE_MODEL_VERSION, "algorithm": "patchcore_primary",
                      "settings": asdict(self.settings), "memory_bank": memory,
                      "memory_bank_hash": bank_hash, "memory_candidate_count": candidates,
                      "training_manifest_hash": self._manifest_hash(training_paths),
                      "geometry_calibrations": geometry_calibrations,
                      "geometry_section_count": config.patchcore_section_count,
                      "backbone_state": {k: v.detach().cpu() for k, v in self._build_backbone()[0].state_dict().items()},
                      "preprocessing": {"mode": "primary_square", "input_width": self._roi.output_size[0],
                                        "input_height": self._roi.output_size[1], "roi_padding_ratio": config.roi_padding_ratio},
                      "training_manifest": [str(p) for p in training_paths],
                      "calibration_manifest": [str(p) for p in calibration_paths]}
        raw = []
        for index, image in enumerate(calibration_images, 1):
            raw.append(self._raw_map(image, checkpoint, torch))
            progress("Production distance calibration", index, len(calibration_images))
        calibration = self._calibration(raw, config, bank_hash, checkpoint["training_manifest_hash"], calibration_paths)
        report = {"total_source_images": len(paths), "accepted_images": len(accepted),
                  "training_images": len(training_images), "calibration_images": len(calibration_images),
                  "nonduplicate_images": nonduplicate_count,
                  "duplicates_removed": qualified_count - nonduplicate_count,
                  "training_limit_removed": nonduplicate_count - len(accepted),
                  "rejected": rejection_counts,
                  "rejected_examples": rejected_examples}
        self._publish_training(config, checkpoint, calibration, report, torch)
        progress("Production model and calibration saved", 1, 1)
        return model_path

    def _publish_training(self, config, checkpoint, calibration, report, torch):
        model_path = self._model_path(config)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        calibration_path = config.patchcore_calibration_file or model_path.with_suffix(".calibration.json")
        report_path = model_path.with_suffix(".training_report.json")
        # Finish all computations and serialize the complete bundle before
        # replacing any previous production asset. Calibration errors cannot
        # leave an old JSON paired with a newly overwritten checkpoint.
        targets = (model_path, calibration_path, report_path)
        if len({path.resolve() for path in targets}) != 3:
            raise ValueError("PatchCore model, calibration, and report paths must differ")
        with ExitStack() as stack:
            sources = []
            for target in targets:
                target.parent.mkdir(parents=True, exist_ok=True)
                directory = stack.enter_context(tempfile.TemporaryDirectory(prefix="patchcore-publish-", dir=target.parent))
                sources.append(Path(directory) / target.name)
            torch.save(checkpoint, sources[0])
            sources[1].write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
            sources[2].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            for source, target in zip(sources, targets):
                source.replace(target)
        self._checkpoint_cache.clear()
        self._runtime_checkpoint = None

    @staticmethod
    def _manifest_hash(paths: list[Path]) -> str:
        return hashlib.sha256("\n".join(str(p.resolve()) for p in paths).encode()).hexdigest()

    def _raw_map(self, image: np.ndarray, checkpoint: dict, torch) -> np.ndarray:
        features, _ = self._extract_embeddings_from_tensor(self._preprocess_image(image))
        flat = features.flatten(2).permute(0, 2, 1).reshape(-1, features.shape[1])
        bank = checkpoint["memory_bank"].to(flat.device)
        nearest = []
        with torch.inference_mode():
            for queries in flat.split(256):
                distances = torch.full((len(queries),), float("inf"), device=queries.device)
                for references in bank.split(2048):
                    distances = torch.minimum(distances, torch.cdist(queries, references).min(dim=1).values)
                nearest.append(distances)
        small = torch.cat(nearest).reshape(features.shape[-2:]).cpu().numpy().astype(np.float32)
        interpolation = cv2.INTER_LINEAR if self._feature_preprocessing and self._feature_preprocessing.get("mode") == "fixed_spatial" else cv2.INTER_CUBIC
        return cv2.resize(small, (image.shape[1], image.shape[0]), interpolation=interpolation)

    def _calibration(self, maps, config, bank_hash, manifest_hash, paths) -> dict:
        if any(not np.isfinite(item).all() or not item.size for item in maps):
            raise ValueError("PatchCore produced empty/nonfinite GOOD calibration distances")
        image_scores = [float(np.quantile(item, .995)) for item in maps]
        patch_scores = np.concatenate([item.ravel() for item in maps])
        median, mad = float(np.median(image_scores)), float(np.median(np.abs(image_scores - np.median(image_scores))))
        floor = float(np.finfo(np.float32).eps)
        candidate = max(float(np.quantile(image_scores, .99)), median + 6 * 1.4826 * mad, floor)
        fail = max(float(np.quantile(image_scores, .999)), median + 8 * 1.4826 * mad, candidate * 1.05)
        return {"version": 1, "model_version": PATCHCORE_MODEL_VERSION, "backbone": self.settings.backbone,
                "embedding_layers": list(self.settings.embedding_layers), "memory_bank_hash": bank_hash,
                "training_manifest_hash": manifest_hash, "calibration_manifest_hash": self._manifest_hash(paths),
                "calibration_date": datetime.now(timezone.utc).isoformat(),
                "thresholds": {"candidate": candidate, "fail": fail,
                               "patch_p999": max(float(np.quantile(patch_scores, .999)), floor)},
                "quality_settings": {"minimum_sharpness": config.training_min_sharpness,
                                     "max_glare_ratio": config.max_glare_ratio,
                                     "max_saturation_ratio": config.max_saturation_ratio}}

    def inspect(self, config: PartModelConfig, image: np.ndarray, *, save_outputs: bool = True,
                crop_to_component: bool = True) -> InspectionResult:
        started = time.perf_counter(); self._configure(config)
        torch = require_module("torch")
        model_path = self._model_path(config)
        checkpoint = self._load_runtime_checkpoint(torch, model_path)
        self._restore_runtime(checkpoint, config, torch)
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
        if checkpoint.get("algorithm") != "patchcore_primary":
            raise ValueError("Model is not a PatchCore-primary checkpoint; retrain it")
        if int(checkpoint.get("version", 0)) != PATCHCORE_MODEL_VERSION:
            raise RuntimeError("PatchCore model predates calibrated broken-fin geometry; retrain it")
        geometry_calibrations = checkpoint.get("geometry_calibrations")
        if not isinstance(geometry_calibrations, list) or not geometry_calibrations:
            raise RuntimeError("PatchCore model has no fin-geometry calibration; retrain it")
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
        else:
            ribs = np.zeros_like(band)
        content = self._content_mask(canonical)
        authority = edge_authority_mask(raw.shape, config.patchcore_edge_ignore_ratio, content)
        valid = band & content & ~ribs & (authority >= 1.0)
        glare = glare_evidence(roi, raw.shape, top=config.inspection_band_top_ratio,
                               bottom=config.inspection_band_bottom_ratio)
        thresholds = calibration["thresholds"]
        fail = config.patchcore_fail_threshold or thresholds["fail"]
        geometry_started = time.perf_counter()
        geometry = self._inspect_geometry(roi, config, geometry_calibrations)
        reflection = glare.mask if config.glare_rejection_enabled else np.zeros_like(valid)
        scoring_pixels = valid & ~(reflection & ~(geometry.defect_mask & (geometry.score >= config.geometry_candidate_threshold)))
        image_score = float(np.quantile(raw[scoring_pixels], .995)) if np.any(scoring_pixels) else 0.0
        area = evaluate_heatmap_area(
            raw, valid, reflection, geometry.defect_mask,
            patch_threshold=thresholds["patch_p999"], image_score=image_score, fail_threshold=fail,
            geometry_score=geometry.score, geometry_threshold=config.geometry_candidate_threshold,
            tolerance_percent=config.heatmap_tolerance_percent,
            min_component_px=config.min_defect_area_px, section_count=config.patchcore_section_count,
            scratch_max_width_px=config.scratch_max_width_px, scratch_min_aspect=config.scratch_min_aspect,
        )
        decision = area.decision
        candidate_sections = area.sections if decision.status in {"FAIL", "CANDIDATE"} else ()
        display, boxes = inspection_overlay(roi, raw * area.valid_mask, decision.confirmed_mask, status=decision.status,
                                             anomaly_score=image_score, min_box_area_px=config.min_defect_area_px,
                                             score_normalizer=max(fail, 1e-6))
        total_ms = (time.perf_counter() - started) * 1000
        return InspectionResult(decision.status, image_score, area.area_px, area.percentage / 100, list(candidate_sections),
                                display_image=display, defect_boxes=boxes, geometry_score=geometry.score,
                                periodicity_score=geometry.periodicity_score, glare_score=glare.score,
                                hybrid_score=max(image_score / max(fail, 1e-6), geometry.score),
                                reason_codes=decision.reason_codes, view_valid=decision.status != "VIEW INVALID",
                                candidate_sections=candidate_sections,
                                raw_heatmap=raw.copy(), filtered_anomaly_mask=area.mask.copy(),
                                valid_area_px=area.valid_area_px, anomaly_percentage=area.percentage,
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
