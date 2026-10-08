"""Original PatchCore embeddings/coreset, exposed as a raw spatial distance map.

No PaDiM, distillation, geometry vote, or image-score acceptance gate is used.
The existing feature hooks, seeded projection, and coreset sampler are reused.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from .anomaly_models import HybridPatchcorePadimInspector, HybridTrainingSettings, _FeatureHook, require_module
from .fixed_settings import InspectionSettings
from .patchcore_inspector import DuplicateImageFilter, evenly_limit_indices
from .stationary_roi import InvalidView, PreparedROI, QualityEvidence, YoloROI
from .trainer import list_images
from .training_cache import DiskPatchEmbeddings
from .training_progress import TrainingProgress


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class _CachedROI:
    path: Path
    source: Path
    bounds: tuple[int, int, int, int]
    yolo_confidence: float
    quality: QualityEvidence

    def load(self) -> PreparedROI:
        with np.load(self.path, allow_pickle=False) as stored:
            return PreparedROI(None, stored["image"], stored["mask"], self.bounds,
                               self.yolo_confidence, self.quality)


class SpatialPatchCore(HybridPatchcorePadimInspector):
    VERSION = 1

    def __init__(self, config: InspectionSettings, device: str | None = None) -> None:
        self.config = config.validate()
        p = config.patchcore
        super().__init__(HybridTrainingSettings(image_size=p.input_width, backbone=p.backbone,
                         embedding_layers=p.embedding_layers, coreset_ratio=p.coreset_ratio,
                         max_coreset_patches=p.memory_bank_size, runtime_memory_bank_limit=p.memory_bank_size,
                         max_training_images=p.max_training_images), device)
        self._loaded_path: Path | None = None
        self._loaded_stamp: tuple[int, int] | None = None
        self._checkpoint: dict | None = None
        self._yolo_digest: str | None = None

    def signature(self) -> dict:
        p, y = self.config.patchcore, self.config.yolo
        if self._yolo_digest is None:
            self._yolo_digest = file_digest(y.model_path)
        return {"input_width": p.input_width, "input_height": p.input_height,
                "backbone": p.backbone, "embedding_layers": list(p.embedding_layers),
                "normalization_mean": [.485, .456, .406], "normalization_std": [.229, .224, .225],
                "resize": "aspect_preserving_letterbox", "color": "BGR_to_RGB",
                "yolo_sha256": self._yolo_digest, "yolo_confidence": y.confidence, "yolo_iou": y.iou,
                "yolo_class_id": y.class_id, "padding_x_percent": y.padding_x_percent,
                "padding_y_percent": y.padding_y_percent, "alignment_enabled": p.alignment_enabled,
                "alignment_max_translation_percent": p.alignment_max_translation_percent,
                "alignment_min_correlation": p.alignment_min_correlation}

    def _preprocess_image(self, image: np.ndarray):
        p = self.config.patchcore
        if image.shape[:2] != (p.input_height, p.input_width):
            raise ValueError("PatchCore requires the shared canonical ROI; no implicit resizing")
        torch = require_module("torch")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0)

    def _extract_embeddings_with_backbone(self, tensor, backbone, hook, torch):
        with torch.inference_mode():
            hook.clear()
            backbone(tensor.to(self._device(torch)))
            maps = [hook.features[layer].float() for layer in self.settings.embedding_layers]
            target = maps[0].shape[-2:]
            maps = [torch.nn.functional.interpolate(m, size=target, mode="bilinear", align_corners=False)
                    if m.shape[-2:] != target else m for m in maps]
            embedding = torch.cat(maps, dim=1)
            p = self.config.patchcore
            grid = (max(1, p.input_height // 8), max(1, p.input_width // 8))
            embedding = torch.nn.functional.adaptive_avg_pool2d(embedding, grid)
            embedding = self._project_embedding(embedding, torch)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return embedding.contiguous(), grid

    def _load(self, angle: int = 60) -> dict:
        path = self.config.patchcore.path_for_angle(angle).resolve()
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        if path == self._loaded_path and stamp == self._loaded_stamp:
            assert self._checkpoint is not None
            return self._checkpoint
        torch, models = require_module("torch"), require_module("torchvision.models")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("algorithm") != "patchcore_fixed_spatial" or checkpoint.get("version") != self.VERSION:
            raise ValueError("This is a legacy/different checkpoint. Preserve it for --legacy and train-fixed a new spatial checkpoint.")
        if checkpoint.get("preprocessing") != self.signature():
            raise ValueError("PatchCore preprocessing/YOLO weights changed; restore training settings or retrain")
        bank = checkpoint.get("memory_bank")
        if not isinstance(bank, torch.Tensor) or bank.ndim != 2 or len(bank) == 0 or not torch.isfinite(bank).all():
            raise ValueError("Invalid PatchCore memory bank")
        if hashlib.sha256(bank.numpy().tobytes()).hexdigest() != checkpoint.get("memory_bank_sha256"):
            raise ValueError("PatchCore memory-bank digest mismatch")
        backbone = getattr(models, self.settings.backbone)(weights=None)
        backbone.load_state_dict(checkpoint["backbone_state"])
        backbone.eval().to(self._device(torch))
        hook = _FeatureHook(self.settings.embedding_layers)
        hook.attach(backbone)
        if self._backbone_cache is not None:
            self._backbone_cache[1].close()
        self._backbone_cache = (backbone, hook, torch)
        self._checkpoint = checkpoint
        self._loaded_path, self._loaded_stamp = path, stamp
        return checkpoint

    def validate_ready(self) -> None:
        angles = range(60, 361, 60) if self.config.patchcore.angle_specific else (60,)
        for angle in angles:
            self._load(angle)

    def _align(self, roi: PreparedROI, reference: np.ndarray, reference_mask: np.ndarray) -> PreparedROI:
        if not self.config.patchcore.alignment_enabled:
            return roi
        source = cv2.cvtColor(roi.image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        target = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        transform = np.eye(2, 3, dtype=np.float32)
        try:
            correlation, transform = cv2.findTransformECC(target, source, transform, cv2.MOTION_TRANSLATION,
                         (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-5),
                         (roi.content_mask & reference_mask).astype(np.uint8))
        except cv2.error as exc:
            raise InvalidView("ROI ALIGNMENT FAILED") from exc
        h, w = source.shape
        maximum = self.config.patchcore.alignment_max_translation_percent / 100
        if correlation < self.config.patchcore.alignment_min_correlation or abs(transform[0, 2]) > w * maximum or abs(transform[1, 2]) > h * maximum:
            raise InvalidView("ROI ALIGNMENT OUTSIDE LIMITS")
        image = cv2.warpAffine(roi.image, transform, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT)
        mask = cv2.warpAffine(roi.content_mask.astype(np.uint8), transform, (w, h),
                             flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP).astype(bool)
        return replace(roi, image=image, content_mask=mask & reference_mask)

    def infer(self, roi: PreparedROI, angle: int = 60) -> tuple[np.ndarray, float, PreparedROI]:
        torch = require_module("torch")
        checkpoint = self._load(angle)
        roi = self._align(roi, checkpoint["reference_roi"].numpy(), checkpoint["reference_mask"].numpy())
        features, grid = self._extract_embeddings_from_tensor(self._preprocess_image(roi.image))
        flat = features.flatten(2).permute(0, 2, 1).reshape(-1, features.shape[1])
        bank = checkpoint["memory_bank"].to(flat.device)
        # Chunk both dimensions: bound temporary distance memory even with a large bank.
        nearest = []
        with torch.inference_mode():
            for queries in flat.split(256):
                distances = torch.full((len(queries),), float("inf"), device=queries.device)
                for reference in bank.split(2048):
                    distances = torch.minimum(distances, torch.cdist(queries, reference).min(dim=1).values)
                nearest.append(distances)
        small = torch.cat(nearest).reshape(grid).cpu().numpy().astype(np.float32)
        heatmap = cv2.resize(small, (roi.image.shape[1], roi.image.shape[0]), interpolation=cv2.INTER_LINEAR)
        if not np.isfinite(heatmap).all():
            raise RuntimeError("PatchCore produced nonfinite distances")
        score = float(np.max(heatmap[roi.content_mask]))
        return heatmap, score, roi

    def train_fixed(self, angle: int | None = None, progress_callback=None) -> Path:
        p = self.config.patchcore
        if p.angle_specific and angle not in range(60, 361, 60):
            raise ValueError("Angle-specific training requires --angle 60/120/180/240/300/360")
        directory = Path(p.normal_image_dir) / f"view_{angle}" if p.angle_specific else Path(p.normal_image_dir)
        paths = list_images(directory)
        if len(paths) < 20:
            raise ValueError(f"At least 20 GOOD source images are required in {directory}")
        with tempfile.TemporaryDirectory(prefix="patchcore-training-") as cache:
            return self._train_cached(paths, Path(cache), angle, progress_callback)

    def _train_cached(self, paths, cache_dir, angle, progress_callback):
        p = self.config.patchcore
        detector = YoloROI(self.config)
        accepted, rejected = [], []
        duplicate_filter = DuplicateImageFilter()
        qualified_count = 0
        started = time.monotonic()
        reference = None
        def progress(stage, completed, total):
            if progress_callback:
                progress_callback(TrainingProgress(stage, completed, total, time.monotonic() - started))
        for index, path in enumerate(paths, 1):
            frame = cv2.imread(str(path))
            try:
                roi = detector.prepare(frame, retain_original=False)
                if reference is None:
                    reference = roi
                roi = self._align(roi, reference.image, reference.content_mask)
                qualified_count += 1
                if duplicate_filter.accept(roi.image):
                    cached = cache_dir / f"roi_{index}.npz"
                    np.savez_compressed(cached, image=roi.image, mask=roi.content_mask)
                    accepted.append(_CachedROI(cached, path, roi.bounds, roi.yolo_confidence, roi.quality))
            except InvalidView as exc:
                rejected.append({"image": str(path), "reason": str(exc)})
            progress("Quality and YOLO ROI", index, len(paths))
        nonduplicate_count = len(accepted)
        accepted = [accepted[i] for i in evenly_limit_indices(len(accepted), p.max_training_images)]
        accepted_paths = [roi.source for roi in accepted]
        if len(accepted) < 10:
            raise ValueError(f"Only {len(accepted)} distinct qualified GOOD images remain; need at least 10. Rejections: {rejected[:5]}")
        del frame, roi, duplicate_filter
        split = max(3, len(accepted) // 5)
        train, calibration = accepted[:-split], accepted[-split:]
        torch = require_module("torch")
        with DiskPatchEmbeddings(cache_dir / "patches.bin") as features:
            for index, cached in enumerate(train, 1):
                roi = cached.load()
                fmap, _ = self._extract_embeddings_from_tensor(self._preprocess_image(roi.image))
                # Exclude letterbox patches from the GOOD memory bank.
                mask = cv2.resize(roi.content_mask.astype(np.uint8), (fmap.shape[-1], fmap.shape[-2]), interpolation=cv2.INTER_NEAREST).astype(bool)
                rows = fmap.flatten(2).permute(0, 2, 1).reshape(-1, fmap.shape[1]).cpu()
                features.append(rows[torch.from_numpy(mask.ravel())])
                progress("PatchCore features", index, len(train))
            progress("PatchCore coreset", 0, 1)
            memory, count = features.build_memory(self, torch)
        backbone, _, _ = self._build_backbone()
        path = p.path_for_angle(angle or 60)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {"version": self.VERSION, "algorithm": "patchcore_fixed_spatial",
                      "preprocessing": self.signature(), "settings": asdict(self.settings),
                      "memory_bank": memory.cpu(), "memory_bank_sha256": hashlib.sha256(memory.numpy().tobytes()).hexdigest(),
                      "backbone_state": {k: v.detach().cpu() for k, v in backbone.state_dict().items()},
                      "reference_roi": torch.from_numpy(reference.image.copy()),
                      "reference_mask": torch.from_numpy(reference.content_mask.copy()),
                      "training_manifest": [str(p) for p in accepted_paths[:-split]],
                      "calibration_manifest": [str(p) for p in accepted_paths[-split:]]}
        temp = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temp)
        temp.replace(path)
        self._loaded_stamp = None
        calibration_maps = []
        for cached in calibration:
            roi = cached.load()
            heatmap, _, aligned = self.infer(roi, angle or 60)
            calibration_maps.append(heatmap[aligned.content_mask])
        report = {"model": str(path), "angle": angle, "source_count": len(paths), "qualified_distinct_count": len(accepted),
                  "quality_accepted_images": qualified_count, "nonduplicate_images": nonduplicate_count,
                  "duplicates_removed": qualified_count - nonduplicate_count,
                  "training_limit_removed": nonduplicate_count - len(accepted),
                  "training_count": len(train), "calibration_count": len(calibration), "memory_candidates": count,
                  "memory_patches": len(memory), "rejected": rejected, "preprocessing": self.signature(),
                  "GOOD_distance_p999": float(np.quantile(np.concatenate(calibration_maps), .999)),
                  "training_manifest": checkpoint["training_manifest"], "calibration_manifest": checkpoint["calibration_manifest"],
                  "note": "Distance distribution only; production tolerances require independent labeled GOOD/NG validation."}
        path.with_suffix(".training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        progress("Completed", 1, 1)
        return path
