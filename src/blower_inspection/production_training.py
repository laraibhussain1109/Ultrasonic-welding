"""Bridge fixed-image settings to the desktop's production artifact contract."""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import cv2

from .anomaly_models import HybridTrainingSettings, require_module
from .config import ModelRegistry, PartModelConfig
from .fixed_settings import InspectionSettings
from .patchcore_inspector import PATCHCORE_MODEL_VERSION, PatchCoreInspector
from .stationary_roi import YoloROI
from .training_progress import TrainingProgress


def production_config(registry: ModelRegistry, settings: InspectionSettings, model_id: str | None) -> PartModelConfig:
    base = registry.get(model_id) if model_id else registry.active()
    if settings.patchcore.angle_specific or settings.patchcore.alignment_enabled:
        raise ValueError("The desktop uses a shared, unaligned model. Use --spatial-only for angle-specific/aligned engineering models.")
    if settings.yolo.padding_x_percent != settings.yolo.padding_y_percent:
        raise ValueError("Production training requires equal horizontal/vertical ROI padding")
    path = base.patchcore_model_file or base.model_file
    if path.suffix.lower() != ".pt":
        path = path.with_name("patchcore_primary.pt")
    p, y, q = settings.patchcore, settings.yolo, settings.quality
    return replace(base, normal_image_dir=Path(p.normal_image_dir), yolo_model_path=Path(y.model_path),
                   production_algorithm="patchcore_geometry", algorithm="hybrid_patchcore_geometry",
                   patchcore_primary=True, padim_enabled=False, distillation_enabled=False,
                   model_file=path, patchcore_model_file=path,
                   patchcore_calibration_file=base.patchcore_calibration_file or path.with_name("patchcore_calibration.json"),
                   image_size=p.input_width, patchcore_embedding_layers=p.embedding_layers,
                   patchcore_memory_bank_size=p.memory_bank_size, patchcore_coreset_ratio=p.coreset_ratio,
                   roi_padding_ratio=y.padding_x_percent / 100, yolo_confidence=y.confidence,
                   training_min_sharpness=q.minimum_blur_score, inference_min_sharpness=q.minimum_blur_score,
                   max_saturation_ratio=q.maximum_saturated_percent / 100)


def _save_registry(registry: ModelRegistry, config: PartModelConfig) -> None:
    registry.update_model_settings(config.id, yolo_model_path=config.yolo_model_path, model_file=config.model_file,
        patchcore_settings={"normal_image_dir": str(config.normal_image_dir), "image_size": config.image_size,
                            "production_algorithm": config.production_algorithm, "algorithm": config.algorithm,
                            "patchcore_primary": True, "padim_enabled": False, "distillation_enabled": False,
                            "patchcore_embedding_layers": list(config.patchcore_embedding_layers),
                            "patchcore_memory_bank_size": config.patchcore_memory_bank_size,
                            "patchcore_coreset_ratio": config.patchcore_coreset_ratio,
                            "roi_padding_ratio": config.roi_padding_ratio,
                            "yolo_confidence": config.yolo_confidence,
                            "patchcore_model_file": str(config.patchcore_model_file),
                            "patchcore_calibration_file": str(config.patchcore_calibration_file),
                            "training_min_sharpness": config.training_min_sharpness,
                            "inference_min_sharpness": config.inference_min_sharpness,
                            "max_saturation_ratio": config.max_saturation_ratio})


def train_production(registry: ModelRegistry, settings: InspectionSettings, model_id=None, progress_callback=None) -> Path:
    config = production_config(registry, settings, model_id)
    p = settings.patchcore
    model = PatchCoreInspector(HybridTrainingSettings(backbone=p.backbone, embedding_layers=p.embedding_layers,
                                                     max_training_images=p.max_training_images))
    path = model.train(config, progress_callback, roi_size=(p.input_width, p.input_height))
    model.validate_ready(config)
    _save_registry(registry, config)
    return path


def import_spatial(registry: ModelRegistry, settings: InspectionSettings, model_id=None,
                   source_path: str | Path | None = None, progress_callback=None) -> Path:
    """Reuse a completed spatial bank; fit only missing geometry and thresholds."""
    from .patchcore_spatial import SpatialPatchCore, file_digest

    config = production_config(registry, settings, model_id)
    source_path = Path(source_path or settings.patchcore.model_path)
    if source_path.resolve() in {config.model_file.resolve(), config.patchcore_calibration_file.resolve()}:
        raise ValueError("Keep the original spatial checkpoint at a different path from the production outputs")
    source_settings = replace(settings, patchcore=replace(settings.patchcore, model_path=str(source_path)))
    source = SpatialPatchCore(source_settings)
    checkpoint = source._load()
    manifests = checkpoint.get("training_manifest", []), checkpoint.get("calibration_manifest", [])
    if len(manifests[0]) < 7 or len(manifests[1]) < 3 or set(manifests[0]) & set(manifests[1]):
        raise ValueError("Spatial checkpoint needs disjoint GOOD training/calibration manifests (at least 7/3 images)")
    detector = YoloROI(source_settings)
    groups, masks, paths = [[], []], [[], []], [[], []]
    started = time.monotonic()
    total = sum(map(len, manifests))
    for group, manifest in enumerate(manifests):
        for entry in manifest:
            path = Path(entry)
            if not path.is_file():
                # Allow a reviewed GOOD folder to move with the installation.
                path = config.normal_image_dir / Path(entry.replace("\\", "/")).name
            frame = cv2.imread(str(path))
            if frame is None:
                raise FileNotFoundError(f"GOOD image required for production calibration not found: {entry}")
            roi = detector.prepare(frame, retain_original=False)
            groups[group].append(roi.image)
            masks[group].append(roi.content_mask)
            paths[group].append(path)
            if progress_callback:
                progress_callback(TrainingProgress("Production geometry/calibration ROIs", sum(map(len, groups)), total,
                                                   time.monotonic() - started))
    model = PatchCoreInspector(source.settings)
    model._backbone_cache = source._backbone_cache
    preprocessing = {**checkpoint["preprocessing"], "mode": "fixed_spatial", "roi_padding_ratio": config.roi_padding_ratio}
    model._feature_preprocessing = preprocessing
    bank = checkpoint["memory_bank"]
    bank_hash = checkpoint["memory_bank_sha256"]
    geometry = model._calibrate_geometry(groups[0], config)
    primary = {"version": PATCHCORE_MODEL_VERSION, "algorithm": "patchcore_primary",
               "settings": checkpoint["settings"], "memory_bank": bank, "memory_bank_hash": bank_hash,
               "backbone_state": checkpoint["backbone_state"], "preprocessing": preprocessing,
               "roi_settings": source_settings.to_dict(), "geometry_calibrations": geometry,
               "geometry_section_count": config.patchcore_section_count,
               "training_manifest": [str(p) for p in paths[0]], "calibration_manifest": [str(p) for p in paths[1]],
               "training_manifest_hash": model._manifest_hash(paths[0]),
               "imported_from": str(source_path), "source_sha256": file_digest(source_path)}
    torch = require_module("torch")
    maps = []
    for index, (image, mask) in enumerate(zip(groups[1], masks[1]), 1):
        maps.append(model._raw_map(image, primary, torch)[mask])
        if progress_callback:
            progress_callback(TrainingProgress("Production distance calibration", index, len(groups[1]), time.monotonic() - started))
    calibration = model._calibration(maps, config, bank_hash, primary["training_manifest_hash"], paths[1])
    report = {"status": "IMPORTED", "source_checkpoint": str(source_path), "source_sha256": primary["source_sha256"],
              "memory_bank_reused": True, "memory_patches": len(bank), "backbone_reused": True,
              "training_images": len(groups[0]), "calibration_images": len(groups[1]),
              "training_manifest": primary["training_manifest"], "calibration_manifest": primary["calibration_manifest"],
              "production_model": str(config.model_file), "calibration_file": str(config.patchcore_calibration_file)}
    model._publish_training(config, primary, calibration, report, torch)
    model.validate_ready(config)
    _save_registry(registry, config)
    return config.model_file
