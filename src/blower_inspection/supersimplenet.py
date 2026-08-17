"""YOLO-localised SuperSimpleNet anomaly inspection.

YOLO owns localisation and tracking in the application.  This module therefore
only ever trains or predicts on the exact component crop, keeping background and
fixture variation out of the anomaly model.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .camera import crop_component_roi
from .config import PartModelConfig
from .trainer import (
    InspectionResult,
    broken_fin_mask,
    clean_mask,
    cylindrical_sector_statistics,
    cylindrical_surface_mask,
    inspection_overlay,
    list_images,
)
from .yolo_tracking import YoloByteTrackDetector

SUPERSIMPLENET_MODEL_VERSION = 1


@dataclass(frozen=True)
class SuperSimpleNetSettings:
    """Training controls passed to anomalib's engine."""

    max_epochs: int = 100
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 0
    random_seed: int = 42


def _require(module_name: str) -> Any:
    if importlib.util.find_spec(module_name) is None:
        raise RuntimeError(
            f"Missing dependency '{module_name}'. Install the industrial stack with "
            "`pip install -e .[industrial]`."
        )
    return importlib.import_module(module_name)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def localize_anomaly_scores(
    score_map: np.ndarray,
    surface_mask: np.ndarray,
    *,
    minimum_contrast_span: float = 0.15,
) -> tuple[np.ndarray, float]:
    """Remove a broad model offset while retaining localized defects.

    SuperSimpleNet can assign a high common offset to an otherwise normal crop
    when lighting or colour handling shifts slightly. Treating that absolute
    offset as a pixel defect paints the complete component red. Physical weld
    and fin defects are local responses, so scores are measured above the 20th
    percentile of the inspected surface. The fixed minimum contrast span avoids
    amplifying tiny sensor noise into an anomaly.
    """
    surface_scores = score_map[surface_mask]
    if surface_scores.size == 0:
        return np.zeros_like(score_map, dtype=np.float32), 0.0
    baseline = float(np.percentile(surface_scores, 20.0))
    contrast_span = max(1.0 - baseline, minimum_contrast_span)
    localized = np.clip((score_map.astype(np.float32) - baseline) / contrast_span, 0.0, 1.0)
    return localized.astype(np.float32), baseline


def suppress_broad_response(
    score_map: np.ndarray,
    defect_mask: np.ndarray,
    surface_mask: np.ndarray,
    *,
    maximum_coverage_ratio: float = 0.50,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Reject non-local model responses that cover most of a valid component.

    A pixel anomaly model does not know that a long normal fin edge is not one
    physical defect. When those responses connect, contour rendering outlines
    the complete blower even though no localized fault exists. Production
    defects for this station are local; a response covering more than half of
    the inspected surface is therefore treated as model/background
    drift rather than painted as a defect. The independent fin-continuity check
    is added after this guard, so a real broken fin remains visible.
    """
    surface_area = int(np.count_nonzero(surface_mask))
    coverage = int(np.count_nonzero(defect_mask & surface_mask)) / max(surface_area, 1)
    if coverage <= maximum_coverage_ratio:
        return score_map, defect_mask, False
    return np.zeros_like(score_map, dtype=np.float32), np.zeros_like(defect_mask, dtype=bool), True


def exceeds_defect_tolerance(
    defect_area: int,
    surface_area: int,
    minimum_pixels: int,
    maximum_ratio: float,
) -> bool:
    """Return whether confirmed defect coverage exceeds operator tolerance."""
    ratio = defect_area / max(surface_area, 1)
    return defect_area >= minimum_pixels and ratio >= maximum_ratio


class SuperSimpleNetInspector:
    """Train anomalib SuperSimpleNet and inspect exact YOLO component crops."""

    def __init__(self, settings: SuperSimpleNetSettings | None = None, device: str | None = None) -> None:
        self.settings = settings or SuperSimpleNetSettings()
        self.device = device or os.environ.get("BLOWER_INSPECTION_DEVICE")
        self._inferencers: dict[Path, tuple[float, Any]] = {}

    def train(self, config: PartModelConfig) -> Path:
        paths = list_images(config.normal_image_dir)
        if len(paths) < 20:
            raise ValueError(
                f"SuperSimpleNet training needs at least 20 normal images in {config.normal_image_dir}; "
                f"found {len(paths)}. Use 100+ for production validation."
            )
        if config.yolo_model_path is None:
            raise ValueError(f"No yolo_model_path is configured for {config.id}. Select best.pt before training.")

        data_module = _require("anomalib.data")
        models = _require("anomalib.models")
        engine_module = _require("anomalib.engine")
        detector = YoloByteTrackDetector(config.yolo_model_path, config.yolo_confidence)

        with tempfile.TemporaryDirectory(prefix=f"{config.id}-supersimplenet-") as temporary:
            root = Path(temporary)
            normal = root / "normal"
            normal.mkdir()
            for index, path in enumerate(paths):
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"Unable to read training image: {path}")
                crop = detector.exact_crop(image)
                destination = normal / f"{index:06d}{path.suffix.lower()}"
                if not cv2.imwrite(str(destination), crop):
                    raise OSError(f"Unable to stage YOLO crop: {destination}")

            datamodule = data_module.Folder(
                name=config.id,
                root=root,
                normal_dir="normal",
                # The production dataset is intentionally normal-only.  Let
                # anomalib synthesize held-out anomalous test samples rather
                # than silently requiring a defect directory.
                test_split_mode="synthetic",
                val_split_mode="from_test",
                train_batch_size=self.settings.train_batch_size,
                eval_batch_size=self.settings.eval_batch_size,
                num_workers=self.settings.num_workers,
            )
            model = models.Supersimplenet()
            engine = engine_module.Engine(
                max_epochs=self.settings.max_epochs,
                default_root_dir=root / "runs",
            )
            engine.fit(model=model, datamodule=datamodule)
            best_path = Path(engine.best_model_path)
            if not best_path.is_file():
                raise RuntimeError("anomalib did not produce a SuperSimpleNet checkpoint")
            config.model_file.parent.mkdir(parents=True, exist_ok=True)
            # ``Engine.fit`` produces a Lightning ``.ckpt`` training
            # checkpoint.  TorchInferencer does *not* accept that format; it
            # consumes the Torch model produced by ``Engine.export``.  Keep the
            # checkpoint beside the exported model so it can be re-exported
            # after an anomalib upgrade without repeating 100 epochs.
            checkpoint_path = config.model_file.with_suffix(".ckpt")
            shutil.copy2(best_path, checkpoint_path)
            self._export_checkpoint(checkpoint_path, config.model_file, model=model, engine=engine)

        metadata = {
            "version": SUPERSIMPLENET_MODEL_VERSION,
            "algorithm": "supersimplenet",
            "trained_at": datetime.now().isoformat(),
            "source_count": len(paths),
            "settings": asdict(self.settings),
            "model_id": config.id,
            "yolo_model_path": str(config.yolo_model_path),
        }
        self._metadata_path(config.model_file).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        self._inferencers.pop(config.model_file, None)
        return config.model_file

    def inspect(
        self,
        config: PartModelConfig,
        image: np.ndarray,
        *,
        save_outputs: bool = True,
        crop_to_component: bool = True,
    ) -> InspectionResult:
        if crop_to_component:
            image = crop_component_roi(image, roi_ratios=config.roi_ratios)
        model_file = self._ensure_inference_model(config.model_file)

        # OpenCV frames are BGR, while anomalib's ndarray inference contract is
        # RGB. Passing BGR here creates a train/inference colour shift and was
        # the primary cause of normal components scoring ~0.97 everywhere.
        inference_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        prediction = self._inferencer(model_file).predict(image=inference_image)
        raw_score_map = self._prediction_map(prediction, image.shape[:2])
        surface = cylindrical_surface_mask(raw_score_map.shape)
        score_map, score_baseline = localize_anomaly_scores(raw_score_map, surface)
        # anomalib anomaly maps are calibrated to 0..1 by the trained model.
        threshold = float(np.clip(config.anomaly_threshold / 10.0, 0.05, 0.95))
        defect_mask = clean_mask((score_map >= threshold) & surface)
        score_map, defect_mask, broad_response_suppressed = suppress_broad_response(
            score_map, defect_mask, surface
        )
        # SuperSimpleNet handles general texture/weld anomalies. Preserve an
        # independent geometry signal for the specific failure visible in the
        # operator's marked example: a localized interruption of a long fin.
        fin_breaks = broken_fin_mask(image, score_map.shape) & surface
        defect_mask = clean_mask(defect_mask | fin_breaks)
        score_map = np.maximum(score_map, fin_breaks.astype(np.float32))
        defect_area = int(defect_mask.sum())
        surface_area = int(np.count_nonzero(surface))
        defect_area_ratio = defect_area / max(surface_area, 1)
        anomaly_score = float(np.max(score_map[surface])) if np.any(surface) else 0.0
        bad_ratio, bad_sectors = cylindrical_sector_statistics(
            surface, score_map, config.expected_fins, min_bad_score=threshold
        )
        coverage_failed = exceeds_defect_tolerance(
            defect_area, surface_area, config.min_defect_area_px, config.max_defect_area_ratio
        )
        status = "FAIL" if coverage_failed or bad_ratio >= config.max_bad_sector_ratio else "PASS"
        display, boxes = inspection_overlay(
            image,
            score_map,
            defect_mask,
            status=status,
            anomaly_score=anomaly_score,
            min_box_area_px=config.min_defect_area_px,
            score_normalizer=threshold,
        )
        overlay_path = report_path = None
        if save_outputs:
            overlay_path, report_path = self._save_outputs(
                config, display, status, anomaly_score, defect_area, bad_ratio, bad_sectors,
                defect_area_ratio, score_baseline, broad_response_suppressed,
            )
        return InspectionResult(
            status, anomaly_score, defect_area, bad_ratio, bad_sectors,
            overlay_path, report_path, display, boxes, defect_area_ratio,
        )

    def runtime_device_name(self) -> str:
        return self.device or ("cuda" if _require("torch").cuda.is_available() else "cpu")

    def _ensure_inference_model(self, configured_path: Path) -> Path:
        """Return an exported Torch model, upgrading an existing ckpt once.

        The first SuperSimpleNet implementation incorrectly copied Lightning's
        training checkpoint to the configured path.  Existing installations
        may consequently have ``supersimplenet.ckpt`` where the application now
        expects ``supersimplenet.pt``. A sibling checkpoint is used for a safe
        automatic migration; a configured ``.ckpt`` is migrated directly.
        """
        if configured_path.suffix.lower() in {".pt", ".pth"} and configured_path.is_file():
            return configured_path

        output_path = configured_path.with_suffix(".pt")
        checkpoint_candidates = (
            configured_path if configured_path.suffix.lower() == ".ckpt" else None,
            configured_path.with_suffix(".ckpt"),
        )
        checkpoint = next((path for path in checkpoint_candidates if path is not None and path.is_file()), None)
        if checkpoint is None:
            raise FileNotFoundError(
                f"SuperSimpleNet model has not been trained: {configured_path}. "
                "Train the selected model to create an exported .pt model."
            )
        return self._export_checkpoint(checkpoint, output_path)

    def _export_checkpoint(
        self,
        checkpoint_path: Path,
        output_path: Path,
        *,
        model: Any | None = None,
        engine: Any | None = None,
    ) -> Path:
        """Export an anomalib Lightning checkpoint to TorchInferencer format."""
        deploy = _require("anomalib.deploy")
        if model is None:
            model = _require("anomalib.models").Supersimplenet()
        if engine is None:
            engine = _require("anomalib.engine").Engine()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="supersimplenet-export-") as temporary:
            exported = Path(engine.export(
                model=model,
                export_type=deploy.ExportType.TORCH,
                export_root=Path(temporary),
                ckpt_path=checkpoint_path,
            ))
            if not exported.is_file():
                raise RuntimeError(f"anomalib did not export a Torch model from {checkpoint_path}")
            temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
            shutil.copy2(exported, temporary_output)
            temporary_output.replace(output_path)
        self._inferencers.pop(output_path, None)
        return output_path

    def _inferencer(self, model_file: Path) -> Any:
        stamp = model_file.stat().st_mtime
        cached = self._inferencers.get(model_file)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        deploy = _require("anomalib.deploy")
        # ``None`` asks anomalib to select CUDA when available; unlike the
        # string "auto", it is also accepted by PyTorch's device handling.
        inferencer = deploy.TorchInferencer(path=model_file, device=self.device)
        self._inferencers[model_file] = (stamp, inferencer)
        return inferencer

    @staticmethod
    def _prediction_map(prediction: Any, shape: tuple[int, int]) -> np.ndarray:
        value = getattr(prediction, "anomaly_map", None)
        if value is None and isinstance(prediction, dict):
            value = prediction.get("anomaly_map")
        if value is None:
            raise RuntimeError("anomalib prediction did not contain an anomaly_map")
        result = np.squeeze(_as_numpy(value)).astype(np.float32)
        if result.ndim != 2:
            raise RuntimeError(f"Unexpected SuperSimpleNet anomaly-map shape: {result.shape}")
        if result.shape != shape:
            result = cv2.resize(result, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
        return np.clip(result, 0.0, 1.0)

    @staticmethod
    def _metadata_path(model_file: Path) -> Path:
        return model_file.with_suffix(model_file.suffix + ".json")

    @staticmethod
    def _save_outputs(
        config: PartModelConfig,
        display: np.ndarray,
        status: str,
        anomaly_score: float,
        defect_area: int,
        bad_ratio: float,
        bad_sectors: list[int],
        defect_area_ratio: float,
        score_baseline: float,
        broad_response_suppressed: bool,
    ) -> tuple[Path, Path]:
        config.result_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base = config.result_dir / f"{stamp}_{status.lower()}_supersimplenet"
        overlay_path = base.with_suffix(".png")
        cv2.imwrite(str(overlay_path), display)
        report_path = base.with_suffix(".json")
        report_path.write_text(json.dumps({
            "algorithm": "supersimplenet", "model_id": config.id, "status": status,
            "anomaly_score": anomaly_score, "defect_area_px": defect_area,
            "defect_area_ratio": defect_area_ratio,
            "bad_sector_ratio": bad_ratio, "bad_sectors": bad_sectors,
            "raw_score_baseline": score_baseline,
            "broad_response_suppressed": broad_response_suppressed,
            "created_at": datetime.now().isoformat(), "overlay_path": str(overlay_path),
        }, indent=2) + "\n", encoding="utf-8")
        return overlay_path, report_path
