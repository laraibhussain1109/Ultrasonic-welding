"""Fail-closed NVIDIA TAO VisualChangeNet Deploy runtime.

TAO training/export remains in NVIDIA's supported container.  This module owns
the production concerns around that model: artifact validation, GPU execution,
normal-set calibration, spatial post-processing and auditable results.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .camera import crop_component_roi
from .config import PartModelConfig
from .trainer import InspectionResult, clean_mask, cylindrical_sector_statistics, cylindrical_surface_mask, inspection_overlay, list_images

CALIBRATION_VERSION = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TaoCalibration:
    model_sha256: str
    reference_sha256: str
    pixel_threshold: float
    image_threshold: float
    normal_p999: float
    normal_score_p999: float
    sample_count: int
    created_at: str


class TaoInspector:
    """Run a two-input TAO VisualChangeNet ONNX model with safeguards."""

    def __init__(self, *, providers: list[str] | None = None) -> None:
        self.providers = providers
        # Keep calibration's GPU-first/CPU-fallback session separate from the
        # production GPU-only session.
        self._sessions: dict[tuple[Path, bool, bool], tuple[int, Any]] = {}

    @staticmethod
    def calibration_path(config: PartModelConfig) -> Path:
        return config.tao_calibration_file or config.model_file.with_suffix(".calibration.json")

    def _session(self, config: PartModelConfig, *, allow_cpu_fallback: bool = False) -> Any:
        path = config.model_file.resolve()
        if path.suffix.lower() != ".onnx":
            raise ValueError("NVIDIA TAO runtime requires a TAO Deploy .onnx export")
        if not path.is_file():
            raise FileNotFoundError(f"TAO model export not found: {path}")
        stamp = path.stat().st_mtime_ns
        cache_key = (path, config.tao_require_gpu, allow_cpu_fallback)
        cached = self._sessions.get(cache_key)
        if cached and cached[0] == stamp:
            return cached[1]
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install the TAO runtime with `pip install -e .[tao]`") from exc
        available = ort.get_available_providers()
        requested_gpu = self.providers or ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
        selected = [provider for provider in requested_gpu if provider in available]
        if config.tao_require_gpu and not selected and not allow_cpu_fallback:
            raise RuntimeError(f"TAO GPU provider unavailable (installed providers: {available}); inspection is inhibited")
        # ONNX Runtime tries providers in order. Calibration therefore uses the
        # trained model on TensorRT/CUDA whenever possible and falls back to CPU
        # only if no GPU provider can execute it.
        if (allow_cpu_fallback or not config.tao_require_gpu) and "CPUExecutionProvider" in available:
            selected.append("CPUExecutionProvider")
        if not selected:
            raise RuntimeError(f"No usable TAO execution provider is available (installed providers: {available})")
        session = ort.InferenceSession(str(path), providers=selected)
        active = session.get_providers()
        if config.tao_require_gpu and not allow_cpu_fallback and not any(p in active for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider")):
            raise RuntimeError(f"TAO session did not activate a GPU provider: {active}")
        if len(session.get_inputs()) != 2:
            raise RuntimeError(
                "VisualChangeNet ONNX must expose exactly two image inputs "
                "(golden reference and inspected image)"
            )
        self._sessions[cache_key] = (stamp, session)
        return session

    @staticmethod
    def _input_tensor(image: np.ndarray, shape: list[Any]) -> np.ndarray:
        if len(shape) != 4:
            raise RuntimeError(f"Unsupported TAO input shape: {shape}")
        nchw = shape[1] in (1, 3) or not isinstance(shape[-1], int)
        height = int(shape[2] if nchw else shape[1])
        width = int(shape[3] if nchw else shape[2])
        if height <= 0 or width <= 0:
            raise RuntimeError("TAO export must use fixed spatial input dimensions")
        rgb = cv2.cvtColor(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        value = rgb.astype(np.float32) / 255.0
        # TAO visual anomaly models use ImageNet-normalized RGB input.
        value = (value - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        if nchw:
            value = np.transpose(value, (2, 0, 1))
        return value[None].astype(np.float32)

    @staticmethod
    def reference_path(config: PartModelConfig) -> Path:
        return config.tao_reference_image or config.model_file.with_name("visual_changenet_reference.png")

    @staticmethod
    def _change_map(value: Any, change_class_index: int) -> np.ndarray:
        output = np.asarray(value, dtype=np.float32)
        while output.ndim > 2 and output.shape[0] == 1:
            output = output[0]
        if output.ndim == 3:
            if change_class_index >= output.shape[0]:
                raise RuntimeError(
                    f"VisualChangeNet change class {change_class_index} is absent from output {output.shape}"
                )
            logits = output
            logits = logits - np.max(logits, axis=0, keepdims=True)
            probabilities = np.exp(logits) / np.maximum(np.exp(logits).sum(axis=0, keepdims=True), 1e-12)
            output = probabilities[change_class_index]
        if output.ndim != 2:
            raise RuntimeError(f"Invalid VisualChangeNet spatial output: {output.shape}")
        if float(output.min()) < 0.0 or float(output.max()) > 1.0:
            output = 1.0 / (1.0 + np.exp(-np.clip(output, -30.0, 30.0)))
        return output.astype(np.float32)

    @staticmethod
    def _discover_map_output(values: dict[str, Any]) -> str:
        """Identify TAO's final segmentation map among deep-supervision outputs."""
        candidates = {
            name: np.asarray(value)
            for name, value in values.items()
            if np.asarray(value).ndim >= 3
        }
        if not candidates:
            raise RuntimeError(
                f"TAO export has no spatial anomaly-map output; outputs={list(values)}"
            )
        if len(candidates) == 1:
            return next(iter(candidates))

        # VisualChangeNet exports can expose intermediate decoder maps as
        # output0..output3 plus the fused `output_final`.  The fused map is the
        # inference result; the numbered maps are training/deep-supervision
        # heads and must not be used for threshold calibration.
        final = [
            name
            for name in candidates
            if name.lower() in {"output_final", "output_final:0", "final"}
            or name.lower().endswith(("/output_final", "/output_final:0"))
        ]
        if len(final) == 1:
            return final[0]
        raise RuntimeError(
            "Set tao_output_name explicitly; unable to identify the final anomaly "
            f"map from {list(values)}"
        )

    def _infer(
        self,
        config: PartModelConfig,
        reference: np.ndarray,
        image: np.ndarray,
        *,
        allow_cpu_fallback: bool = False,
    ) -> tuple[np.ndarray, float]:
        session = self._session(config, allow_cpu_fallback=allow_cpu_fallback)
        inputs = session.get_inputs()
        reference_name = config.tao_reference_input_name or config.tao_input_name or inputs[0].name
        test_name = config.tao_test_input_name or inputs[1].name
        by_name = {item.name: item for item in inputs}
        if reference_name not in by_name or test_name not in by_name or reference_name == test_name:
            raise RuntimeError(
                f"Invalid VisualChangeNet input bindings reference={reference_name!r}, "
                f"test={test_name!r}; inputs={list(by_name)}"
            )
        outputs = session.run(None, {
            reference_name: self._input_tensor(reference, by_name[reference_name].shape),
            test_name: self._input_tensor(image, by_name[test_name].shape),
        })
        names = [item.name for item in session.get_outputs()]
        values = dict(zip(names, outputs))
        map_name = config.tao_output_name
        if map_name and map_name not in values:
            raise RuntimeError(f"Configured TAO anomaly-map output '{map_name}' is absent; outputs={names}")
        if map_name is None:
            map_name = self._discover_map_output(values)
        anomaly_map = self._change_map(values[map_name], config.tao_change_class_index)
        if not np.isfinite(anomaly_map).all():
            raise RuntimeError(f"Invalid TAO anomaly map shape/data: {anomaly_map.shape}")
        if config.tao_score_output_name:
            if config.tao_score_output_name not in values:
                raise RuntimeError(f"Configured score output '{config.tao_score_output_name}' is absent")
            score = float(np.asarray(values[config.tao_score_output_name]).squeeze())
        else:
            score = float(np.max(anomaly_map))
        if not np.isfinite(score):
            raise RuntimeError("TAO returned a non-finite anomaly score")
        return anomaly_map, score

    def train(self, config: PartModelConfig) -> Path:
        """Calibrate an already trained/exported TAO model on production normals."""
        paths = list_images(config.normal_image_dir)
        if len(paths) < 20:
            raise ValueError(f"TAO calibration requires at least 20 reviewed normal images; found {len(paths)}")
        crops: list[np.ndarray] = []
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Unreadable calibration image: {path}")
            crops.append(crop_component_roi(image, roi_ratios=config.roi_ratios))
        reference_path = self.reference_path(config)
        if reference_path.is_file():
            reference = cv2.imread(str(reference_path))
            if reference is None:
                raise ValueError(f"Unreadable VisualChangeNet reference: {reference_path}")
        else:
            # Select a real reviewed crop nearest the normal-set median. A real
            # medoid avoids the blurred edges produced by saving a median image.
            sample = crops[: min(50, len(crops))]
            thumbs = np.stack([cv2.resize(item, (128, 128)) for item in sample]).astype(np.float32)
            median = np.median(thumbs, axis=0)
            index = int(np.argmin(np.mean(np.abs(thumbs - median), axis=(1, 2, 3))))
            reference = sample[index]
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(reference_path), reference):
                raise RuntimeError(f"Unable to save VisualChangeNet reference: {reference_path}")
        pixels: list[np.ndarray] = []
        scores: list[float] = []
        for crop in crops:
            anomaly_map, score = self._infer(
                config, reference, crop, allow_cpu_fallback=True
            )
            pixels.append(anomaly_map.ravel())
            scores.append(score)
        normal_pixels = np.concatenate(pixels)
        p999 = float(np.quantile(normal_pixels, 0.999))
        score_p999 = float(np.quantile(scores, 0.999))
        # A margin above the measured normal tail avoids thresholding camera
        # noise while remaining on the model's native, non-normalized scale.
        pixel_threshold = p999 + max(float(np.std(normal_pixels)) * 3.0, 1e-6)
        image_threshold = score_p999 + max(float(np.std(scores)) * 3.0, 1e-6)
        calibration = TaoCalibration(_sha256(config.model_file), _sha256(reference_path), pixel_threshold, image_threshold, p999, score_p999, len(paths), datetime.now(timezone.utc).isoformat())
        output = self.calibration_path(config)
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(output.suffix + ".tmp")
        temp.write_text(json.dumps({"version": CALIBRATION_VERSION, **calibration.__dict__}, indent=2) + "\n", encoding="utf-8")
        temp.replace(output)
        return output

    def _calibration(self, config: PartModelConfig) -> TaoCalibration:
        path = self.calibration_path(config)
        if not path.is_file():
            raise FileNotFoundError(f"TAO calibration missing: {path}; run TRAIN SELECTED MODEL after export")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.pop("version", None) != CALIBRATION_VERSION:
            raise RuntimeError("Unsupported TAO calibration version; recalibrate the model")
        calibration = TaoCalibration(**data)
        reference_path = self.reference_path(config)
        if not reference_path.is_file():
            raise RuntimeError("VisualChangeNet golden reference is missing; inspection is inhibited")
        if (calibration.sample_count < 20 or calibration.model_sha256 != _sha256(config.model_file)
                or calibration.reference_sha256 != _sha256(reference_path)):
            raise RuntimeError("TAO calibration is stale or insufficient; inspection is inhibited")
        return calibration

    def inspect(self, config: PartModelConfig, image: np.ndarray, *, save_outputs: bool = True, crop_to_component: bool = True) -> InspectionResult:
        if crop_to_component:
            image = crop_component_roi(image, roi_ratios=config.roi_ratios)
        calibration = self._calibration(config)
        reference = cv2.imread(str(self.reference_path(config)))
        if reference is None:
            raise RuntimeError("VisualChangeNet golden reference is unreadable; inspection is inhibited")
        raw_map, image_score = self._infer(config, reference, image)
        score_map = cv2.resize(raw_map, (config.image_size, config.image_size), interpolation=cv2.INTER_CUBIC)
        surface = cylindrical_surface_mask(score_map.shape)
        defect_mask = clean_mask((score_map >= calibration.pixel_threshold) & surface)
        defect_area = int(defect_mask.sum())
        bad_ratio, bad_sectors = cylindrical_sector_statistics(surface, score_map, config.expected_fins, min_bad_score=calibration.pixel_threshold)
        fail_area = max(config.min_defect_area_px, int(score_map.size * 0.0004))
        status = "FAIL" if image_score >= calibration.image_threshold or defect_area >= fail_area or bad_ratio >= config.max_bad_sector_ratio else "PASS"
        display, boxes = inspection_overlay(image, score_map, defect_mask, status=status, anomaly_score=image_score, min_box_area_px=config.min_defect_area_px, score_normalizer=calibration.pixel_threshold)
        overlay_path = report_path = None
        if save_outputs:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            config.result_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = config.result_dir / f"{stamp}_{status.lower()}_overlay.png"
            report_path = config.result_dir / f"{stamp}_{status.lower()}.json"
            cv2.imwrite(str(overlay_path), display)
            report_path.write_text(json.dumps({"algorithm": "nvidia_tao_visual_changenet", "model_sha256": calibration.model_sha256, "reference_sha256": calibration.reference_sha256, "status": status, "anomaly_score": image_score, "pixel_threshold": calibration.pixel_threshold, "image_threshold": calibration.image_threshold, "defect_area_px": defect_area, "bad_sector_ratio": bad_ratio, "bad_sectors": bad_sectors}, indent=2) + "\n", encoding="utf-8")
        return InspectionResult(status, image_score, defect_area, bad_ratio, bad_sectors, overlay_path, report_path, display, boxes)


def inspector_for_model(config: PartModelConfig) -> Any:
    if config.algorithm == "nvidia_tao":
        return TaoInspector()
    if config.algorithm == "hybrid_patchcore_padim":
        from .anomaly_models import HybridPatchcorePadimInspector
        return HybridPatchcorePadimInspector()
    raise ValueError(f"Unsupported inspection algorithm: {config.algorithm}")
