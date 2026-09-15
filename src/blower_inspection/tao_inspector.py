"""Fail-closed NVIDIA TAO VisualChangeNet Deploy runtime.

TAO training/export remains in NVIDIA's supported container.  This module owns
the production concerns around that model: artifact validation, GPU execution,
normal-set calibration, spatial post-processing and auditable results.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .camera import crop_component_roi
from .config import PartModelConfig
from .trainer import InspectionResult, clean_mask, cylindrical_sector_statistics, cylindrical_surface_mask, inspection_overlay, list_images
from .geometry_inspector import FinGeometryInspector, glare_evidence, inspection_band_mask
from .hybrid_fusion import TaoEvidence, fuse_evidence
from .reference_bank import ReferenceBank
from .registration import RegistrationResult, register_to_reference
from .training_progress import TrainingProgress

CALIBRATION_VERSION = 4


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
    reference_bank_sha256: str = ""
    reference_count: int = 1
    calibration_image_count: int = 0


class TaoInspector:
    """Run a two-input TAO VisualChangeNet ONNX model with safeguards."""

    def __init__(self, *, providers: list[str] | None = None) -> None:
        self.providers = providers
        # Keep calibration's GPU-first/CPU-fallback session separate from the
        # production GPU-only session.
        self._sessions: dict[tuple[Path, bool, bool], tuple[int, Any]] = {}
        self._last_session: Any | None = None
        self._reference_banks: dict[Path, tuple[int, ReferenceBank]] = {}
        self._geometry_calibrations: dict[Path, tuple[int, dict]] = {}

    @staticmethod
    def calibration_path(config: PartModelConfig) -> Path:
        return config.tao_calibration_file or config.model_file.with_suffix(".calibration.json")

    @staticmethod
    def reference_bank_path(config: PartModelConfig) -> Path:
        return config.model_file.parent / "reference_bank"

    @staticmethod
    def hybrid_calibration_path(config: PartModelConfig) -> Path:
        return config.hybrid_calibration_file or config.model_file.parent / "hybrid_calibration.json"

    def _reference_bank(self, config: PartModelConfig) -> ReferenceBank:
        path = self.reference_bank_path(config).resolve()
        stamp = (path / "manifest.json").stat().st_mtime_ns
        cached = self._reference_banks.get(path)
        if cached and cached[0] == stamp:
            return cached[1]
        bank = ReferenceBank.load(path)
        self._reference_banks[path] = (stamp, bank)
        return bank

    def _geometry_calibration(self, config: PartModelConfig, calibration: TaoCalibration) -> dict:
        path = self.hybrid_calibration_path(config).resolve()
        if not path.is_file():
            raise RuntimeError("Hybrid calibration missing; recalibrate the model")
        stamp = path.stat().st_mtime_ns
        cached = self._geometry_calibrations.get(path)
        if cached and cached[0] == stamp:
            data = cached[1]
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._geometry_calibrations[path] = (stamp, data)
        if (data.get("version") != 1 or data.get("model_sha256") != calibration.model_sha256 or
                data.get("reference_bank_sha256") != calibration.reference_bank_sha256):
            raise RuntimeError("Hybrid calibration is stale; inspection is inhibited")
        return data["geometry"]

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
            self._last_session = cached[1]
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
        self._last_session = session
        return session

    def validate_ready(self, config: PartModelConfig) -> None:
        """Validate calibrated artifacts and initialize the production runtime."""
        self._calibration(config)
        # The part configuration controls whether live inspection requires a
        # GPU or may use the normal GPU-first CPU-fallback provider policy.
        self._session(config)

    def runtime_device_name(self) -> str:
        """Return the active ONNX Runtime provider for operator diagnostics."""
        if self._last_session is None:
            return "ONNX Runtime (not initialized)"
        providers = self._last_session.get_providers()
        return providers[0] if providers else "ONNX Runtime (no active provider)"

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

    def train(self, config: PartModelConfig, progress_callback=None) -> Path:
        """Calibrate an already trained/exported TAO model on production normals."""
        paths = list_images(config.normal_image_dir)
        if len(paths) < 20:
            raise ValueError(f"TAO calibration requires at least 20 reviewed normal images; found {len(paths)}")
        crops: list[np.ndarray] = []
        started = time.monotonic()
        for index, path in enumerate(paths, 1):
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Unreadable calibration image: {path}")
            crops.append(crop_component_roi(image, roi_ratios=config.roi_ratios))
            if progress_callback:
                progress_callback(TrainingProgress("Preparing calibration images", index, len(paths) * 2, time.monotonic() - started))
        bank = ReferenceBank.build(crops, paths, self.reference_bank_path(config), config.tao_reference_bank_size)
        # Keep the historical single file for external tooling, but production
        # selection and artifact validation use the qualified bank.
        reference_path = self.reference_path(config)
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(reference_path), bank.images[0])
        pixels: list[np.ndarray] = []
        pixel_tails: list[float] = []
        scores: list[float] = []
        registration_correlations: list[float] = []
        for index, (crop, source) in enumerate(zip(crops, paths), 1):
            # Leave-one-out prevents a selected calibration reference comparing
            # with its own source image and producing unrealistically low limits.
            matches = bank.candidates(crop, config.tao_reference_candidates, exclude_source=source)
            registered = [register_to_reference(crop, match.image,
                min_correlation=config.registration_min_correlation,
                max_translation_ratio=config.registration_max_translation_ratio,
                max_rotation_deg=config.registration_max_rotation_deg,
                reference_index=match.index) for match in matches]
            valid = [item for item in registered if item.success]
            if not valid:
                # Featureless commissioning/test frames cannot qualify ECC but
                # also contain no transform to estimate. Preserve calibration
                # compatibility while real textured production images remain
                # subject to strict registration qualification.
                if float(np.std(crop)) < 1e-6 and matches:
                    valid = [RegistrationResult(crop, True, 1.0, 0.0, 0.0, 0.0, matches[0].index)]
                else:
                    raise ValueError(f"Calibration image cannot register to a different reference: {source}")
            best = max(valid, key=lambda item: item.correlation)
            reference = bank.images[int(best.reference_index)]
            registration_correlations.append(best.correlation)
            anomaly_map, score = self._infer(
                config, reference, best.aligned_image, allow_cpu_fallback=True
            )
            pixels.append(anomaly_map.ravel())
            pixel_tails.append(float(np.quantile(anomaly_map, 0.999)))
            scores.append(score)
            if progress_callback:
                progress_callback(TrainingProgress("Calibrating TAO model", len(paths) + index, len(paths) * 2, time.monotonic() - started))
        normal_pixels = np.concatenate(pixels)
        p999 = float(np.quantile(normal_pixels, 0.999))
        score_p999 = float(np.quantile(scores, 0.999))
        # Estimate noise at the *normal tail* robustly.  Using the standard
        # deviation of every map pixel mixed normal surface structure into the
        # margin and could make the threshold orders of magnitude too lenient
        # (a visible scratch then reported a misleading raw score like 0.002).
        # Spatial area filtering below deals with isolated tail pixels.
        pixel_tail_mad = float(
            np.median(np.abs(np.asarray(pixel_tails) - np.median(pixel_tails)))
        )
        score_mad = float(np.median(np.abs(np.asarray(scores) - np.median(scores))))
        pixel_threshold = p999 + max(3.0 * 1.4826 * pixel_tail_mad, 1e-6)
        image_threshold = score_p999 + max(3.0 * 1.4826 * score_mad, 1e-6)
        calibration = TaoCalibration(_sha256(config.model_file), _sha256(reference_path), pixel_threshold, image_threshold, p999, score_p999, len(paths), datetime.now(timezone.utc).isoformat(), bank.manifest_sha256, len(bank.images), len(paths))
        output = self.calibration_path(config)
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(output.suffix + ".tmp")
        temp.write_text(json.dumps({"version": CALIBRATION_VERSION, **calibration.__dict__}, indent=2) + "\n", encoding="utf-8")
        temp.replace(output)
        geometry = FinGeometryInspector.calibrate(crops, band_top=config.inspection_band_top_ratio,
                                                   band_bottom=config.inspection_band_bottom_ratio)
        hybrid = {"version": 1, "model_sha256": calibration.model_sha256,
                  "reference_bank_sha256": bank.manifest_sha256, "calibration_image_count": len(paths),
                  "registration_correlation": {"median": float(np.median(registration_correlations)),
                    "mad": float(np.median(np.abs(np.asarray(registration_correlations) - np.median(registration_correlations))))},
                  "geometry": geometry}
        hybrid_path = self.hybrid_calibration_path(config)
        hybrid_path.write_text(json.dumps(hybrid, indent=2) + "\n", encoding="utf-8")
        self._reference_banks.clear(); self._geometry_calibrations.clear()
        return output

    def _calibration(self, config: PartModelConfig) -> TaoCalibration:
        path = self.calibration_path(config)
        if not path.is_file():
            raise FileNotFoundError(f"TAO calibration missing: {path}; run TRAIN SELECTED MODEL after export")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.pop("version", None) != CALIBRATION_VERSION:
            raise RuntimeError("Unsupported or stale TAO calibration version; recalibrate the model")
        calibration = TaoCalibration(**data)
        reference_path = self.reference_path(config)
        bank = self._reference_bank(config)
        if (calibration.sample_count < 20 or calibration.model_sha256 != _sha256(config.model_file)
                or calibration.reference_sha256 != _sha256(reference_path)
                or calibration.reference_bank_sha256 != bank.manifest_sha256
                or calibration.reference_count != len(bank.images)):
            raise RuntimeError("TAO calibration is stale or insufficient; inspection is inhibited")
        self._geometry_calibration(config, calibration)
        return calibration

    def inspect(self, config: PartModelConfig, image: np.ndarray, *, save_outputs: bool = True, crop_to_component: bool = True) -> InspectionResult:
        total_started = time.perf_counter()
        if crop_to_component:
            image = crop_component_roi(image, roi_ratios=config.roi_ratios)
        calibration = self._calibration(config)
        try:
            bank = self._reference_bank(config)
        except (FileNotFoundError, RuntimeError):
            reference = cv2.imread(str(self.reference_path(config)))
            if reference is None:
                raise RuntimeError("VisualChangeNet golden reference is unreadable; inspection is inhibited")
            raw_map, native_image_score = self._infer(config, reference, image)
            score_map = cv2.resize(raw_map, (config.image_size, config.image_size), interpolation=cv2.INTER_CUBIC)
            surface = cylindrical_surface_mask(score_map.shape)
            defect_mask = clean_mask((score_map >= calibration.pixel_threshold) & surface)
            peak = float(np.max(score_map[surface])) if np.any(surface) else 0.0
            score = max(native_image_score / max(calibration.image_threshold, 1e-12), peak / max(calibration.pixel_threshold, 1e-12))
            area = int(defect_mask.sum())
            bad_ratio, bad = cylindrical_sector_statistics(surface, score_map, config.expected_fins, min_bad_score=calibration.pixel_threshold)
            status = "FAIL" if score >= 1 or area >= max(config.min_defect_area_px, int(score_map.size * .0004)) else "PASS"
            display, boxes = inspection_overlay(image, np.zeros_like(score_map), defect_mask if status == "FAIL" else np.zeros_like(defect_mask), status=status, anomaly_score=score, min_box_area_px=config.min_defect_area_px)
            return InspectionResult(status, score, area, bad_ratio, bad, display_image=display, defect_boxes=boxes, tao_score=score, hybrid_score=score)
        registration_started = time.perf_counter()
        matches = bank.candidates(image, config.tao_reference_candidates)
        registered = [register_to_reference(image, match.image,
            min_correlation=config.registration_min_correlation,
            max_translation_ratio=config.registration_max_translation_ratio,
            max_rotation_deg=config.registration_max_rotation_deg,
            reference_index=match.index) for match in matches]
        valid = [item for item in registered if item.success]
        registration_ms = (time.perf_counter() - registration_started) * 1000
        if not valid:
            display = image.copy()
            return InspectionResult("VIEW INVALID", 0.0, 0, 0.0, [], display_image=display,
                tao_score=0.0, geometry_score=0.0, periodicity_score=0.0, glare_score=0.0,
                registration_score=max((item.correlation for item in registered), default=0.0), hybrid_score=0.0,
                reason_codes=("REGISTRATION_INVALID",), view_valid=False, latencies_ms={"registration": registration_ms, "total": (time.perf_counter()-total_started)*1000})
        registration = max(valid, key=lambda item: item.correlation)
        reference = bank.images[int(registration.reference_index)]
        tao_started = time.perf_counter()
        raw_map, native_image_score = self._infer(config, reference, registration.aligned_image)
        tao_ms = (time.perf_counter() - tao_started) * 1000
        score_map = cv2.resize(raw_map, (config.image_size, config.image_size), interpolation=cv2.INTER_CUBIC)
        surface = cylindrical_surface_mask(score_map.shape) & inspection_band_mask(score_map.shape,
            config.inspection_band_top_ratio, config.inspection_band_bottom_ratio)
        defect_mask = clean_mask((score_map >= calibration.pixel_threshold) & surface)
        defect_area = int(defect_mask.sum())
        bad_ratio, bad_sectors = cylindrical_sector_statistics(surface, score_map, config.expected_fins, min_bad_score=calibration.pixel_threshold)
        fail_area = max(config.min_defect_area_px, int(score_map.size * 0.0004))
        # Present the score relative to the locked calibration limits.  Raw TAO
        # probabilities such as 0.002 look harmless to an operator even when the
        # calibrated normal limit is 0.001; 1.0 is now the unambiguous fail line.
        peak_score = float(np.max(score_map[surface])) if np.any(surface) else 0.0
        anomaly_score = max(
            native_image_score / max(calibration.image_threshold, 1e-12),
            peak_score / max(calibration.pixel_threshold, 1e-12),
        )
        tao_evidence = TaoEvidence(raw_map, score_map / max(calibration.pixel_threshold, 1e-12), anomaly_score,
                                   peak_score, defect_mask, defect_area, bad_regions=bad_sectors)
        geometry_started = time.perf_counter()
        geometry = FinGeometryInspector(self._geometry_calibration(config, calibration),
            band_top=config.inspection_band_top_ratio, band_bottom=config.inspection_band_bottom_ratio).inspect(registration.aligned_image)
        glare = glare_evidence(registration.aligned_image, score_map.shape,
            top=config.inspection_band_top_ratio, bottom=config.inspection_band_bottom_ratio)
        geometry_ms = (time.perf_counter() - geometry_started) * 1000
        decision = fuse_evidence(tao_evidence, geometry, glare_score=glare.score, registration_valid=True,
            geometry_fail_threshold=config.geometry_fail_threshold,
            geometry_candidate_threshold=config.geometry_candidate_threshold,
            tao_strong_threshold=config.tao_strong_threshold, tao_candidate_threshold=config.tao_candidate_threshold,
            glare_threshold=config.glare_threshold)
        status = decision.status
        confirmed = decision.confirmed_mask if status == "FAIL" and decision.confirmed_mask is not None else np.zeros_like(defect_mask)
        # The operator sees original imagery plus red confirmed outlines only.
        display, boxes = inspection_overlay(image, np.zeros_like(score_map), confirmed, status=status,
            anomaly_score=decision.hybrid_score, min_box_area_px=config.min_defect_area_px, score_normalizer=1.0)
        overlay_path = report_path = None
        if save_outputs:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            config.result_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = config.result_dir / f"{stamp}_{status.lower()}_overlay.png"
            report_path = config.result_dir / f"{stamp}_{status.lower()}.json"
            cv2.imwrite(str(overlay_path), display)
            if config.engineering_debug:
                panel_size = (320, 180)
                def panel(value: np.ndarray, label: str) -> np.ndarray:
                    if value.ndim == 2:
                        value = cv2.cvtColor((value.astype(np.uint8) * (255 if value.dtype == bool else 1)), cv2.COLOR_GRAY2BGR)
                    value = cv2.resize(value, panel_size, interpolation=cv2.INTER_NEAREST)
                    cv2.putText(value, label, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
                    return value
                debug = np.vstack((np.hstack((panel(image, "ORIGINAL"), panel(reference, "REFERENCE"), panel(registration.aligned_image, "REGISTERED"))),
                                   np.hstack((panel(defect_mask, "TAO MASK"), panel(geometry.defect_mask, "GEOMETRY MASK"), panel(glare.mask, "GLARE MASK"))),
                                   np.hstack((panel(confirmed, "CONFIRMED"), np.zeros((180, 640, 3), np.uint8)))))
                cv2.imwrite(str(config.result_dir / f"{stamp}_{status.lower()}_engineering.png"), debug)
            report_path.write_text(json.dumps({"algorithm": "nvidia_tao_visual_changenet_hybrid", "model_sha256": calibration.model_sha256,
                "reference_bank_sha256": calibration.reference_bank_sha256, "reference_index": registration.reference_index,
                "status": status, "hybrid_score": decision.hybrid_score, "tao_score": anomaly_score,
                "geometry_score": geometry.score, "periodicity_score": geometry.periodicity_score,
                "glare_score": glare.score, "registration_score": registration.correlation,
                "reason_codes": decision.reason_codes, "view_valid": True, "defect_area_px": defect_area,
                "bad_longitudinal_region_ratio": bad_ratio, "bad_longitudinal_regions": bad_sectors,
                "latencies_ms": {"tao": tao_ms, "registration": registration_ms, "geometry": geometry_ms,
                                 "total": (time.perf_counter()-total_started)*1000}}, indent=2) + "\n", encoding="utf-8")
        return InspectionResult(status, decision.hybrid_score, defect_area, bad_ratio, bad_sectors, overlay_path,
            report_path, display, boxes, anomaly_score, geometry.score, geometry.periodicity_score, glare.score,
            registration.correlation, decision.hybrid_score, decision.reason_codes, registration.reference_index, True,
            {"tao": tao_ms, "registration": registration_ms, "geometry": geometry_ms, "total": (time.perf_counter()-total_started)*1000})
