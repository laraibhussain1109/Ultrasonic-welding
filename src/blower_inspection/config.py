"""Configuration models and loading utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PartModelConfig:
    id: str
    name: str
    normal_image_dir: Path
    model_file: Path
    result_dir: Path
    expected_fins: int
    outer_radius_ratio: float = 0.95
    inner_radius_ratio: float = 0.22
    anomaly_threshold: float = 4.0
    min_defect_area_px: int = 120
    max_bad_sector_ratio: float = 0.18
    roi_ratios: tuple[float, float, float, float] | None = None
    camera_width: int = 1920
    camera_height: int = 1080
    camera_fps: int = 30
    image_size: int = 640
    yolo_model_path: Path | None = None
    yolo_confidence: float = 0.40
    inspection_lost_timeout_s: float = 1.0
    # Slightly left of frame center so the part reaches the count line within
    # the usable fixture/conveyor travel visible in the production camera.
    counting_line_ratio: float = 0.45
    counting_direction: str = "left_to_right"
    algorithm: str = "nvidia_tao"
    # TAO Deploy exports an ONNX model.  Calibration is deliberately stored
    # separately so replacing an engine cannot silently retain stale limits.
    tao_calibration_file: Path | None = None
    tao_input_name: str | None = None
    tao_reference_input_name: str | None = None
    tao_test_input_name: str | None = None
    tao_output_name: str | None = None
    tao_score_output_name: str | None = None
    tao_require_gpu: bool = True
    tao_reference_image: Path | None = None
    tao_change_class_index: int = 1


class ModelRegistry:
    def __init__(self, path: str | Path = "config/models.json") -> None:
        self.path = Path(path)
        self._data = self._load()

    def _load(self) -> dict:
        with self.path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def all(self) -> list[PartModelConfig]:
        return [self._parse(entry) for entry in self._data.get("models", [])]

    def get(self, model_id: str) -> PartModelConfig:
        for model in self.all():
            if model.id == model_id:
                return model
        known = ", ".join(model.id for model in self.all())
        raise KeyError(f"Unknown model '{model_id}'. Known models: {known}")

    def active(self) -> PartModelConfig:
        return self.get(self._data.get("active_model", self.all()[0].id))

    def update_model_settings(
        self,
        model_id: str,
        *,
        roi_ratios: tuple[float, float, float, float] | None = None,
        camera_width: int | None = None,
        camera_height: int | None = None,
        camera_fps: int | None = None,
        yolo_model_path: str | Path | None = None,
        model_file: str | Path | None = None,
    ) -> PartModelConfig:
        for entry in self._data.get("models", []):
            if entry.get("id") != model_id:
                continue
            if roi_ratios is not None:
                entry["roi_ratios"] = [round(float(value), 6) for value in roi_ratios]
            if camera_width is not None:
                entry["camera_width"] = int(camera_width)
            if camera_height is not None:
                entry["camera_height"] = int(camera_height)
            if camera_fps is not None:
                entry["camera_fps"] = int(camera_fps)
            if yolo_model_path is not None:
                entry["yolo_model_path"] = str(yolo_model_path)
            if model_file is not None:
                entry["model_file"] = str(model_file)
            self._save()
            self._data = self._load()
            return self.get(model_id)
        known = ", ".join(model.id for model in self.all())
        raise KeyError(f"Unknown model '{model_id}'. Known models: {known}")

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(self._data, handle, indent=2)
            handle.write("\n")
        temp_path.replace(self.path)

    @staticmethod
    def _parse(entry: dict) -> PartModelConfig:
        roi = entry.get("roi_ratios")
        return PartModelConfig(
            id=entry["id"],
            name=entry["name"],
            normal_image_dir=Path(entry["normal_image_dir"]),
            model_file=Path(entry["model_file"]),
            result_dir=Path(entry["result_dir"]),
            expected_fins=int(entry["expected_fins"]),
            outer_radius_ratio=float(entry.get("outer_radius_ratio", 0.95)),
            inner_radius_ratio=float(entry.get("inner_radius_ratio", 0.22)),
            anomaly_threshold=float(entry.get("anomaly_threshold", 4.0)),
            min_defect_area_px=int(entry.get("min_defect_area_px", 120)),
            max_bad_sector_ratio=float(entry.get("max_bad_sector_ratio", 0.18)),
            roi_ratios=tuple(float(value) for value in roi) if roi is not None else None,
            camera_width=int(entry.get("camera_width", 1920)),
            camera_height=int(entry.get("camera_height", 1080)),
            camera_fps=int(entry.get("camera_fps", 30)),
            image_size=int(entry.get("image_size", 640)),
            yolo_model_path=Path(entry["yolo_model_path"]) if entry.get("yolo_model_path") else None,
            yolo_confidence=float(entry.get("yolo_confidence", 0.40)),
            inspection_lost_timeout_s=float(entry.get("inspection_lost_timeout_s", 1.0)),
            counting_line_ratio=float(entry.get("counting_line_ratio", 0.45)),
            counting_direction=str(entry.get("counting_direction", "left_to_right")),
            algorithm=str(entry.get("algorithm", "nvidia_tao")),
            tao_calibration_file=Path(entry["tao_calibration_file"]) if entry.get("tao_calibration_file") else None,
            tao_input_name=entry.get("tao_input_name"),
            tao_reference_input_name=entry.get("tao_reference_input_name"),
            tao_test_input_name=entry.get("tao_test_input_name"),
            tao_output_name=entry.get("tao_output_name"),
            tao_score_output_name=entry.get("tao_score_output_name"),
            tao_require_gpu=bool(entry.get("tao_require_gpu", True)),
            tao_reference_image=Path(entry["tao_reference_image"]) if entry.get("tao_reference_image") else None,
            tao_change_class_index=max(0, int(entry.get("tao_change_class_index", 1))),
        )


def ensure_model_folders(registry: ModelRegistry) -> None:
    for model in registry.all():
        model.normal_image_dir.mkdir(parents=True, exist_ok=True)
        model.model_file.parent.mkdir(parents=True, exist_ok=True)
        model.result_dir.mkdir(parents=True, exist_ok=True)
