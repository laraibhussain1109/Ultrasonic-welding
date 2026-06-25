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
        )


def ensure_model_folders(registry: ModelRegistry) -> None:
    for model in registry.all():
        model.normal_image_dir.mkdir(parents=True, exist_ok=True)
        model.model_file.parent.mkdir(parents=True, exist_ok=True)
        model.result_dir.mkdir(parents=True, exist_ok=True)
