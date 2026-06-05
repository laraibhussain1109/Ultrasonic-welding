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

    @staticmethod
    def _parse(entry: dict) -> PartModelConfig:
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
        )


def ensure_model_folders(registry: ModelRegistry) -> None:
    for model in registry.all():
        model.normal_image_dir.mkdir(parents=True, exist_ok=True)
        model.model_file.parent.mkdir(parents=True, exist_ok=True)
        model.result_dir.mkdir(parents=True, exist_ok=True)
