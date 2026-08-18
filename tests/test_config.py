import json
from pathlib import Path

from blower_inspection.config import ModelRegistry


def _registry_file(tmp_path: Path) -> Path:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "active_model": "BF-001",
                "models": [
                    {
                        "id": "BF-001",
                        "name": "Blower Fan Model 1",
                        "normal_image_dir": "data/training/BF-001/normal",
                        "model_file": "data/models/BF-001/model.pt",
                        "result_dir": "data/results/BF-001",
                        "expected_fins": 24,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_model_registry_defaults_camera_and_allows_missing_roi(tmp_path):
    registry = ModelRegistry(_registry_file(tmp_path))

    model = registry.get("BF-001")

    assert model.roi_ratios is None
    assert model.camera_width == 1920
    assert model.camera_height == 1080
    assert model.camera_fps == 30
    assert model.image_size == 640
    assert model.max_defect_area_ratio == 0.05
    assert model.scoring_end_exclusion_ratio == 0.12
    assert model.scoring_mask_erosion_px == 8
    assert model.counting_line_ratio == 0.45
    assert model.counting_direction == "left_to_right"


def test_model_registry_persists_roi_and_camera_settings(tmp_path):
    path = _registry_file(tmp_path)
    registry = ModelRegistry(path)

    updated = registry.update_model_settings(
        "BF-001",
        roi_ratios=(0.1, 0.2, 0.7, 0.3),
        camera_width=3840,
        camera_height=2160,
        camera_fps=30,
    )

    assert updated.roi_ratios == (0.1, 0.2, 0.7, 0.3)
    assert updated.camera_width == 3840
    assert updated.camera_height == 2160
    assert updated.camera_fps == 30
    reloaded = ModelRegistry(path).get("BF-001")
    assert reloaded.roi_ratios == (0.1, 0.2, 0.7, 0.3)
    assert reloaded.camera_width == 3840


def test_model_registry_persists_yolo_detector_path(tmp_path):
    path = _registry_file(tmp_path)
    registry = ModelRegistry(path)

    updated = registry.update_model_settings("BF-001", yolo_model_path="models/best.pt")

    assert str(updated.yolo_model_path) == "models/best.pt"
    assert str(ModelRegistry(path).get("BF-001").yolo_model_path) == "models/best.pt"
