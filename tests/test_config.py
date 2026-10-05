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
    assert model.camera_width == 3840
    assert model.camera_height == 2160
    assert model.camera_fps == 30
    assert model.image_size == 640
    assert model.vit_auto_download is True
    assert model.surface_require_gpu is True
    assert model.vit_input_size == 518
    assert model.surface_near_candidate_ratio == .85
    assert model.geometry_surface_support_threshold == .55
    assert model.surface_tile_batch_size == 8
    assert model.surface_corroboration_max_area_ratio == .02
    assert model.fine_break_candidate_area_px == 20
    assert model.fine_break_strong_area_px == 60
    assert model.counting_line_ratio == 0.45
    assert model.counting_direction == "left_to_right"
    assert model.tao_change_class_index == 1
    assert model.registration_calibration_min_valid_ratio == 0.70
    assert model.inspection_completion_mode == "counting_line"
    assert model.crop_aspect_ratio_tolerance == 0.35
    assert model.lock_roi_after_confirmation is True
    assert model.yolo_confidence == 0.70
    assert model.counting_axis == "x"
    assert model.minimum_rotation_descriptor_distance == 0.06
    assert model.tao_candidate_min_area_ratio == 0.001
    assert model.tao_candidate_max_area_ratio == 0.12


def test_model_ids_are_case_insensitive_for_cli_convenience(tmp_path):
    registry = ModelRegistry(_registry_file(tmp_path))

    assert registry.get("bf-001").id == "BF-001"


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


def test_model_registry_persists_tao_artifact_path(tmp_path):
    path = _registry_file(tmp_path)
    registry = ModelRegistry(path)

    updated = registry.update_model_settings("BF-001", model_file="models/export.onnx")

    assert str(updated.model_file) == "models/export.onnx"
    assert str(ModelRegistry(path).get("BF-001").model_file) == "models/export.onnx"


def test_supplied_models_are_vit_surface_production_with_legacy_comparison_artifacts():
    models = ModelRegistry("config/models.json").all()

    assert models
    assert all(model.algorithm == "hybrid_patchcore_geometry" for model in models)
    assert all(model.production_algorithm == "vit_surface_geometry" for model in models)
    assert all(model.surface_tile_size == 768 and model.surface_tile_overlap == .25 for model in models)
    assert all(model.vit_auto_download and model.surface_require_gpu for model in models)
    assert all(model.vit_input_size == 518 for model in models)
    assert all(model.surface_near_candidate_ratio == .85 for model in models)
    assert all(model.model_file.suffix == ".pt" for model in models)
    assert all(model.patchcore_model_file == model.model_file for model in models)
    assert all(model.tao_model_file is not None and model.tao_model_file.suffix == ".onnx" for model in models)
    assert all(model.tao_require_gpu is False for model in models)
    assert all(model.inspection_completion_mode == "minimum_views" for model in models)
    assert all(model.minimum_rotation_views == 6 for model in models)
    assert all(model.visible_surface_arc_degrees == 60.0 for model in models)
    assert all(model.minimum_qualified_views == 6 for model in models)
    assert all(model.yolo_presence_confidence == 0.95 for model in models)
    assert all(model.runtime_storage_mode == "memory" for model in models)
    assert all(model.yolo_confidence >= 0.70 for model in models)
    assert all(model.counting_axis == "y" for model in models)
