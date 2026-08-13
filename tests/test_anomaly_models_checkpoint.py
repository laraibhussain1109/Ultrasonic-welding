import pickle
from pathlib import Path

import pytest

pytest.importorskip("cv2")
pytest.importorskip("numpy")

from blower_inspection.anomaly_models import HybridPatchcorePadimInspector
from blower_inspection.config import PartModelConfig


def _config(tmp_path: Path) -> PartModelConfig:
    return PartModelConfig(
        id="BF-T",
        name="Test Model",
        normal_image_dir=tmp_path / "normal",
        model_file=tmp_path / "model.pt",
        result_dir=tmp_path / "results",
        expected_fins=24,
    )


def test_serializable_model_config_stores_paths_as_strings(tmp_path):
    data = HybridPatchcorePadimInspector._serializable_model_config(_config(tmp_path))

    assert data["normal_image_dir"] == str(tmp_path / "normal")
    assert data["model_file"] == str(tmp_path / "model.pt")
    assert data["result_dir"] == str(tmp_path / "results")


def test_load_hybrid_checkpoint_falls_back_for_legacy_path_pickles(tmp_path):
    class FakeTorch:
        def __init__(self):
            self.calls = []

        def load(self, *args, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("weights_only") is True:
                raise pickle.UnpicklingError("Weights only load failed. Unsupported global: pathlib.WindowsPath")
            return {"algorithm": "hybrid_patchcore_padim"}

    torch = FakeTorch()
    checkpoint = HybridPatchcorePadimInspector._load_hybrid_checkpoint(torch, tmp_path / "legacy.pt")

    assert checkpoint == {"algorithm": "hybrid_patchcore_padim"}
    assert torch.calls == [
        {"map_location": "cpu", "weights_only": True},
        {"map_location": "cpu", "weights_only": False},
    ]


def test_apply_model_settings_uses_model_image_size(tmp_path):
    config = _config(tmp_path)
    config = PartModelConfig(**{**config.__dict__, "image_size": 256})
    inspector = HybridPatchcorePadimInspector()

    inspector._apply_model_settings(config)

    assert inspector.settings.image_size == 256


def test_apply_checkpoint_settings_restores_training_image_size(tmp_path):
    inspector = HybridPatchcorePadimInspector()

    inspector._apply_checkpoint_settings({"settings": {"image_size": 384}})

    assert inspector.settings.image_size == 384


def test_default_hybrid_profile_preserves_small_defect_detail():
    inspector = HybridPatchcorePadimInspector()

    assert inspector.settings.image_size == 640
    assert inspector.settings.embedding_grid_size == 80
    assert inspector.settings.max_coreset_patches == 8192
    assert inspector.settings.runtime_memory_bank_limit == 1024
    assert inspector.settings.distillation_epochs == 8
    assert inspector.settings.distillation_weight == 0.65


def test_training_requires_yolo_for_automatic_exact_crops(tmp_path):
    config = _config(tmp_path)
    config.normal_image_dir.mkdir()
    for index in range(20):
        (config.normal_image_dir / f"normal_{index}.png").touch()

    with pytest.raises(ValueError, match="Select best.pt before training"):
        HybridPatchcorePadimInspector().train(config)
