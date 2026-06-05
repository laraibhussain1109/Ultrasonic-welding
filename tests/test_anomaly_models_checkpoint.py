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
