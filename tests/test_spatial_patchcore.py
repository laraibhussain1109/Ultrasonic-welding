"""Real Torch/backbone training-save-reload/inference tests, without downloads.

Synthetic images and untrained ResNet weights verify software contracts only,
not production accuracy on real GOOD/NG blowers.
"""
from dataclasses import replace

import cv2
import numpy as np
import pytest
import torch
from torchvision import models

from blower_inspection.anomaly_models import _FeatureHook
from blower_inspection.fixed_settings import InspectionSettings
from blower_inspection.patchcore_spatial import SpatialPatchCore
from blower_inspection.stationary_roi import PreparedROI, assess_quality


@pytest.fixture
def trained(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    defaults = InspectionSettings()
    normal = tmp_path / "normal"
    normal.mkdir()
    yolo = tmp_path / "yolo.pt"
    yolo.write_bytes(b"unit-test-detector-identity")
    settings = replace(defaults, yolo=replace(defaults.yolo, model_path=str(yolo)),
        patchcore=replace(defaults.patchcore, backbone="resnet18", input_width=64, input_height=32,
            memory_bank_size=32, normal_image_dir=str(normal), model_path=str(tmp_path / "model.pt")))
    rng = np.random.default_rng(42)
    for index in range(24):
        frame = rng.integers(30, 220, (32,64,3), dtype=np.uint8)
        assert cv2.imwrite(str(normal / f"good_{index}.png"), frame)
    class Detector:
        def __init__(self, settings): self.settings = settings
        def prepare(self, image):
            return PreparedROI(image, image, np.ones(image.shape[:2], bool), (0,0,64,32), .97,
                               assess_quality(image, self.settings.quality))
    monkeypatch.setattr("blower_inspection.patchcore_spatial.YoloROI", Detector)
    model = SpatialPatchCore(settings, device="cpu")
    backbone = models.resnet18(weights=None).eval()
    hook = _FeatureHook(settings.patchcore.embedding_layers)
    hook.attach(backbone)
    model._backbone_cache = (backbone, hook, torch)
    model.train_fixed()
    frame = cv2.imread(str(normal / "good_0.png"))
    roi = Detector(settings).prepare(frame)
    return settings, model, roi


def test_original_patchcore_real_backbone_train_save_reload_and_same_preprocessing(trained):
    settings, trained_model, roi = trained
    map_one, score, _ = trained_model.infer(roi)
    reloaded = SpatialPatchCore(settings, device="cpu")
    reloaded.validate_ready()
    map_two, _, _ = reloaded.infer(roi)
    assert map_one.shape == (32,64)
    assert np.isfinite(map_one).all()
    assert np.allclose(map_one, map_two, atol=1e-6)
    assert score == pytest.approx(map_one.max())
    checkpoint = torch.load(settings.patchcore.model_path, weights_only=True)
    assert set(checkpoint["training_manifest"]).isdisjoint(checkpoint["calibration_manifest"])
    assert len(checkpoint["memory_bank"]) <= settings.patchcore.memory_bank_size
    assert "padim_mean" not in checkpoint and "geometry_calibrations" not in checkpoint


def test_preprocessing_change_requires_retraining(trained):
    settings, _, roi = trained
    changed = replace(settings, patchcore=replace(settings.patchcore, input_height=40))
    with pytest.raises(ValueError, match="preprocessing"):
        SpatialPatchCore(changed)._load()


def test_corrupt_memory_bank_is_rejected(trained):
    settings, _, _ = trained
    checkpoint = torch.load(settings.patchcore.model_path, weights_only=True)
    checkpoint["memory_bank"][0,0] += .1
    torch.save(checkpoint, settings.patchcore.model_path)
    with pytest.raises(ValueError, match="digest"):
        SpatialPatchCore(settings)._load()


def test_legacy_model_is_preserved_and_rejected_in_new_mode(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"algorithm":"patchcore_primary", "version":3}, path)
    settings = InspectionSettings(patchcore=replace(InspectionSettings().patchcore, model_path=str(path)))
    with pytest.raises(ValueError, match="legacy"):
        SpatialPatchCore(settings)._load()
    assert torch.load(path, weights_only=True)["algorithm"] == "patchcore_primary"


def test_angle_specific_paths_do_not_silently_fall_back_to_shared_model():
    settings = replace(InspectionSettings().patchcore, angle_specific=True,
                       angle_model_paths={str(a):f"model_{a}.pt" for a in range(60,361,60)})
    assert settings.path_for_angle(300).name == "model_300.pt"
    with pytest.raises(ValueError, match="angle-specific"):
        settings.path_for_angle(0)
