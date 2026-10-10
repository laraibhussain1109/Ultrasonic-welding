"""Exercise actual CLI training/import, fresh desktop loading, and still inference."""

import json
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from torchvision import models

from blower_inspection.anomaly_models import HybridPatchcorePadimInspector, _FeatureHook
from blower_inspection.cli import main
from blower_inspection.config import ModelRegistry
from blower_inspection.fixed_settings import InspectionSettings
from blower_inspection.inspector_factory import inspector_for_model
from blower_inspection.patchcore_spatial import SpatialPatchCore
from blower_inspection.production_training import import_spatial
from blower_inspection.stationary_roi import YoloROI


@pytest.fixture
def case(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    normal = tmp_path / "GOOD"
    normal.mkdir()
    yolo = tmp_path / "best.pt"
    yolo.write_bytes(b"test-yolo-identity")
    rng = np.random.default_rng(42)
    for index in range(24):
        image = rng.integers(30, 220, (96, 192, 3), dtype=np.uint8)
        assert cv2.imwrite(str(normal / f"good_{index:03d}.png"), image)
    defaults = InspectionSettings()
    settings = replace(defaults, quality=replace(defaults.quality, minimum_blur_score=0),
        yolo=replace(defaults.yolo, model_path=str(yolo)),
        patchcore=replace(defaults.patchcore, backbone="resnet18", input_width=64, input_height=32,
                          memory_bank_size=8, normal_image_dir=str(normal), model_path=str(tmp_path / "patchcore_fixed.pt")))
    settings_path = tmp_path / "fixed.json"
    settings.save(settings_path)
    registry_path = tmp_path / "models.json"
    primary = tmp_path / "production" / "patchcore_primary.pt"
    calibration = primary.with_name("patchcore_calibration.json")
    registry_path.write_text(json.dumps({"active_model": "BF-001", "models": [{
        "id": "BF-001", "name": "blower", "expected_fins": 24,
        "model_file": str(primary), "patchcore_model_file": str(primary),
        "patchcore_calibration_file": str(calibration), "normal_image_dir": str(tmp_path / "not-configured"),
        "result_dir": str(tmp_path / "results"), "image_size": 640, "patchcore_section_count": 2,
    }]}))

    class Boxes:
        def __init__(self, frame):
            h, w = frame.shape[:2]
            self.conf = torch.tensor([.97])
            self.xyxy = torch.tensor([[8, 8, w - 8, h - 8]], dtype=torch.float32)
        def __len__(self): return 1

    detector = SimpleNamespace(predict=lambda frame, **_kwargs: [SimpleNamespace(boxes=Boxes(frame))])
    monkeypatch.setattr("blower_inspection.yolo_tracking.YoloByteTrackDetector._load", lambda _self: detector)

    def backbone(self):
        if self._backbone_cache is None:
            network = models.resnet18(weights=None).eval()
            hook = _FeatureHook(self.settings.embedding_layers)
            hook.attach(network)
            self._backbone_cache = (network, hook, torch)
        return self._backbone_cache

    monkeypatch.setattr(HybridPatchcorePadimInspector, "_build_backbone", backbone)
    return SimpleNamespace(settings=settings, settings_path=settings_path, registry_path=registry_path,
                           registry=ModelRegistry(registry_path), primary=primary, calibration=calibration,
                           normal=normal, yolo=yolo)


@pytest.fixture
def spatial(case):
    model = SpatialPatchCore(case.settings, device="cpu")
    model.train_fixed()
    return case, model


def test_default_train_fixed_creates_desktop_bundle_and_reloadable_frozen_features(case):
    assert main(["train-fixed", "--settings", str(case.settings_path), "--models", str(case.registry_path)]) == 0
    assert case.primary.is_file() and case.calibration.is_file()
    assert not case.settings.patchcore.path_for_angle(60).exists()
    registry = ModelRegistry(case.registry_path)
    config = registry.get("BF-001")
    assert config.normal_image_dir == case.normal
    assert config.yolo_model_path == case.yolo
    assert config.patchcore_model_file == case.primary
    checkpoint = torch.load(case.primary, weights_only=True)
    assert checkpoint["algorithm"] == "patchcore_primary" and checkpoint["version"] == 3
    assert checkpoint["backbone_state"]
    assert set(checkpoint["training_manifest"]).isdisjoint(checkpoint["calibration_manifest"])
    calibration = json.loads(case.calibration.read_text())
    assert calibration["memory_bank_hash"] == checkpoint["memory_bank_hash"]
    backend = inspector_for_model(config)
    backend.validate_ready(config)
    image = cv2.imread(str(case.normal / "good_000.png"))
    result = backend.inspect(config, image, save_outputs=False)
    assert result.view_valid and result.raw_heatmap.shape == (32, 64)
    # A second desktop instance restores exactly the same saved weights/settings.
    reloaded = inspector_for_model(config)
    reloaded.validate_ready(config)
    second = reloaded.inspect(config, image, save_outputs=False)
    assert np.allclose(result.raw_heatmap, second.raw_heatmap, atol=1e-6)


def test_import_fixed_reuses_bank_and_backbone_and_drives_actual_desktop_workers(spatial):
    from blower_inspection.app import InspectionWorker
    case, source = spatial
    source_path = case.settings.patchcore.path_for_angle(60)
    before = source_path.read_bytes()
    original = torch.load(source_path, weights_only=True)
    assert main(["import-fixed", "BF-001", "--settings", str(case.settings_path),
                 "--models", str(case.registry_path)]) == 0
    assert source_path.read_bytes() == before
    checkpoint = torch.load(case.primary, weights_only=True)
    assert torch.equal(checkpoint["memory_bank"], original["memory_bank"])
    for key, weights in original["backbone_state"].items():
        assert torch.equal(checkpoint["backbone_state"][key], weights)
    config = ModelRegistry(case.registry_path).get("BF-001")
    backend = inspector_for_model(config)
    backend.validate_ready(config)
    image = cv2.imread(str(case.normal / "good_000.png"))
    roi = YoloROI(case.settings).prepare(image)
    expected, _, _ = source.infer(roi)
    imported = backend._raw_map(roi.image, backend._load_runtime_checkpoint(torch, case.primary), torch)
    assert np.allclose(imported, expected, atol=1e-6)
    detection = backend.detect_inspection_roi(image, config, None)
    assert detection.bounds == roi.bounds
    x, y, width, height = detection.bounds
    crop = image[y:y + height, x:x + width]
    results, errors = [], []
    for angle in range(60, 361, 60):
        worker = InspectionWorker(backend, config, 1, crop, angle)
        worker.finished_result.connect(lambda track, result, elapsed: results.append(result))
        worker.failed.connect(errors.append)
        worker.run()
    assert not errors
    assert [result.view_angle for result in results] == list(range(60, 361, 60))
    assert all(result.view_valid for result in results)
    assert all(np.allclose(result.raw_heatmap, expected, atol=1e-6) for result in results)


def test_import_uses_full_calibrated_bank_even_if_runtime_limit_changes(spatial):
    case, _source = spatial
    import_spatial(case.registry, case.settings)
    config = replace(ModelRegistry(case.registry_path).active(), patchcore_memory_bank_size=1)
    backend = inspector_for_model(config)
    backend.validate_ready(config)
    assert len(backend._load_runtime_checkpoint(torch, case.primary)["memory_bank"]) == 8


@pytest.mark.parametrize("failure", ["missing_good", "calibration_error", "weights_changed", "overlapping_manifests"])
def test_failed_import_preserves_existing_assets_and_registry(spatial, monkeypatch, failure):
    from blower_inspection.patchcore_inspector import PatchCoreInspector
    case, _source = spatial
    case.primary.parent.mkdir()
    case.primary.write_bytes(b"previous-primary")
    case.calibration.write_bytes(b"previous-calibration")
    before = case.registry_path.read_bytes()
    if failure == "missing_good":
        next(case.normal.iterdir()).unlink()
    elif failure == "calibration_error":
        monkeypatch.setattr(PatchCoreInspector, "_calibration", lambda *_args: (_ for _ in ()).throw(RuntimeError("calibration failed")))
    elif failure == "weights_changed":
        case.yolo.write_bytes(b"different-detector")
    else:
        path = case.settings.patchcore.path_for_angle(60)
        checkpoint = torch.load(path, weights_only=True)
        checkpoint["calibration_manifest"][0] = checkpoint["training_manifest"][0]
        torch.save(checkpoint, path)
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        import_spatial(case.registry, case.settings)
    assert case.primary.read_bytes() == b"previous-primary"
    assert case.calibration.read_bytes() == b"previous-calibration"
    assert case.registry_path.read_bytes() == before


def test_explicit_spatial_only_keeps_engineering_checkpoint_separate(case):
    assert main(["train-fixed", "--spatial-only", "--settings", str(case.settings_path)]) == 0
    assert case.settings.patchcore.path_for_angle(60).exists()
    assert not case.primary.exists() and not case.calibration.exists()


def test_angle_requires_explicit_engineering_mode(case):
    with pytest.raises(SystemExit):
        main(["train-fixed", "--angle", "60", "--settings", str(case.settings_path)])
    assert not case.primary.exists()


def test_original_desktop_startup_accepts_imported_artifacts_before_opening_camera(spatial, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from blower_inspection import app as ui
    from blower_inspection.auth import User
    case, _source = spatial
    import_spatial(case.registry, case.settings)
    application = QApplication.instance() or QApplication([])
    registry = ModelRegistry(case.registry_path)
    monkeypatch.setattr(ui, "ModelRegistry", lambda: registry)
    bridge = SimpleNamespace(config=SimpleNamespace(enabled=False), reset=lambda: None, close=lambda: None)
    monkeypatch.setattr(ui, "ESP32FailOutputBridge", lambda: bridge)
    window = ui.InspectionWindow(User("admin", "admin"))
    opened, errors = [], []
    monkeypatch.setattr(ui.USBCamera, "open", lambda _camera: opened.append(True))
    def camera_frame(_camera):
        import time
        time.sleep(.01)
        return np.zeros((80, 240, 3), dtype=np.uint8)
    monkeypatch.setattr(ui.USBCamera, "read", camera_frame)
    monkeypatch.setattr(window, "_confirm_and_lock_roi", lambda _model: False)
    monkeypatch.setattr(ui.QMessageBox, "critical", lambda *_args: errors.append(_args[2]))
    try:
        window.start_inspection()
        assert opened == [True] and not errors
    finally:
        window.close()
        application.processEvents()


def test_renamed_spatial_checkpoint_has_actionable_import_error(spatial):
    case, _source = spatial
    case.primary.parent.mkdir()
    case.primary.write_bytes(case.settings.patchcore.path_for_angle(60).read_bytes())
    case.calibration.write_text("{}")
    with pytest.raises(ValueError, match="import-fixed"):
        inspector_for_model(case.registry.active()).validate_ready(case.registry.active())
