"""Exercise the production inspector, using controlled raw distances, not weights."""
import json
from dataclasses import replace

import numpy as np
import pytest

from blower_inspection.config import PartModelConfig
from blower_inspection.geometry_inspector import GeometryEvidence
from blower_inspection.patchcore_inspector import PatchCoreInspector


@pytest.fixture
def backend(monkeypatch, tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"memory_bank_hash": "same-bank", "embedding_layers": ["layer2", "layer3"],
                                       "thresholds": {"candidate": .7, "fail": 1.5, "patch_p999": 1.0}}))
    config = PartModelConfig("BF-001", "blower", tmp_path, tmp_path / "model.pt", tmp_path, 24,
                             image_size=240, min_defect_area_px=8, inference_min_sharpness=0,
                             patchcore_calibration_file=calibration, support_rib_mask_enabled=False,
                             patchcore_edge_ignore_ratio=0, roi_padding_ratio=0)
    inspector = PatchCoreInspector()
    checkpoint = {"algorithm": "patchcore_primary", "version": 3, "memory_bank_hash": "same-bank",
                  "geometry_calibrations": [{}] * 6}
    monkeypatch.setattr(inspector, "_load_runtime_checkpoint", lambda *_args: checkpoint)
    def raw_map(image, *_args):
        raw = np.full(image.shape[:2], .1, np.float32)
        raw[45:65, 70:90] = 3
        return raw
    monkeypatch.setattr(inspector, "_raw_map", raw_map)
    def geometry(image, *_args):
        mask = np.zeros(image.shape[:2], bool)
        mask[45:65, 70:90] = True
        return GeometryEvidence(1.2, 0, 0, 1.2, 1.2, 0, .8, mask)
    monkeypatch.setattr(inspector, "_inspect_geometry", geometry)
    monkeypatch.setattr("blower_inspection.patchcore_inspector.support_rib_mask",
                        lambda image: (np.zeros(image.shape[:2], bool), 0))
    image = np.random.default_rng(12).integers(40, 180, (80, 240, 3), dtype=np.uint8)
    return inspector, config, image


def test_production_inspector_applies_tolerance_and_retains_heatmap(backend):
    inspector, config, image = backend
    permissive = inspector.inspect(replace(config, heatmap_tolerance_percent=5), image, crop_to_component=False)
    strict = inspector.inspect(replace(config, heatmap_tolerance_percent=1), image, crop_to_component=False)
    assert permissive.status == "PASS" and strict.status == "FAIL"
    assert strict.defect_area_px == 400
    assert strict.valid_area_px < strict.raw_heatmap.size  # letterbox and borders excluded
    assert strict.anomaly_percentage == pytest.approx(100 * 400 / strict.valid_area_px)
    assert strict.raw_heatmap[50, 75] == 3 and strict.filtered_anomaly_mask[50, 75]
    assert strict.candidate_sections == (1, 2)
    assert permissive.candidate_sections == ()


def test_model_calibration_mismatch_is_still_rejected(backend):
    inspector, config, image = backend
    calibration = json.loads(config.patchcore_calibration_file.read_text())
    calibration["memory_bank_hash"] = "different-bank"
    config.patchcore_calibration_file.write_text(json.dumps(calibration))
    with pytest.raises(RuntimeError, match="Stale PatchCore calibration"):
        inspector.inspect(config, image, crop_to_component=False)


def test_readiness_reports_exact_missing_checkpoint_before_capture(backend):
    inspector, config, _image = backend
    with pytest.raises(FileNotFoundError, match="PatchCore checkpoint not found") as error:
        inspector.validate_ready(config)
    assert str(config.model_file.resolve()) in str(error.value)


def test_readiness_reports_missing_calibration_before_loading_model(backend):
    inspector, config, _image = backend
    config.model_file.touch()
    config.patchcore_calibration_file.unlink()
    with pytest.raises(FileNotFoundError, match="PatchCore calibration not found") as error:
        inspector.validate_ready(config)
    assert str(config.patchcore_calibration_file.resolve()) in str(error.value)


def test_readiness_validates_existing_checkpoint_and_calibration(backend):
    inspector, config, _image = backend
    config.model_file.touch()
    inspector.validate_ready(config)
    calibration = json.loads(config.patchcore_calibration_file.read_text())
    calibration["thresholds"]["patch_p999"] = float("nan")
    config.patchcore_calibration_file.write_text(json.dumps(calibration))
    with pytest.raises(ValueError, match="patch_p999 threshold"):
        inspector.validate_ready(config)
