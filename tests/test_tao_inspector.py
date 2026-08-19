import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2", exc_type=ImportError)

from blower_inspection.config import PartModelConfig
from blower_inspection.tao_inspector import TaoCalibration, TaoInspector, inspector_for_model


def config(tmp_path: Path) -> PartModelConfig:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"tao-export")
    return PartModelConfig("T", "TAO", tmp_path / "normal", model, tmp_path / "results", 24)


def test_factory_selects_tao_by_default(tmp_path):
    assert isinstance(inspector_for_model(config(tmp_path)), TaoInspector)


def test_calibration_rejects_changed_model(tmp_path):
    cfg = config(tmp_path)
    path = TaoInspector.calibration_path(cfg)
    reference = TaoInspector.reference_path(cfg)
    reference.write_bytes(b"reference")
    path.write_text(json.dumps({"version": 2, **TaoCalibration("wrong", "wrong", 1, 1, .5, .5, 20, "now").__dict__}))
    with pytest.raises(RuntimeError, match="stale"):
        TaoInspector()._calibration(cfg)


def test_preprocess_is_fixed_shape_nchw():
    image = np.zeros((30, 40, 3), np.uint8)
    tensor = TaoInspector._input_tensor(image, [1, 3, 64, 96])
    assert tensor.shape == (1, 3, 64, 96)
    assert tensor.dtype == np.float32


def test_dynamic_spatial_export_is_rejected():
    with pytest.raises((TypeError, RuntimeError)):
        TaoInspector._input_tensor(np.zeros((10, 10, 3), np.uint8), [1, 3, "height", "width"])


def test_visual_changenet_logits_become_change_probability_map():
    logits = np.zeros((1, 2, 4, 5), np.float32)
    logits[:, 1, 1:3, 2:4] = 5.0
    change = TaoInspector._change_map(logits, 1)
    assert change.shape == (4, 5)
    assert np.all(change[1:3, 2:4] > 0.98)
