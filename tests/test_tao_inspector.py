import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2", exc_type=ImportError)

from blower_inspection.config import PartModelConfig
from blower_inspection.inspector_factory import inspector_for_model
from blower_inspection.tao_inspector import TaoCalibration, TaoInspector


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


def test_visual_changenet_prefers_fused_final_map_over_decoder_outputs():
    values = {
        "output0": np.zeros((1, 2, 8, 8), np.float32),
        "output1": np.zeros((1, 2, 16, 16), np.float32),
        "output2": np.zeros((1, 2, 32, 32), np.float32),
        "output3": np.zeros((1, 2, 64, 64), np.float32),
        "output_final": np.zeros((1, 2, 64, 64), np.float32),
    }

    assert TaoInspector._discover_map_output(values) == "output_final"


def test_visual_changenet_ambiguous_outputs_still_require_explicit_binding():
    values = {
        "decoder_a": np.zeros((1, 2, 8, 8), np.float32),
        "decoder_b": np.zeros((1, 2, 8, 8), np.float32),
    }

    with pytest.raises(RuntimeError, match="tao_output_name"):
        TaoInspector._discover_map_output(values)


def test_validate_ready_checks_calibration_and_production_session(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    inspector = TaoInspector()
    calls = []

    monkeypatch.setattr(inspector, "_calibration", lambda runtime_config: calls.append("calibration"))
    monkeypatch.setattr(
        inspector,
        "_session",
        lambda runtime_config: calls.append("gpu-session"),
    )

    inspector.validate_ready(cfg)

    assert calls == ["calibration", "gpu-session"]


def test_runtime_device_name_reports_active_provider():
    inspector = TaoInspector()

    class Session:
        @staticmethod
        def get_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    inspector._last_session = Session()
    assert inspector.runtime_device_name() == "CUDAExecutionProvider"


def test_calibration_requests_gpu_first_with_cpu_fallback(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg.normal_image_dir.mkdir()
    image = np.zeros((16, 16, 3), np.uint8)
    import cv2

    for index in range(20):
        assert cv2.imwrite(str(cfg.normal_image_dir / f"normal-{index}.png"), image)

    fallback_requests = []

    def infer(runtime_config, reference, candidate, *, allow_cpu_fallback=False):
        fallback_requests.append(allow_cpu_fallback)
        return np.zeros((4, 4), np.float32), 0.0

    inspector = TaoInspector()
    monkeypatch.setattr(inspector, "_infer", infer)
    inspector.train(cfg)

    assert fallback_requests == [True] * 20
    assert cfg.tao_require_gpu is True


def test_cpu_calibration_session_cannot_be_reused_as_gpu_session(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    inspector = TaoInspector()
    cpu_session = object()
    inspector._sessions[(cfg.model_file.resolve(), True, True)] = (
        cfg.model_file.stat().st_mtime_ns,
        cpu_session,
    )

    class FakeOrt:
        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", FakeOrt)
    with pytest.raises(RuntimeError, match="GPU provider unavailable"):
        inspector._session(cfg)


def test_calibration_provider_order_prefers_gpu_then_cpu(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    selected = []

    class Input:
        name = "input"

    class Session:
        def __init__(self, path, providers):
            selected.extend(providers)

        def get_providers(self):
            return selected

        def get_inputs(self):
            return [Input(), Input()]

    class FakeOrt:
        InferenceSession = Session

        @staticmethod
        def get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", FakeOrt)
    TaoInspector()._session(cfg, allow_cpu_fallback=True)

    assert selected == ["CUDAExecutionProvider", "CPUExecutionProvider"]
