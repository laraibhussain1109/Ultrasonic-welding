import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2", exc_type=ImportError)

from blower_inspection.config import PartModelConfig
from blower_inspection.inspector_factory import inspector_for_model
from blower_inspection.registration import RegistrationResult
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
    path.write_text(json.dumps({"version": 3, **TaoCalibration("wrong", "wrong", 1, 1, .5, .5, 20, "now").__dict__}))
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


def test_calibration_excludes_an_unregistrable_normal_instead_of_aborting(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg.normal_image_dir.mkdir()
    import cv2

    for index in range(21):
        image = np.zeros((32, 48, 3), np.uint8)
        cv2.line(image, (0, 8 + index % 8), (47, 8 + index % 8), (180, 180, 180), 2)
        assert cv2.imwrite(str(cfg.normal_image_dir / f"normal-{index:02d}.png"), image)

    first_candidate = None

    def registration(candidate, reference, **kwargs):
        nonlocal first_candidate
        if first_candidate is None:
            first_candidate = candidate
        # Reject every bank candidate for the first calibration image only.
        if candidate is first_candidate:
            return RegistrationResult(candidate, False, 0.1, 0, 0, 0, kwargs["reference_index"])
        return RegistrationResult(candidate, True, 0.9, 0, 0, 0, kwargs["reference_index"])

    monkeypatch.setattr("blower_inspection.tao_inspector.register_to_reference", registration)
    inspector = TaoInspector()
    monkeypatch.setattr(inspector, "_infer", lambda *args, **kwargs: (np.zeros((4, 4), np.float32), 0.0))
    output = inspector.train(cfg)
    data = json.loads(output.read_text())
    assert data["sample_count"] == 20
    assert data["calibration_image_count"] == 21
    assert data["excluded_registration_count"] == 1


def test_calibration_valid_ratio_is_configurable(tmp_path):
    from dataclasses import replace

    cfg = replace(config(tmp_path), registration_calibration_min_valid_ratio=0.70)
    assert cfg.registration_calibration_min_valid_ratio == pytest.approx(0.70)


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


def test_cpu_fallback_is_used_when_permitted_and_cuda_is_unavailable(tmp_path, monkeypatch):
    from dataclasses import replace

    cfg = replace(config(tmp_path), tao_require_gpu=False)
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
            return ["CPUExecutionProvider"]

    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", FakeOrt)
    TaoInspector()._session(cfg, allow_cpu_fallback=True)

    assert selected == ["CPUExecutionProvider"]


def test_required_gpu_validates_active_session_provider(tmp_path, monkeypatch):
    cfg = config(tmp_path)

    class Input:
        name = "input"

    class Session:
        def __init__(self, path, providers):
            assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]

        @staticmethod
        def get_providers():
            return ["CPUExecutionProvider"]

        @staticmethod
        def get_inputs():
            return [Input(), Input()]

    class FakeOrt:
        InferenceSession = Session

        @staticmethod
        def get_available_providers():
            return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", FakeOrt)
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider is not active"):
        TaoInspector()._session(cfg)


def test_torch_loads_before_onnxruntime_and_preloads_dlls(tmp_path, monkeypatch):
    import types
    from dataclasses import replace

    cfg = replace(config(tmp_path), tao_require_gpu=False)
    events = []
    torch_module = types.SimpleNamespace(version=types.SimpleNamespace(cuda="13.2"))

    class Input:
        name = "input"

    class Session:
        @staticmethod
        def get_providers():
            return ["CPUExecutionProvider"]

        @staticmethod
        def get_inputs():
            return [Input(), Input()]

    ort_module = types.SimpleNamespace(
        __version__="1.30.0",
        get_available_providers=lambda: ["CPUExecutionProvider"],
        preload_dlls=lambda: events.append("preload"),
        InferenceSession=lambda *args, **kwargs: Session(),
    )

    def load(name):
        events.append(name)
        return torch_module if name == "torch" else ort_module

    monkeypatch.setattr("blower_inspection.tao_inspector.importlib.import_module", load)
    TaoInspector()._session(cfg)

    assert events == ["torch", "onnxruntime", "preload"]


def test_inspection_score_is_normalized_to_calibrated_fail_line(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    reference = TaoInspector.reference_path(cfg)
    import cv2

    assert cv2.imwrite(str(reference), np.zeros((20, 30, 3), np.uint8))
    calibration = TaoCalibration("model", "reference", 0.001, 0.001, 0, 0, 20, "now")
    inspector = TaoInspector()
    monkeypatch.setattr(inspector, "_calibration", lambda runtime_config: calibration)
    monkeypatch.setattr(
        inspector,
        "_infer",
        lambda runtime_config, golden, candidate: (
            np.full((8, 8), 0.002, np.float32),
            0.0001,
        ),
    )

    result = inspector.inspect(
        cfg, np.zeros((20, 30, 3), np.uint8), save_outputs=False, crop_to_component=False
    )

    assert result.status == "FAIL"
    assert result.anomaly_score == pytest.approx(2.0)
