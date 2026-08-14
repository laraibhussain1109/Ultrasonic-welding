"""Score-map regressions that do not require the OpenCV shared libraries."""

import ast
from pathlib import Path

import numpy as np
import pytest


def _load_localizer():
    source = Path("src/blower_inspection/supersimplenet.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    function = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "localize_anomaly_scores"
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "supersimplenet.py", "exec"), namespace)
    return namespace["localize_anomaly_scores"]


def test_uniform_high_model_response_does_not_paint_the_component_red():
    localize = _load_localizer()
    raw = np.full((40, 120), 0.97, dtype=np.float32)
    surface = np.ones_like(raw, dtype=bool)

    localized, baseline = localize(raw, surface)

    assert baseline == pytest.approx(0.97)
    assert np.count_nonzero(localized) == 0


def test_localized_high_response_survives_surface_baseline_removal():
    localize = _load_localizer()
    raw = np.full((40, 120), 0.20, dtype=np.float32)
    raw[15:20, 50:60] = 0.92
    surface = np.ones_like(raw, dtype=bool)

    localized, baseline = localize(raw, surface)

    assert baseline == pytest.approx(0.20)
    assert float(localized[17, 55]) > 0.85
    assert np.count_nonzero(localized >= 0.65) == 50


def test_live_inference_explicitly_converts_opencv_bgr_to_rgb():
    source = Path("src/blower_inspection/supersimplenet.py").read_text(encoding="utf-8")

    assert "cv2.cvtColor(image, cv2.COLOR_BGR2RGB)" in source
