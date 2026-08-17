"""Score-map regressions that do not require the OpenCV shared libraries."""

import ast
from pathlib import Path

import numpy as np
import pytest


def _load_function(name):
    source = Path("src/blower_inspection/supersimplenet.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    function = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "supersimplenet.py", "exec"), namespace)
    return namespace[name]


def test_uniform_high_model_response_does_not_paint_the_component_red():
    localize = _load_function("localize_anomaly_scores")
    raw = np.full((40, 120), 0.97, dtype=np.float32)
    surface = np.ones_like(raw, dtype=bool)

    localized, baseline = localize(raw, surface)

    assert baseline == pytest.approx(0.97)
    assert np.count_nonzero(localized) == 0


def test_localized_high_response_survives_surface_baseline_removal():
    localize = _load_function("localize_anomaly_scores")
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


def test_component_sized_response_is_suppressed_before_overlay():
    suppress = _load_function("suppress_broad_response")
    surface = np.ones((40, 120), dtype=bool)
    mask = np.zeros_like(surface)
    mask[5:35, 5:115] = True
    scores = mask.astype(np.float32)

    filtered_scores, filtered_mask, was_suppressed = suppress(scores, mask, surface)

    assert was_suppressed
    assert np.count_nonzero(filtered_mask) == 0
    assert np.count_nonzero(filtered_scores) == 0


def test_local_defect_response_is_not_suppressed():
    suppress = _load_function("suppress_broad_response")
    surface = np.ones((40, 120), dtype=bool)
    mask = np.zeros_like(surface)
    mask[15:20, 50:60] = True
    scores = mask.astype(np.float32)

    filtered_scores, filtered_mask, was_suppressed = suppress(scores, mask, surface)

    assert not was_suppressed
    assert np.array_equal(filtered_mask, mask)
    assert np.array_equal(filtered_scores, scores)
