"""Small source-level regression checks that do not require a Qt display."""

from pathlib import Path


def test_app_uses_the_initialized_fail_output_bridge_at_startup():
    source = Path("src/blower_inspection/app.py").read_text(encoding="utf-8")

    assert "self.fail_output = ESP32FailOutputBridge()" in source
    assert "self.fail_output.reset()" in source
    assert "self.esp32_output" not in source


def test_app_exposes_visual_changenet_export_before_calibration():
    source = Path("src/blower_inspection/app.py").read_text(encoding="utf-8")
    assert "EXPORT TAO MODEL" in source
    assert "run_visual_changenet_task(" in source
    assert '"export", spec' in source


def test_app_validates_tao_readiness_before_opening_camera():
    source = Path("src/blower_inspection/app.py").read_text(encoding="utf-8")
    validation = source.index("self.inspector.validate_ready(model)")
    camera_open = source.index("self.camera.open()", validation)

    assert validation < camera_open
    assert '"TAO model not ready"' in source


def test_app_does_not_replace_the_inspector_after_readiness_validation():
    source = Path("src/blower_inspection/app.py").read_text(encoding="utf-8")
    start = source.index("def start_inspection")
    validation = source.index("self.inspector.validate_ready(model)", start)
    camera_open = source.index("self.camera.open()", validation)

    assert "self._apply_selected_camera_settings()" not in source[validation:camera_open]


def test_app_distinguishes_view_pass_from_final_part_pass():
    source = Path("src/blower_inspection/app.py").read_text(encoding="utf-8")

    assert 'badge_object, badge_text = "statusPass", "VIEW PASS"' in source
    assert "VIEW SCORE:" in source
    assert "VIEWS:" in source


def test_app_imports_backend_factory_from_its_own_module():
    source = Path("src/blower_inspection/app.py").read_text(encoding="utf-8")
    assert "from .inspector_factory import inspector_for_model" in source
    assert "from .tao_inspector import inspector_for_model" not in source


def test_app_offers_reduced_tolerance_presets_and_manual_float_input():
    source = Path("src/blower_inspection/app.py").read_text(encoding="utf-8")

    assert "enumerate([1, 3, 5, 8])" in source
    assert 'QPushButton("MANUAL")' in source
    assert "self.manual_tolerance.setRange(0.00, 50.00)" in source
    assert "self.manual_tolerance.setDecimals(2)" in source
