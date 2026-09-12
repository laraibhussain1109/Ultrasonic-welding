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
