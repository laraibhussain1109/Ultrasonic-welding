"""Small source-level regression checks that do not require a Qt display."""

from pathlib import Path


def test_app_uses_the_initialized_fail_output_bridge_at_startup():
    source = Path("src/blower_inspection/app.py").read_text(encoding="utf-8")

    assert "self.fail_output = ESP32FailOutputBridge()" in source
    assert "self.fail_output.reset()" in source
    assert "self.esp32_output" not in source


def test_tolerance_buttons_control_defect_coverage_and_show_selection():
    source = Path("src/blower_inspection/app.py").read_text(encoding="utf-8")

    assert "[0, 1, 3, 5, 8, 10, 13, 15, 20]" in source
    assert "button.setCheckable(True)" in source
    assert "max_defect_area_ratio=tolerance_ratio" in source
    assert "tolerance_threshold = max(model.anomaly_threshold, 7.5)" in source
