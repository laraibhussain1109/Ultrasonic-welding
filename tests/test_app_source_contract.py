"""Small source-level regression checks that do not require a Qt display."""

from pathlib import Path


def test_app_uses_the_initialized_fail_output_bridge_at_startup():
    source = Path("src/blower_inspection/app.py").read_text(encoding="utf-8")

    assert "self.fail_output = ESP32FailOutputBridge()" in source
    assert "self.fail_output.reset()" in source
    assert "self.esp32_output" not in source
