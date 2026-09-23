"""Small source-level regression checks that do not require a Qt display."""

from pathlib import Path


def test_app_uses_the_initialized_fail_output_bridge_at_startup():
    source = Path("src/blower_inspection/app.py").read_text(encoding="utf-8")

    assert "self.fail_output = ESP32FailOutputBridge()" in source
    assert "self.fail_output.reset()" in source
    assert "self.esp32_output" not in source


def test_final_pass_pulse_is_not_immediately_followed_by_departure_reset():
    source = Path("src/blower_inspection/app.py").read_text(encoding="utf-8")

    assert 'if part.status == "PASS":' in source
    assert "self.fail_output.signal_pass()" in source
    assert "if removed_parts and self.active_fail_asserted:" in source


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

    assert 'badge_object, badge_text = "statusStandby", "VIEW OK — CHECKING"' in source
    assert "VIEW SCORE:" in source
    assert "VIEWS:" in source


def test_app_preflights_part_departure_runtime_support():
    source = Path("src/blower_inspection/app.py").read_text(encoding="utf-8")

    assert "SUPPORTED_COMPLETION_MODES" in source
    assert "Inspection runtime is out of date" in source
    assert "blower-inspection doctor" in source


def test_app_can_draw_a_horizontal_x_axis_counting_line():
    source = Path("src/blower_inspection/app.py").read_text(encoding="utf-8")

    assert 'if counting_axis == "y"' in source
    assert "(0, line_y), (display.shape[1] - 1, line_y)" in source


def test_app_waits_for_distinct_rotation_phase_before_another_view():
    source = Path("src/blower_inspection/app.py").read_text(encoding="utf-8")

    assert "RotationPhaseGate" in source
    assert 'self.status_badge.setText("WAITING FOR ROTATION")' in source


def test_fixed_nest_uses_one_operator_approved_yolo_roi():
    source = Path("src/blower_inspection/app.py").read_text(encoding="utf-8")

    assert "def _confirm_and_lock_roi" in source
    assert "self.part_detector.detect_best(raw_frame)" in source
    assert '"Confirm fixed inspection ROI"' in source
    assert "self.live_roi_bounds = detected.bounds" in source
    assert "def _locked_roi_tracks" in source
    assert "TrackedPart(self.locked_track_id, self.live_roi_bounds" in source


def test_completed_fixed_nest_polls_for_departure_without_the_inspection_debounce():
    source = Path("src/blower_inspection/app.py").read_text(encoding="utf-8")

    assert "self.rotating_parts.awaiting_departure(self.locked_track_id)" in source
    assert "or awaiting_departure" in source
    assert "missing_limit = (2 if awaiting_departure" in source


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
