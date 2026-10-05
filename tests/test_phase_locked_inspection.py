from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", exc_type=ImportError)

from blower_inspection.phase_locked import (
    ANGLES, FitmentObservation, GoldenBank, GoldenPhaseModel, PhaseLockedConfig,
    PhaseLockedStructuralInspector, Verdict, analyze_fitment, calibration_report,
    glare_mask, illumination_normalized,
)
from blower_inspection.plc_inspection import PLCEvent, PhaseInspectionController


def fan(shape=(180, 320), missing=False, tilted=False):
    image = np.full((*shape, 3), 85, np.uint8)
    for y in range(25, shape[0] - 20, 12):
        if missing and y == 85:
            continue
        if tilted and y == 85:
            cv2.line(image, (10, y - 8), (shape[1] - 10, y + 8), (230, 230, 230), 2)
        else:
            cv2.line(image, (10, y), (shape[1] - 10, y), (230, 230, 230), 2)
    cv2.line(image, (80, 15), (80, shape[0] - 15), (150, 150, 150), 5)
    return image


@pytest.fixture
def bank(tmp_path):
    root = tmp_path / "models"
    samples = [np.clip(fan().astype(np.int16) + delta, 0, 255).astype(np.uint8) for delta in (-4, 0, 4)]
    for angle in ANGLES:
        GoldenPhaseModel.calibrate(angle, samples, model_id="BF", roi=(0, 0, 320, 180),
                                   physical_part_ids=["A", "B", "C"]).save(
            root / "BF" / f"phase_{angle:03d}" / "golden_v1.npz")
    return GoldenBank(root, "BF")


def config(**kwargs):
    values = dict(settle_delay_ms=0, burst_frame_count=5, minimum_qualified_frames=3,
                  temporal_confirmation_frames=3, minimum_sharpness=1,
                  registration_min_correlation=.2, minimum_region_area=8,
                  golden_candidate_threshold=2.5)
    values.update(kwargs)
    return PhaseLockedConfig(**values)


def test_golden_median_mad_and_metadata(bank):
    model = bank.get(120, (180, 320))
    assert model.metadata["physical_good_parts"] == 3
    assert np.all(model.scale_gradient >= .035)
    with pytest.raises(ValueError, match="Wrong phase"):
        GoldenPhaseModel.load(bank.path(120), expected_angle=240)


def test_phase_specific_reference_loading_missing(bank):
    with pytest.raises(FileNotFoundError, match="Phase 60"):
        GoldenBank(bank.root, "OTHER").get(60)


def test_illumination_and_broad_brightness_are_tolerated(bank):
    normalized_a = illumination_normalized(fan())
    normalized_b = illumination_normalized(np.clip(fan().astype(np.int16) + 35, 0, 255).astype(np.uint8))
    assert np.mean(np.abs(normalized_a - normalized_b)) < .04
    result = PhaseLockedStructuralInspector(bank, config()).inspect_phase(60, [fan() + 20] * 5)
    assert result.verdict == Verdict.PASS


def test_broad_glare_is_not_a_fail(bank):
    image = fan()
    cv2.rectangle(image, (120, 20), (230, 160), (255, 255, 255), -1)
    assert glare_mask(image).any()
    result = PhaseLockedStructuralInspector(bank, config()).inspect_phase(60, [image] * 5)
    assert result.verdict != Verdict.FAIL


def test_single_frame_structural_noise_is_not_fail(bank):
    damaged = fan(); cv2.rectangle(damaged, (150, 80), (190, 91), (85, 85, 85), -1)
    result = PhaseLockedStructuralInspector(bank, config()).inspect_phase(60, [damaged, fan(), fan(), fan(), fan()])
    assert result.verdict != Verdict.FAIL


def test_persistent_local_edge_break_fails(bank):
    damaged = fan(); cv2.rectangle(damaged, (140, 80), (210, 91), (85, 85, 85), -1)
    result = PhaseLockedStructuralInspector(bank, config()).inspect_phase(180, [damaged] * 5)
    assert result.verdict == Verdict.FAIL
    assert result.defects[0].persistence_count >= 3
    assert "STATIONARY_TEMPORAL_PERSISTENCE" in result.defects[0].evidence


def test_missing_and_tilted_fins_are_structural(bank):
    engine = PhaseLockedStructuralInspector(bank, config())
    assert engine.inspect_phase(240, [fan(missing=True)] * 5).verdict == Verdict.FAIL
    assert engine.inspect_phase(300, [fan(tilted=True)] * 5).verdict == Verdict.FAIL


def test_blur_rejection_and_wrong_phase(bank):
    engine = PhaseLockedStructuralInspector(bank, config(minimum_sharpness=1000))
    assert engine.inspect_phase(60, [cv2.GaussianBlur(fan(), (31, 31), 8)] * 5).verdict == Verdict.VIEW_INVALID
    assert engine.inspect_phase(90, [fan()] * 5).reason == "WRONG_PLC_PHASE"


def test_fitment_runout_and_tracking():
    good = [FitmentObservation(100 + np.sin(i) * .5, 50, 200, 40, 100) for i in range(20)]
    bad = [FitmentObservation(100 + (-1) ** i * 20, 50, 200, 40, 100) for i in range(20)]
    assert analyze_fitment(good, config()).verdict == Verdict.PASS
    result = analyze_fitment(bad, config())
    assert result.verdict == Verdict.FITMENT_FAIL
    assert result.statistics["runout_p99"] > 8


def _fitment(controller):
    controller.event(PLCEvent("PART_PRESENT"))
    controller.event(PLCEvent("FITMENT_START"))
    for i in range(12):
        controller.observe_fitment(FitmentObservation(100, 50, 200, 40, 100))
    controller.event(PLCEvent("FITMENT_COMPLETE"))
    controller.event(PLCEvent("HOME", 0, True))
    controller.lock_home(fan())


def test_plc_order_duplicate_and_moving_frame_protection(bank, tmp_path):
    controller = PhaseInspectionController(PhaseLockedStructuralInspector(bank, config()), config(), tmp_path)
    _fitment(controller)
    with pytest.raises(RuntimeError, match="Moving"):
        controller.submit_frame(fan())
    controller.event(PLCEvent("POSITION", 120, True))
    assert controller.final_verdict == Verdict.INSPECTION_INVALID
    assert "PLC_PHASE_ORDER_ERROR" in controller.reason_codes


def test_stationary_state_machine_and_final_aggregation(bank, tmp_path):
    cfg = config()
    controller = PhaseInspectionController(PhaseLockedStructuralInspector(bank, cfg), cfg, tmp_path)
    _fitment(controller)
    for angle in ANGLES:
        controller.event(PLCEvent("POSITION", angle, True))
        for _ in range(5):
            result = controller.submit_frame(fan(), now=controller.phase_started_at + 1)
        assert result and result.verdict == Verdict.PASS
    assert controller.check_closure(fan()) == Verdict.PASS
    assert next(tmp_path.glob("*.json"))


def test_final_home_drift_is_invalid(bank, tmp_path):
    cfg = config(closure_max_translation_ratio=.005)
    controller = PhaseInspectionController(PhaseLockedStructuralInspector(bank, cfg), cfg, tmp_path)
    _fitment(controller)
    for angle in ANGLES:
        controller.event(PLCEvent("POSITION", angle, True))
        for _ in range(5): controller.submit_frame(fan(), now=controller.phase_started_at + 1)
    matrix = np.float32([[1, 0, 15], [0, 1, 0]])
    shifted = cv2.warpAffine(fan(), matrix, (320, 180))
    assert controller.check_closure(shifted) == Verdict.INSPECTION_INVALID
    assert "POSITION_DRIFT" in controller.reason_codes


def test_plc_protocol_and_qualification_report():
    assert PLCEvent.parse("POSITION_180").angle == 180
    assert PLCEvent.parse("PHASE:360").motor_stopped
    with pytest.raises(ValueError): PLCEvent.parse("POSITION_90")
    report = calibration_report({"runout": [1, 2, 3, 4]}, [Verdict.PASS, Verdict.PASS])
    assert report["good_false_reject_rate"] == 0
    assert report["channels"]["runout"]["p99"] > 3
