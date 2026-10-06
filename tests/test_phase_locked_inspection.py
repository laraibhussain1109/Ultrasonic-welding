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


@pytest.fixture
def passthrough_yolo(monkeypatch):
    class Detector:
        def __init__(self, *_args, **_kwargs): pass
        def exact_crop(self, image): return image.copy()
    monkeypatch.setattr("blower_inspection.yolo_tracking.YoloByteTrackDetector", Detector)


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


def test_desktop_train_contract_builds_every_phase_bank(tmp_path, passthrough_yolo):
    from blower_inspection.config import PartModelConfig

    training = tmp_path / "training" / "BF" / "normal"
    model_root = tmp_path / "models"
    for angle in ANGLES:
        for part_index, delta in enumerate((-3, 0, 3), 1):
            folder = training.parent / f"phase_{angle:03d}" / "good" / f"PART-{part_index:03d}"
            folder.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(folder / "frame.png"), np.clip(fan().astype(np.int16) + delta, 0, 255).astype(np.uint8))
    model = PartModelConfig("BF", "Blower", training, tmp_path / "legacy.pt", tmp_path / "results", 36,
                            golden_model_root=model_root, training_min_sharpness=1,
                            yolo_model_path=tmp_path / "yolo.pt", phase_input_width=320, phase_input_height=180)
    engine = PhaseLockedStructuralInspector(GoldenBank(model_root, "BF"), config())
    updates = []
    manifest = engine.train(model, updates.append)
    assert manifest.is_file()
    assert len(updates) == 24
    assert updates[-1].completed == updates[-1].total
    assert all(engine.bank.path(angle).is_file() for angle in ANGLES)


def test_unkeyed_normal_folder_automatically_calibrates_all_plc_phases(tmp_path, passthrough_yolo):
    from blower_inspection.config import PartModelConfig

    normal = tmp_path / "training" / "BF" / "normal"
    normal.mkdir(parents=True)
    for index, delta in enumerate((-6, -3, 0, 3, 6)):
        cv2.imwrite(str(normal / f"good-{index}.png"),
                    np.clip(fan().astype(np.int16) + delta, 0, 255).astype(np.uint8))
    model_root = tmp_path / "models"
    model = PartModelConfig("BF", "Blower", normal, tmp_path / "legacy.pt", tmp_path / "results", 36,
                            golden_model_root=model_root, golden_calibration_max_images=4,
                            training_min_sharpness=1, yolo_model_path=tmp_path / "yolo.pt",
                            phase_input_width=320, phase_input_height=180)
    engine = PhaseLockedStructuralInspector(GoldenBank(model_root, "BF"), config())

    manifest = engine.train(model)

    assert manifest.is_file()
    for angle in ANGLES:
        golden = GoldenPhaseModel.load(engine.bank.path(angle), expected_angle=angle)
        assert golden.metadata["calibration_source_mode"] == "pooled_unkeyed_rotation"
        assert golden.metadata["calibration_images"] == 4
        assert golden.median_gray.shape == (180, 320)


def test_phase_calibration_rejects_one_physical_part(tmp_path, passthrough_yolo):
    from blower_inspection.config import PartModelConfig

    training = tmp_path / "training" / "BF" / "normal"
    for angle in ANGLES:
        folder = training.parent / f"phase_{angle:03d}" / "good" / "SAME-PART"
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(3):
            cv2.imwrite(str(folder / f"{index}.png"), fan())
    model = PartModelConfig("BF", "Blower", training, tmp_path / "legacy.pt", tmp_path / "results", 36,
                            golden_model_root=tmp_path / "models", training_min_sharpness=1,
                            yolo_model_path=tmp_path / "yolo.pt")
    engine = PhaseLockedStructuralInspector(GoldenBank(model.golden_model_root, "BF"), config())
    with pytest.raises(ValueError, match="independent physical GOOD part IDs"):
        engine.train(model)


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


def test_operator_pixel_area_ignore_suppresses_smaller_regions(bank):
    damaged = fan(); cv2.rectangle(damaged, (155, 80), (175, 91), (85, 85, 85), -1)
    strict = PhaseLockedStructuralInspector(bank, config(minimum_region_area=8))
    ignore_small = PhaseLockedStructuralInspector(bank, config(minimum_region_area=1000))
    assert strict.inspect_phase(180, [damaged] * 5).verdict == Verdict.FAIL
    assert ignore_small.inspect_phase(180, [damaged] * 5).verdict == Verdict.PASS


def test_missing_and_tilted_fins_are_structural(bank):
    engine = PhaseLockedStructuralInspector(bank, config())
    assert engine.inspect_phase(240, [fan(missing=True)] * 5).verdict == Verdict.FAIL
    assert engine.inspect_phase(300, [fan(tilted=True)] * 5).verdict == Verdict.FAIL


def test_blur_rejection_and_wrong_phase(bank):
    engine = PhaseLockedStructuralInspector(bank, config(minimum_sharpness=1000))
    assert engine.inspect_phase(60, [cv2.GaussianBlur(fan(), (31, 31), 8)] * 5).verdict == Verdict.VIEW_INVALID
    assert engine.inspect_phase(90, [fan()] * 5).reason == "WRONG_PLC_PHASE"


def test_generic_live_call_is_fail_closed_without_attribute_error(bank):
    engine = PhaseLockedStructuralInspector(bank, config())
    result = engine.inspect(None, fan())
    assert result.status == "VIEW INVALID"
    assert result.reason_codes == ("PLC_PHASE_REQUIRED",)
    assert not result.view_valid


def test_generic_live_call_accepts_authoritative_phase_burst(bank):
    engine = PhaseLockedStructuralInspector(bank, config())
    result = engine.inspect(None, fan(), phase_angle=60, burst_frames=[fan()] * 5)
    assert result.status == "PASS"
    assert result.view_valid


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
