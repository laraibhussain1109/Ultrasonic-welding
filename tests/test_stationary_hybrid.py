"""Regression tests for stationary capture, heatmap area, and physical coverage."""
from dataclasses import replace

import numpy as np
import pytest

from blower_inspection.frame_quality import FrameQualityAnalyzer
from blower_inspection.heatmap_area import evaluate_heatmap_area
from blower_inspection.stationary_capture import INSPECTION_ANGLES, StationaryViewCapture
from blower_inspection.yolo_tracking import RotatingPartInspector, TrackedPart


def surface():
    return np.random.default_rng(42).integers(40, 190, (80, 240, 3), dtype=np.uint8)


def move_then_stop(capture, image, start=0, dwell=30):
    packets = []
    for index in range(8 + dwell):
        frame = np.roll(image, index * 3, axis=1) if index < 8 else image
        packet = capture.offer(frame, now=start + index / 30)
        if packet:
            packets.append(packet)
    return packets


def test_fit_rotation_is_skipped_then_six_identical_stopped_surfaces_are_captured():
    image = surface()
    capture = StationaryViewCapture(burst_frames=10, settle_ms=50)
    assert move_then_stop(capture, image) == []
    packets = []
    for index, angle in enumerate(INSPECTION_ANGLES):
        selected = move_then_stop(capture, image, start=(index + 1) * 2)
        assert len(selected) == 1
        assert selected[0].angle == angle
        assert selected[0].valid and selected[0].burst_frames == 10
        np.testing.assert_array_equal(selected[0].frame, image)
        packets.extend(selected)
    assert len(packets) == 6
    assert move_then_stop(capture, image, start=20) == []


def test_stationary_dwell_never_invents_rotation_or_duplicate_views():
    image = surface()
    capture = StationaryViewCapture(skip_fit_rotation=False, settle_ms=0, burst_frames=10)
    assert all(capture.offer(image, now=i / 30) is None for i in range(100))
    assert [packet.angle for packet in move_then_stop(capture, image, start=4, dwell=100)] == [60]


def test_burst_quality_selects_sharpest_qualified_still_and_does_not_infer_blurry_view(monkeypatch):
    image = surface()
    capture = StationaryViewCapture(skip_fit_rotation=False, settle_ms=0, burst_frames=10)
    motion = iter([False] + [True] * 5 + [False] * 15)
    monkeypatch.setattr(capture, "_moving", lambda _image: next(motion))
    packets = []
    for index in range(21):
        # A sharper but overexposed photograph must not win selection.
        frame = np.full_like(image, 100) if index < 10 else image
        if index == 13:
            frame = image.copy()
            frame[:, ::2] = 255
        packet = capture.offer(frame, now=index / 30)
        if packet:
            packets.append(packet)
    assert len(packets) == 1 and packets[0].valid
    np.testing.assert_array_equal(packets[0].frame, image)


def test_interrupted_stop_is_invalid_and_does_not_relabel_the_next_stop(monkeypatch):
    capture = StationaryViewCapture(skip_fit_rotation=False, settle_ms=0, burst_frames=10,
                                    minimum_burst_frames=4)
    image = surface()
    motion = iter([False] + [True] * 3 + [False] * 5 + [True] * 3 + [False] * 20)
    monkeypatch.setattr(capture, "_moving", lambda _image: next(motion))
    packets = [packet for i in range(32) if (packet := capture.offer(image, now=i / 30))]
    assert [(packet.angle, packet.valid) for packet in packets] == [(60, False), (120, True)]
    assert packets[0].reasons == ("INSUFFICIENT_STATIONARY_FRAMES",)


def test_short_stop_uses_buffered_qualified_stills_without_selecting_moving_frame(monkeypatch):
    capture = StationaryViewCapture(skip_fit_rotation=False, settle_ms=0, burst_frames=15)
    image = surface()
    moving = image.copy()
    moving[:, ::2] = 230
    motion = iter([False] + [True] * 3 + [False] * 7 + [True] * 2)
    monkeypatch.setattr(capture, "_moving", lambda _image: next(motion))
    packets = []
    for index in range(13):
        packet = capture.offer(moving if index in (11, 12) else image, now=index / 30)
        if packet:
            packets.append(packet)
    assert len(packets) == 1 and packets[0].angle == 60
    assert packets[0].valid and 3 <= packets[0].burst_frames < 15
    np.testing.assert_array_equal(packets[0].frame, image)


def test_qualified_short_burst_finishes_within_time_window_while_motor_remains_stopped(monkeypatch):
    capture = StationaryViewCapture(skip_fit_rotation=False, settle_ms=200, burst_frames=15,
                                    burst_window_ms=350)
    image = surface()
    motion = iter([False] + [True] * 3 + [False] * 15)
    monkeypatch.setattr(capture, "_moving", lambda _image: next(motion))
    packets = [packet for index in range(19) if (packet := capture.offer(image, now=index / 10))]
    assert len(packets) == 1 and packets[0].valid
    assert 3 <= packets[0].burst_frames < 15


def test_three_blurred_stills_and_two_sharp_stills_do_not_pass_minimum_evidence(monkeypatch):
    capture = StationaryViewCapture(skip_fit_rotation=False, settle_ms=0, burst_frames=15)
    image = surface()
    motion = iter([False] + [True] * 3 + [False] * 7 + [True] * 2)
    monkeypatch.setattr(capture, "_moving", lambda _image: next(motion))
    packets = []
    for index in range(13):
        frame = image if index in (6, 7) else np.full_like(image, 100)
        packet = capture.offer(frame, now=index / 30)
        if packet:
            packets.append(packet)
    assert len(packets) == 1 and not packets[0].valid
    assert packets[0].reasons == ("INSUFFICIENT_STATIONARY_FRAMES",)


def test_stop_with_no_settled_frame_never_uses_the_first_moving_frame(monkeypatch):
    capture = StationaryViewCapture(skip_fit_rotation=False, settle_ms=500)
    image = surface()
    motion = iter([False] + [True] * 3 + [False] * 3 + [True] * 2)
    monkeypatch.setattr(capture, "_moving", lambda _image: next(motion))
    packets = [packet for index in range(9) if (packet := capture.offer(image, now=index / 30))]
    assert len(packets) == 1 and not packets[0].valid and packets[0].burst_frames == 0
    assert packets[0].reasons == ("NO_SETTLED_STATIONARY_FRAMES",)


def area_result(raw, *, tolerance=5, valid=None, reflection=None, geometry=None, **kwargs):
    shape = raw.shape
    return evaluate_heatmap_area(
        raw, np.ones(shape, bool) if valid is None else valid,
        np.zeros(shape, bool) if reflection is None else reflection,
        np.zeros(shape, bool) if geometry is None else geometry,
        patch_threshold=1, image_score=2, fail_threshold=1.5,
        geometry_score=kwargs.pop("geometry_score", 0), geometry_threshold=.55,
        tolerance_percent=tolerance, min_component_px=kwargs.pop("min_component_px", 4),
        section_count=6, **kwargs,
    )


def test_tolerance_controls_retained_heatmap_area_not_geometry_or_scalar_score():
    raw = np.zeros((100, 120), np.float32)
    raw[20:30, 30:54] = 3  # 240 / 12000 = 2%
    geometry = raw > 0
    allowed = area_result(raw, tolerance=3, geometry=geometry, geometry_score=2)
    rejected = area_result(raw, tolerance=1, geometry=geometry, geometry_score=2)
    assert allowed.percentage == 2 and allowed.decision.status == "PASS"
    assert rejected.area_px == 240 and rejected.decision.status == "FAIL"
    assert area_result(raw, tolerance=2).decision.status == "FAIL"
    assert area_result(np.zeros_like(raw), tolerance=0).decision.status == "PASS"


def test_area_denominator_excludes_padding_and_reflections():
    raw = np.zeros((40, 120), np.float32)
    valid = np.zeros_like(raw, bool)
    valid[10:30] = True
    reflection = np.zeros_like(valid)
    reflection[10:30, :20] = True
    raw[10:30, :20] = 4
    raw[15:25, 50:60] = 4
    result = area_result(raw, valid=valid, reflection=reflection, tolerance=6)
    assert result.valid_area_px == 2000
    assert result.area_px == 100 and result.percentage == 5
    assert result.sections == (2,) and result.decision.status == "PASS"


def test_small_noise_and_thin_scratches_are_filtered_but_geometry_preserves_damage():
    raw = np.zeros((100, 120), np.float32)
    raw[10:12, 10:12] = 3
    raw[30, 10:100] = 3
    filtered = area_result(raw, min_component_px=8, tolerance=0)
    assert filtered.area_px == 0
    preserved = area_result(raw, min_component_px=8, tolerance=0, geometry=raw > 0, geometry_score=1)
    assert preserved.area_px == 90 and preserved.decision.status == "FAIL"


def test_local_geometry_corroboration_preserves_a_reflected_defect():
    raw = np.zeros((60, 120), np.float32)
    raw[20:35, 35:50] = 3
    reflection = raw > 0
    assert area_result(raw, reflection=reflection, tolerance=1).area_px == 0
    retained = area_result(raw, reflection=reflection, geometry=reflection, geometry_score=1, tolerance=1)
    assert retained.area_px == 225 and retained.decision.status == "FAIL"


def test_empty_inspectable_area_and_invalid_heatmap_cannot_pass():
    raw = np.zeros((20, 30), np.float32)
    assert area_result(raw, valid=np.zeros_like(raw, bool)).decision.status == "VIEW INVALID"
    raw[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        area_result(raw)


def session():
    aggregate = RotatingPartInspector(completion_mode="minimum_views", fixed_view_angles=INSPECTION_ANGLES)
    aggregate.observe_tracks([TrackedPart(1, (0, 0, 100, 40), .99)], 120, now=0)
    return aggregate


def test_duplicate_angle_never_increases_coverage_and_sixth_view_finalizes_once():
    aggregate = session()
    for angle in INSPECTION_ANGLES[:-1]:
        assert aggregate.record_inspection(1, is_pass=True, anomaly_score=0, view_angle=angle) is None
        assert aggregate.record_inspection(1, is_pass=True, anomaly_score=0, view_angle=angle) is None
    assert aggregate.view_progress(1) == (5, 5, 6)
    completed = aggregate.record_inspection(1, is_pass=True, anomaly_score=0, view_angle=360)
    assert completed.status == "PASS" and completed.valid_views == 6
    assert aggregate.record_inspection(1, is_pass=True, anomaly_score=0, view_angle=360) is None


def test_six_stops_with_one_invalid_view_fail_closed():
    aggregate = session()
    for angle in INSPECTION_ANGLES:
        completed = aggregate.record_inspection(1, is_pass=True, anomaly_score=0,
                                                view_angle=angle, view_valid=angle != 180)
    assert completed.status == "FAIL" and completed.valid_views == 5
    assert "INSUFFICIENT_VALID_VIEWS" in completed.reason_codes


def test_confirmed_sections_survive_good_views_and_finalization_until_departure():
    aggregate = session()
    for angle in INSPECTION_ANGLES:
        completed = aggregate.record_inspection(1, is_pass=angle != 120, anomaly_score=2,
                                                view_angle=angle, candidate_sections=(2,) if angle == 120 else ())
        assert aggregate.defect_sections(1) == (() if angle == 60 else (2,))
    assert completed.status == "FAIL"
    assert completed.failed_angles == (120,) and completed.defect_sections == (2,)
    assert aggregate.defect_section_angles(1) == {2: (120,)}
    aggregate.observe_tracks([TrackedPart(1, (0, 0, 100, 40), .99)], 120, now=1)
    assert aggregate.defect_sections(1) == (2,)
    aggregate.observe_tracks([], 120, now=2)
    assert aggregate.defect_sections(1) == ()
    assert aggregate.defect_section_angles(1) == {}


def test_pending_stills_are_drained_before_removed_part_is_finalized():
    aggregate = session()
    assert aggregate.observe_tracks([], 120, now=3, pending_track_ids={1}) == []
    for angle in INSPECTION_ANGLES:
        completed = aggregate.record_inspection(1, is_pass=True, anomaly_score=0, view_angle=angle)
    assert completed.status == "PASS"
    assert aggregate.observe_tracks([], 120, now=4) == []


def test_nuisance_candidates_in_different_sections_do_not_accumulate_as_one_flaw():
    aggregate = session()
    for angle, section in zip(INSPECTION_ANGLES, range(6)):
        completed = aggregate.record_inspection(1, is_pass=False, immediate_failure=False,
                                                provisional_candidate=True, anomaly_score=.8,
                                                candidate_sections=(section,), view_angle=angle)
    assert completed.status == "PASS" and aggregate.defect_sections(1) == ()


def test_same_section_recurring_after_an_intervening_good_view_is_latched():
    aggregate = session()
    for angle in INSPECTION_ANGLES:
        candidate = angle in (60, 180)
        completed = aggregate.record_inspection(1, is_pass=not candidate, immediate_failure=False,
                                                provisional_candidate=candidate, anomaly_score=.8,
                                                candidate_sections=(3,) if candidate else (), view_angle=angle)
    assert completed.status == "FAIL" and aggregate.defect_sections(1) == (3,)
    assert completed.failed_angles == (60, 180)
    assert aggregate.defect_section_angles(1) == {3: (60, 180)}
