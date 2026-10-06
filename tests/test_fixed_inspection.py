"""Behavioral checks for the stopped-position PatchCore/tolerance workflow."""
import json
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from blower_inspection.anomaly_tolerance import process_heatmap, heatmap_image
from blower_inspection.fixed_settings import InspectionSettings, ToleranceSettings
from blower_inspection.fixed_views import ANGLES, FixedViewController, FixedViewPipeline, PartInspectionResult, ViewInspectionResult
from blower_inspection.inspection_storage import ResultStorage, export_history, load_saved_view
from blower_inspection.machine_signals import MachineEvent
from blower_inspection.stationary_roi import InvalidView, PreparedROI, QualityEvidence, YoloROI, select_best_frame
from blower_inspection.tolerance_calibration import CalibrationSample, evaluate_samples


def tolerance(**changes):
    return replace(ToleranceSettings(), minimum_component_pixels=1, ignore_top_percent=0,
                   ignore_bottom_percent=0, ignore_left_percent=0, ignore_right_percent=0, **changes)


def make_view(angle=60, failed=False, settings=None):
    settings = settings or tolerance()
    heatmap = np.zeros((100, 100), np.float32)
    if failed:
        heatmap[10:70, 10:70] = 1
    mask = np.ones_like(heatmap, bool)
    image = np.full((100, 100, 3), 100, np.uint8)
    return ViewInspectionResult(ANGLES.index(angle)+1, angle, "2026-10-06T00:00:00Z", image, image, mask,
        (0, 0, 100, 100), QualityEvidence(True, 92, 845, 100, 0, 0, ()), .97, 10000,
        heatmap, process_heatmap(heatmap, settings, mask), settings, "test.pt", 10)


@pytest.mark.parametrize("mode,px,percent,expected", [
    ("PIXEL", 100, 2, "FAIL"), ("PERCENTAGE", 101, 1, "FAIL"),
    ("BOTH", 100, 1, "FAIL"), ("BOTH", 100, 2, "PASS"),
    ("BOTH", 101, 1, "PASS"), ("EITHER", 101, 1, "FAIL"), ("EITHER", 101, 2, "PASS")])
def test_tolerance_modes_use_inclusive_pixel_and_percentage_limits(mode, px, percent, expected):
    heatmap = np.zeros((100, 100), np.float32)
    heatmap[10:20, 10:20] = .5
    decision = process_heatmap(heatmap, tolerance(decision_mode=mode, pixel_threshold=px, percentage_threshold=percent))
    assert decision.filtered_anomaly_pixels == 100
    assert decision.anomaly_percentage == 1
    assert decision.verdict == expected


def test_small_components_borders_and_letterbox_do_not_reject():
    heatmap = np.zeros((100, 200), np.float32)
    heatmap[:20] = 1  # letterbox padding
    heatmap[20:24, :] = 1  # ignored upper content edge
    heatmap[30:34, 50:55] = 1  # 20 px noise
    heatmap[40:50, 50:60] = 1  # 100 px retained
    content = np.zeros_like(heatmap, bool)
    content[20:80, 20:180] = True
    settings = replace(tolerance(), minimum_component_pixels=50, ignore_top_percent=10,
                       ignore_bottom_percent=10, ignore_left_percent=10, ignore_right_percent=10)
    d = process_heatmap(heatmap, settings, content)
    assert d.valid_roi_pixels == 48 * 128
    assert d.raw_anomaly_pixels > 4000
    assert d.valid_candidate_pixels == 120
    assert d.filtered_anomaly_pixels == d.largest_component_pixels == 100
    assert len(d.regions) == 1
    assert d.regions[0].bounding_box == (50, 40, 10, 10)
    assert not d.filtered_mask[~d.valid_mask].any()


def test_image_score_has_no_acceptance_authority_and_retuning_preserves_raw_map():
    view = make_view()
    assert view.patchcore_score == 10000
    assert view.verdict == "PASS"
    failed = make_view(failed=True)
    original = failed.heatmap.copy()
    tuned = failed.retune(replace(failed.tolerance, heatmap_threshold=1.1))
    assert tuned.verdict == "PASS" and failed.verdict == "FAIL"
    assert np.array_equal(original, tuned.heatmap)


def test_fixed_heatmap_color_scale_does_not_renormalize_each_image():
    one = np.array([[.25, .5]], np.float32)
    two = np.array([[.25, .99]], np.float32)
    assert np.array_equal(heatmap_image(one, 1)[0, 0], heatmap_image(two, 1)[0, 0])


@pytest.mark.parametrize("bad", [np.array([[np.nan]]), np.array([[np.inf]]), np.zeros((3, 3, 3)), np.zeros((0, 0))])
def test_invalid_heatmap_cannot_become_pass(bad):
    with pytest.raises(ValueError):
        process_heatmap(bad, tolerance())


def test_morphology_does_not_restore_ignored_border_pixels():
    map = np.ones((40, 40), np.float32)
    d = process_heatmap(map, replace(tolerance(), opening_kernel=3, closing_kernel=3, ignore_top_percent=10))
    assert not d.filtered_mask[:4].any()


def controller():
    defaults = InspectionSettings()
    return FixedViewController(replace(defaults, camera=replace(defaults.camera, burst_frames=10),
            machine=replace(defaults.machine, settle_delay_s=.1, acquisition_time_s=.6, minimum_burst_frames=2)))


def collect(c, angle, time=0, failed=False):
    c.stopped(angle, now=time)
    frame = np.zeros((10, 10, 3), np.uint8)
    for index in range(10):
        ready = c.frame(frame, time+.11+index*.04)
    assert ready
    c.accept(make_view(angle, failed), now=time+.8)


def test_exact_six_angles_one_failed_side_never_averaged_away():
    c = controller()
    c.begin("PART_1", fit_check_complete=True, now=0)
    for i, angle in enumerate(ANGLES):
        if i:
            c.moving()
        collect(c, angle, time=i+1, failed=angle==300)
        assert len(c.part.views) == i+1
        if i < 5:
            assert c.part.final_verdict == "INCOMPLETE"
    assert c.part.final_verdict == "FAIL"
    assert c.part.failed_views == [300]
    c.stopped(360, now=10)  # duplicate state must not restart view 6
    assert len(c.part.views) == 6


def test_pass_requires_six_out_of_six_not_five():
    c = controller()
    c.begin("PART", fit_check_complete=True, now=0)
    for i, angle in enumerate(ANGLES):
        if i:
            c.moving()
        collect(c, angle, time=i+1)
        assert c.part.final_verdict == ("PASS" if i==5 else "INCOMPLETE")


def test_fit_check_skipped_angle_and_missing_motion_are_rejected():
    c = controller()
    with pytest.raises(ValueError, match="fit check"):
        c.begin("PART", fit_check_complete=False, now=0)
    c.begin("PART", fit_check_complete=True, now=0)
    with pytest.raises(ValueError, match="Expected"):
        c.stopped(120, now=0)
    collect(c, 60)
    with pytest.raises(ValueError, match="moving"):
        c.stopped(120, now=1)
    assert len(c.part.views) == 1


def test_motion_settling_and_stale_frame_cannot_count_a_view():
    c = controller()
    c.begin("PART", fit_check_complete=True, now=0)
    image = np.zeros((10, 10, 3), np.uint8)
    assert not c.frame(image, .5)
    c.stopped(60, now=1)
    assert not c.frame(image, 1.05)  # first post-stop frame is ignored
    assert not c.frame(image, 1.2)
    assert not c.frame(image, 1.2)  # duplicate capture timestamp
    assert len(c.burst) == 1
    c.moving()
    assert not c.burst
    with pytest.raises(ValueError, match="Stale"):
        c.accept(make_view(), now=1.3)
    assert len(c.part.views) == 0


def test_invalid_view_retries_same_position_and_abort_is_a_fault():
    c = controller()
    c.begin("PART", fit_check_complete=True, now=0)
    c.stopped(60, now=1)
    c.invalid("IMAGE QUALITY FAILED")
    assert "VIEW 1" in c.state and c.expected_angle == 60
    collect(c, 60, time=2)
    c.fault("ABORT", now=3)
    assert c.part.final_verdict == "FAULT"


def test_missing_camera_and_cycle_timeout_never_pass():
    c = controller()
    c.begin("PART", fit_check_complete=True, now=0)
    c.stopped(60, now=1)
    c.tick(4)
    assert "ACQUISITION TIMEOUT" in c.state and c.part.final_verdict == "INCOMPLETE"
    c.tick(100)
    assert c.part.final_verdict == "FAULT"


def test_best_frame_selection_rejects_blur_and_overexposure():
    frame = np.full((100, 100, 3), 100, np.uint8)
    sharp = frame.copy()
    sharp[::4] = 200
    selected, quality = select_best_frame([frame, sharp, np.full_like(frame, 255)], InspectionSettings().quality)
    assert selected is sharp and quality.valid
    with pytest.raises(InvalidView, match="QUALITY"):
        select_best_frame([frame], InspectionSettings().quality)


class FakeArray:
    def __init__(self, values): self.values = np.asarray(values)
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.values


class FakeYolo:
    def __init__(self, box, confidence=.97):
        class Boxes(SimpleNamespace):
            def __len__(self):
                return 1
        self.boxes = Boxes(conf=FakeArray([confidence]), xyxy=FakeArray([box]))
        self.options = None
    def predict(self, frame, **options):
        self.options = options
        return [SimpleNamespace(boxes=self.boxes)]


@pytest.mark.parametrize("box,confidence", [((-1,10,80,80),.97), ((10,10,110,80),.97), ((10,10,80,80),.4)])
def test_yolo_partial_roi_and_low_confidence_are_invalid(box, confidence):
    settings = InspectionSettings()
    frame = np.full((100, 100, 3), 100, np.uint8)
    frame[::4] = 200
    with pytest.raises(InvalidView, match="PARTIAL|CONFIDENCE"):
        YoloROI(settings, FakeYolo(box, confidence)).prepare(frame)


def test_yolo_normalization_letterboxes_and_passes_iou():
    frame = np.full((100, 100, 3), 100, np.uint8)
    frame[::4] = 200
    settings = InspectionSettings()
    model = FakeYolo((10,10,90,90))
    roi = YoloROI(settings, model).prepare(frame)
    assert roi.image.shape == (256,640,3)
    assert roi.content_mask.sum() == 256 * 256
    assert not roi.content_mask[:, :100].any()
    assert model.options["iou"] == settings.yolo.iou


def test_pipeline_uses_best_frame_and_spatial_map_not_huge_image_score():
    settings = InspectionSettings()
    frame = np.full((100, 100, 3), 100, np.uint8)
    frame[::4] = 200
    detector = YoloROI(settings, FakeYolo((10,10,90,90)))
    class Model:
        calls = 0
        def infer(self, roi, angle):
            self.calls += 1
            return np.zeros(roi.image.shape[:2], np.float32), 10000, roi
    model = Model()
    view = FixedViewPipeline(settings, detector, model).inspect_frames([np.full_like(frame, 100), frame], 120)
    assert view.verdict == "PASS" and view.angle == 120 and model.calls == 1
    view.retune(settings.tolerance)
    assert model.calls == 1


def test_failed_view_storage_roundtrip_raw_map_masks_and_history(tmp_path):
    settings = InspectionSettings(storage=replace(InspectionSettings().storage, results_dir=str(tmp_path)))
    storage = ResultStorage(settings)
    views = {angle: make_view(angle, failed=angle==300) for angle in ANGLES}
    part = PartInspectionResult("../../PART", "now", 0, views, end_time="then", cycle_time_s=5)
    path = storage.begin(part)
    for view in views.values():
        storage.save_view(view)
    storage.update_part(part)
    storage.update_part(part)
    directory = path / "view_300"
    expected = {"original.png", "roi.png", "heatmap.npy", "heatmap.png", "overlay.png", "anomaly_mask.png", "filtered_mask.png", "result.png", "result.json", "content_mask.png"}
    assert expected <= {f.name for f in directory.iterdir()}
    loaded = load_saved_view(directory)
    assert np.array_equal(loaded.heatmap, views[300].heatmap)
    assert loaded.verdict == "FAIL"
    assert not (path / "view_60" / "original.png").exists()
    assert export_history(tmp_path, tmp_path / "history.csv") == 6
    assert len((tmp_path / "history.jsonl").read_text().splitlines()) == 1


def test_calibration_statistics_distinguish_false_positive_and_false_negative():
    samples = [CalibrationSample("good-pass", "GOOD", make_view()), CalibrationSample("good-fail", "GOOD", make_view(failed=True)),
               CalibrationSample("ng-fail", "NG", make_view(failed=True)), CalibrationSample("ng-pass", "NG", make_view())]
    rows, metrics = evaluate_samples(samples, tolerance())
    assert len(rows) == 4 and metrics["false_positive_rate"] == .5 and metrics["false_negative_rate"] == .5
    assert evaluate_samples([], tolerance())[1]["false_positive_rate"] is None


@pytest.mark.parametrize("data", [{"event":"begin", "part_id":"P"}, {"event":"stopped","angle":0},
    {"event":"stopped","angle":True}, {"event":"wrong"}])
def test_machine_protocol_rejects_missing_fit_check_and_invalid_position(data):
    with pytest.raises(ValueError):
        MachineEvent.parse(json.dumps(data))


def test_machine_protocol_and_settings_roundtrip(tmp_path):
    assert MachineEvent.parse('{"event":"begin","part_id":"P","fit_check_complete":true}').fit_check_complete
    settings = InspectionSettings()
    settings.save(tmp_path / "settings.json")
    assert InspectionSettings.load(tmp_path / "settings.json") == settings


@pytest.mark.parametrize("section,changes", [("machine", {"views":5}), ("machine", {"output_enabled":True}),
    ("tolerance", {"ignore_left_percent":60,"ignore_right_percent":50}), ("tolerance", {"opening_kernel":7}),
    ("tolerance", {"heatmap_threshold":float("nan")}), ("camera", {"fps":1.1}), ("camera", {"width":True}),
    ("quality", {"minimum_blur_score":"60"}), ("patchcore", {"input_width":63})])
def test_invalid_settings_do_not_silently_clamp_or_enable_output(section, changes):
    data = InspectionSettings().to_dict()
    data[section].update(changes)
    with pytest.raises((ValueError, TypeError)):
        InspectionSettings.from_dict(data)
