import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", exc_type=ImportError)

from blower_inspection.frame_quality import FrameQualityAnalyzer
from blower_inspection.geometry_inspector import GeometryEvidence
from blower_inspection.inspection_fusion import fuse_patchcore_geometry
from blower_inspection.patchcore_inspector import edge_authority_mask, evenly_limit_indices, filter_duplicate_images
from blower_inspection.patchcore_inspector import PatchCoreInspector
from blower_inspection.roi_stabilizer import ROIStabilizer


def test_yolo_jitter_is_median_stabilized_and_letterboxed():
    stabilizer = ROIStabilizer((200, 80), smoothing_frames=5)
    frame = np.full((120, 320, 3), 80, np.uint8)
    bounds = [stabilizer.canonicalize(frame, (50 + dx, 30, 200, 50), 7).bounds for dx in (0, 2, -2, 1, -1)]
    assert max(item[0] for item in bounds[2:]) - min(item[0] for item in bounds[2:]) <= 2
    assert stabilizer.canonicalize(frame, (50, 30, 200, 50), 7).image.shape[:2] == (80, 200)


def test_edge_authority_suppresses_edge_not_interior():
    authority = edge_authority_mask((100, 200), .05)
    assert authority[0, 100] == 0
    assert authority[50, 100] == 1


def test_blur_is_invalid_and_duplicate_frames_are_reduced():
    blurry = np.full((80, 200, 3), 100, np.uint8)
    assert "MOTION_BLUR" in FrameQualityAnalyzer(10).analyze(blurry).reasons
    detailed = blurry.copy(); detailed[:, ::4] = 255
    assert filter_duplicate_images([detailed, detailed.copy()]) == [0]


def test_similar_repetitive_rotation_is_not_mistaken_for_duplicate():
    first = np.full((80, 200, 3), 90, np.uint8)
    first[:, ::6] = 180
    rotated = np.roll(first, 2, axis=1)

    assert filter_duplicate_images([first, rotated]) == [0, 1]
    assert evenly_limit_indices(1000, 300)[0] == 0
    assert evenly_limit_indices(1000, 300)[-1] == 999


def test_letterbox_pixels_do_not_trigger_underexposure():
    image = np.zeros((100, 200, 3), np.uint8)
    image[25:75] = 90
    image[25:75, ::4] = 180
    valid = np.zeros((100, 200), bool)
    valid[25:75] = True

    quality = FrameQualityAnalyzer(minimum_sharpness=0, max_dark_ratio=.20).analyze(image, valid)

    assert quality.dark_ratio == 0
    assert "UNDEREXPOSED" not in quality.reasons


def _geometry(score=0):
    mask = np.zeros((20, 20), bool)
    return GeometryEvidence(score, score, score, score, score, score, .8, mask,
                            missing_fin_score=score, tilted_fin_score=score)


def test_glare_spike_does_not_fail_but_catastrophic_geometry_does():
    mask = np.ones((20, 20), bool)
    glare = fuse_patchcore_geometry(2, mask, _geometry(), glare_score=1,
                                    candidate_threshold=.7, fail_threshold=1)
    assert glare.status == "PASS" and "LIKELY_GLARE" in glare.reason_codes
    catastrophic = fuse_patchcore_geometry(.1, mask, _geometry(1.2), glare_score=1,
                                           candidate_threshold=.7, fail_threshold=1)
    assert catastrophic.status == "FAIL" and catastrophic.immediate_failure


def test_local_geometry_is_not_averaged_over_full_blower(monkeypatch):
    class FakeInspector:
        def __init__(self, calibration, **_kwargs):
            self.calibration = calibration

        def inspect(self, section):
            score = 1.2 if float(section.mean()) > 100 else 0.0
            mask = np.zeros(section.shape[:2], bool)
            return GeometryEvidence(score, score, 0, score, 0, 0, .8, mask,
                                    missing_fin_score=0, tilted_fin_score=score)

    monkeypatch.setattr("blower_inspection.patchcore_inspector.FinGeometryInspector", FakeInspector)
    image = np.zeros((30, 100, 3), np.uint8)
    image[:, :50] = 255
    config = type("Config", (), {
        "patchcore_section_count": 2,
        "inspection_band_top_ratio": .2,
        "inspection_band_bottom_ratio": .8,
        "geometry_candidate_threshold": .55,
    })()

    evidence = PatchCoreInspector()._inspect_geometry(image, config, [{}, {}])

    assert evidence.score == 1.2
    assert evidence.tilted_fin_score == 1.2
    assert np.any(evidence.defect_mask[:, :50])
    assert not np.any(evidence.defect_mask[:, 50:])
