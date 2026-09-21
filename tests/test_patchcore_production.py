import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", exc_type=ImportError)

from blower_inspection.frame_quality import FrameQualityAnalyzer
from blower_inspection.geometry_inspector import GeometryEvidence
from blower_inspection.inspection_fusion import fuse_patchcore_geometry
from blower_inspection.patchcore_inspector import edge_authority_mask, filter_duplicate_images
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
