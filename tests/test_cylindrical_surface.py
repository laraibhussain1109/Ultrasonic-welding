import pytest
import numpy as np

cv2 = pytest.importorskip("cv2", exc_type=ImportError)

from blower_inspection.trainer import (
    broken_fin_mask,
    cylindrical_sector_statistics,
    cylindrical_surface_mask,
    inspection_overlay,
)


def fin_image(*, broken=False):
    image = np.full((120, 600, 3), 35, dtype=np.uint8)
    for y in range(20, 105, 12):
        cv2.line(image, (10, y), (590, y), (175, 175, 175), 2)
        if broken and y == 56:
            cv2.rectangle(image, (150, y - 2), (170, y + 2), (35, 35, 35), -1)
    for x in (100, 250, 400, 550):
        cv2.line(image, (x, 10), (x, 110), (120, 120, 120), 4)
    return image


def test_surface_mask_includes_center_and_excludes_crop_edges():
    mask = cylindrical_surface_mask((100, 300))

    assert mask[50, 150]
    assert not mask[0, 150]
    assert not mask[50, 0]


def test_overlay_leaves_normal_pixels_unmodified_and_marks_only_anomaly():
    image = np.full((100, 300, 3), 80, dtype=np.uint8)
    scores = np.zeros((50, 50), dtype=np.float32)
    scores[20:25, 30:35] = 1.0
    defects = scores >= 0.65

    overlay, boxes = inspection_overlay(
        image, scores, defects, status="FAIL", anomaly_score=1.0, min_box_area_px=5, score_normalizer=0.65
    )

    assert np.array_equal(overlay[10, 10], image[10, 10])
    assert not np.array_equal(overlay[45, 195], image[45, 195])
    assert boxes


def test_cylindrical_sectors_find_a_localized_surface_defect():
    mask = cylindrical_surface_mask((100, 240))
    scores = np.full((100, 240), 0.1, dtype=np.float32)
    scores[20:80, 100:110] = 0.95

    ratio, sectors = cylindrical_sector_statistics(mask, scores, 24, min_bad_score=0.65)

    assert ratio > 0
    assert sectors == [10]


def test_broken_fin_detector_marks_small_horizontal_discontinuity():
    detected = broken_fin_mask(fin_image(broken=True), (120, 600))

    assert np.count_nonzero(detected[50:63, 145:175]) > 0


def test_broken_fin_detector_does_not_mark_continuous_fins():
    detected = broken_fin_mask(fin_image(), (120, 600))

    assert np.count_nonzero(detected) == 0
