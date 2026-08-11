import pytest
import numpy as np

cv2 = pytest.importorskip("cv2", exc_type=ImportError)

from blower_inspection.trainer import (
    cylindrical_sector_statistics,
    cylindrical_surface_mask,
    inspection_overlay,
)


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
