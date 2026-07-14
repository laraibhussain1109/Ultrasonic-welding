import pytest

pytest.importorskip("cv2", exc_type=ImportError)
import numpy as np

from blower_inspection.trainer import sector_statistics


def test_sector_statistics_ignores_below_visual_fail_threshold():
    mask = np.ones((64, 64), dtype=bool)
    score_map = np.zeros((64, 64), dtype=np.float32)
    score_map[:, 48:] = 0.50

    ratio, sectors = sector_statistics(mask, score_map, 8, min_bad_score=0.65)

    assert ratio == 0.0
    assert sectors == []
