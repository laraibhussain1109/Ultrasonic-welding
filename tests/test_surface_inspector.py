import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", exc_type=ImportError)

from blower_inspection.surface_inspector import (
    generate_tiles, map_roi_box_to_frame, merge_tile_maps,
    score_anomaly_map, tile_positions,
)


def test_native_tile_generation_covers_boundaries_with_overlap():
    image = np.zeros((900, 3200, 3), np.uint8)
    tiles = generate_tiles(image, 768, .25)
    assert tile_positions(3200, 768, .25)[-1] == 3200 - 768
    assert tile_positions(900, 768, .25)[-1] == 900 - 768
    assert all(tile.image.shape[:2] == (768, 768) for tile in tiles)
    assert max(tile.x + tile.valid_width for tile in tiles) == 3200
    assert max(tile.y + tile.valid_height for tile in tiles) == 900


def test_weighted_merge_has_no_tile_seams_for_constant_evidence():
    image = np.zeros((913, 2017, 3), np.uint8)
    tiles = generate_tiles(image, 768, .25)
    merged = merge_tile_maps([np.ones((54, 54), np.float32) for _ in tiles], tiles, image.shape[:2])
    assert merged.shape == image.shape[:2]
    assert np.allclose(merged, 1.0, atol=1e-5)


def test_minute_extreme_anomaly_survives_global_percentile():
    anomaly = np.zeros((1000, 1000), np.float32)
    anomaly[500:502, 600:602] = 12
    stats, mask = score_anomaly_map(anomaly, component_threshold=5)
    assert stats.global_score == 0
    assert stats.peak_score == 12
    assert stats.topk_score > 0
    assert stats.components[0].area == 4
    assert mask.sum() == 4


def test_large_moderate_anomaly_and_normal_map_scoring():
    normal_stats, normal_mask = score_anomaly_map(np.zeros((200, 300), np.float32), component_threshold=1)
    anomaly = np.zeros((200, 300), np.float32)
    anomaly[40:140, 80:220] = 2
    stats, mask = score_anomaly_map(anomaly, component_threshold=1)
    assert normal_stats.peak_score == 0 and not normal_mask.any()
    assert stats.global_score == 2 and mask.sum() == 14000


def test_soft_authority_reduces_but_does_not_erase_rib_anomaly():
    anomaly = np.zeros((20, 20), np.float32)
    anomaly[10, 10] = 10
    authority = np.ones_like(anomaly)
    authority[10, 10] = .5
    stats, mask = score_anomaly_map(anomaly, authority, component_threshold=4)
    assert stats.peak_score == 5
    assert mask[10, 10]


def test_native_roi_coordinate_mapping_is_translation_only():
    assert map_roi_box_to_frame((25, 15, 20, 10), (100, 200, 3200, 900)) == (125, 215, 20, 10)
