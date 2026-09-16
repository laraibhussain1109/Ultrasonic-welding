import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", exc_type=ImportError)

from blower_inspection.frame_selection import RotationPhaseGate, SharpFrameSampler


def test_sampler_selects_sharpest_picture_from_each_burst():
    sharp = np.zeros((80, 120, 3), np.uint8)
    sharp[:, ::4] = 255
    blurred = cv2.GaussianBlur(sharp, (21, 21), 8)
    sampler = SharpFrameSampler(burst_size=3, minimum_sharpness=1.0)

    assert sampler.offer(4, blurred) is None
    assert sampler.offer(4, sharp) is None
    selected = sampler.offer(4, blurred)

    assert selected is not None
    assert np.array_equal(selected.frame, sharp)
    assert selected.sharpness > sampler.sharpness(blurred)


def test_sampler_rejects_a_burst_when_every_picture_is_blurry():
    flat = np.full((40, 60, 3), 100, np.uint8)
    sampler = SharpFrameSampler(burst_size=2, minimum_sharpness=1.0)

    assert sampler.offer(9, flat) is None
    assert sampler.offer(9, flat) is None
    assert sampler.rejected_blurry_frames == 2


def test_sampler_does_not_choose_transient_small_zoomed_detector_crop():
    normal = np.zeros((80, 240, 3), np.uint8)
    normal[:, ::8] = 180
    zoomed = np.zeros((35, 60, 3), np.uint8)
    zoomed[:, ::2] = 255
    sampler = SharpFrameSampler(burst_size=3, minimum_sharpness=1.0)

    assert sampler.offer(3, normal) is None
    assert sampler.offer(3, zoomed) is None
    selected = sampler.offer(3, normal)

    assert selected is not None
    assert selected.frame.shape == normal.shape


def test_rotation_phase_gate_rejects_repeated_stationary_view():
    first = np.zeros((80, 240, 3), np.uint8)
    first[:, ::10] = 255
    different = np.zeros_like(first)
    different[::8, :] = 255
    gate = RotationPhaseGate(minimum_distance=.06)

    assert gate.accept(5, first)
    assert not gate.accept(5, first.copy())
    assert gate.accept(5, different)

    gate.discard_missing(set())
    assert gate.accept(5, first)
