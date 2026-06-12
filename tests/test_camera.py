import sys

import pytest
import numpy as np

cv2 = pytest.importorskip("cv2")

from blower_inspection import camera


def test_preferred_capture_backend_uses_any_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")

    assert camera._preferred_capture_backend() == cv2.CAP_ANY


def test_preferred_capture_backend_uses_directshow_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")

    assert camera._preferred_capture_backend() == cv2.CAP_DSHOW


def test_component_roi_bounds_crops_wide_table_to_blower_component():
    frame = np.full((900, 1800, 3), 185, dtype=np.uint8)
    # Unwanted objects in the field of view.
    cv2.rectangle(frame, (720, 0), (1380, 120), (45, 45, 70), -1)  # keyboard
    cv2.rectangle(frame, (150, 230), (1650, 360), (55, 55, 60), -1)  # tooling rail
    # The actual blower fan, matching the lower wide dark part in the camera image.
    cv2.rectangle(frame, (65, 505), (1580, 700), (25, 25, 35), -1)
    for x in range(100, 1550, 95):
        cv2.line(frame, (x, 505), (x + 25, 700), (70, 70, 95), 5)

    x, y, w, h = camera.component_roi_bounds(frame)

    assert x <= 70
    assert y <= 510
    assert x + w >= 1575
    assert y + h >= 695
    assert h < 300
    assert y > 450


def test_crop_component_roi_returns_component_only_from_wide_frame():
    frame = np.full((500, 1000, 3), 190, dtype=np.uint8)
    cv2.rectangle(frame, (20, 300), (920, 405), (20, 20, 30), -1)

    cropped = camera.crop_component_roi(frame)

    assert cropped.shape[1] > 850
    assert cropped.shape[0] < 150
    assert cropped.mean() < frame.mean()


def test_component_roi_bounds_falls_back_to_full_frame_when_no_part_found():
    frame = np.full((120, 240, 3), 180, dtype=np.uint8)

    assert camera.component_roi_bounds(frame) == (0, 0, 240, 120)
