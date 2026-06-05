import sys

import pytest

cv2 = pytest.importorskip("cv2")

from blower_inspection import camera


def test_preferred_capture_backend_uses_any_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")

    assert camera._preferred_capture_backend() == cv2.CAP_ANY


def test_preferred_capture_backend_uses_directshow_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")

    assert camera._preferred_capture_backend() == cv2.CAP_DSHOW
