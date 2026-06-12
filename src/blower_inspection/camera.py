"""OpenCV USB camera access and capture helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np


def _preferred_capture_backend() -> int:
    """Return an OpenCV backend suitable for the current operating system."""
    if sys.platform.startswith("win") and hasattr(cv2, "CAP_DSHOW"):
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def component_roi_bounds(image: np.ndarray, *, padding_ratio: float = 0.035) -> tuple[int, int, int, int]:
    """Return ``(x, y, w, h)`` bounds for the blower component inside a wide camera frame.

    The production camera sees the whole work table, but only the long dark blower
    fan should be inspected. This detector intentionally favours wide, dark,
    horizontally-oriented regions and adds a small margin so fixtures/table clutter
    are removed before the anomaly model sees the frame.
    """
    if image.size == 0:
        raise ValueError("Cannot crop an empty image")

    height, width = image.shape[:2]
    if height < 4 or width < 4:
        return 0, 0, width, height

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # The inspected part is the dominant dark horizontal object. Use both Otsu and
    # an absolute cap so very bright glare on the table does not raise the threshold
    # enough to include the full background.
    otsu_threshold, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    dark_threshold = min(max(int(otsu_threshold), 55), 120)
    dark_mask = (blurred <= dark_threshold).astype(np.uint8) * 255

    close_w = max(31, (width // 35) | 1)
    close_h = max(9, (height // 85) | 1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    open_w = max(7, (width // 220) | 1)
    open_h = max(3, (height // 220) | 1)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_w, open_h))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    frame_area = float(width * height)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        if w < width * 0.25 or h < height * 0.04 or area < frame_area * 0.01:
            continue
        aspect = w / max(h, 1)
        if aspect < 2.0:
            continue
        vertical_center = (y + h / 2.0) / height
        lower_half_bonus = 1.35 if vertical_center >= 0.45 else 1.0
        width_bonus = 1.0 + (w / width)
        score = area * aspect * lower_half_bonus * width_bonus
        candidates.append((score, (x, y, w, h)))

    if not candidates:
        return 0, 0, width, height

    _score, (x, y, w, h) = max(candidates, key=lambda item: item[0])
    pad_x = max(4, int(w * padding_ratio))
    pad_y = max(4, int(h * padding_ratio))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(width, x + w + pad_x)
    y1 = min(height, y + h + pad_y)
    return x0, y0, x1 - x0, y1 - y0


def crop_component_roi(image: np.ndarray, *, padding_ratio: float = 0.035) -> np.ndarray:
    """Crop a wide camera frame down to the blower component inspection ROI."""
    x, y, w, h = component_roi_bounds(image, padding_ratio=padding_ratio)
    return image[y : y + h, x : x + w].copy()


class USBCamera:
    """Small wrapper around OpenCV VideoCapture for an 8.3 MP USB3 camera."""

    def __init__(self, index: int = 0, width: int = 3840, height: int = 2160, fps: int = 30) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        self.capture = cv2.VideoCapture(self.index, _preferred_capture_backend())
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open camera index {self.index}")
        if hasattr(cv2, "CAP_PROP_FOURCC"):
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)

    def read(self) -> np.ndarray:
        if self.capture is None:
            self.open()
        assert self.capture is not None
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError("Camera frame capture failed")
        return frame

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


def save_capture(image: np.ndarray, directory: str | Path, prefix: str = "capture") -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = directory / f"{prefix}_{stamp}.png"
    cv2.imwrite(str(path), image)
    return path
