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
