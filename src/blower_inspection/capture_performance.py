"""Small CPU-only counters and bounded FIFO storage for camera evidence."""
from collections import deque
import threading
import time


class CaptureQueueFull(RuntimeError):
    """Losing ordered inspection evidence requires an inspection fault."""


class EvidenceQueue:
    """Never overwrite inspection evidence; cap both item count and image bytes."""

    def __init__(self, max_items=16, max_bytes=256 * 1024 * 1024):
        self.max_items, self.max_bytes = max_items, max_bytes
        self._items = deque()
        self._bytes = 0
        self.high_water = 0
        self._lock = threading.Lock()

    @staticmethod
    def _size(item):
        if hasattr(item, "nbytes"):
            return item.nbytes
        image = item.frame if hasattr(item, "frame") else item[1].frame
        return image.nbytes

    def append(self, item):
        size = self._size(item)
        with self._lock:
            if len(self._items) >= self.max_items or self._bytes + size > self.max_bytes:
                raise CaptureQueueFull("Ordered camera evidence queue full; inspection must stop (no frames silently discarded)")
            self._items.append((item, size))
            self._bytes += size
            self.high_water = max(self.high_water, len(self._items))

    def popleft(self):
        with self._lock:
            item, size = self._items.popleft()
            self._bytes -= size
            return item

    def clear(self):
        with self._lock:
            self._items.clear()
            self._bytes = 0

    def __len__(self):
        with self._lock:
            return len(self._items)

    def __iter__(self):
        with self._lock:
            return iter([item for item, _size in self._items])

    def __getitem__(self, index):
        with self._lock:
            return self._items[index][0]


class StageMeter:
    """Rates use actual completed operations and a rolling wall-clock window."""

    def __init__(self):
        self._times = deque(maxlen=120)
        self._latencies = deque(maxlen=120)
        self.count = 0
        self._lock = threading.Lock()

    def record(self, duration_s=0.0, now=None):
        with self._lock:
            self._times.append(time.monotonic() if now is None else now)
            self._latencies.append(duration_s * 1000)
            self.count += 1

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        with self._lock:
            times, latency, count = list(self._times), sorted(self._latencies), self.count
        fps = (len(times) - 1) / max(now - times[0], .001) if len(times) > 1 else 0.0
        return {"fps": fps, "count": count,
                "mean_ms": sum(latency) / len(latency) if latency else 0.0,
                "p95_ms": latency[min(len(latency) - 1, int(.95 * len(latency)))] if latency else 0.0}


def preview_image(frame, width, height):
    """Resize before drawing, color conversion, QImage or QPixmap allocation."""
    import cv2
    h, w = frame.shape[:2]
    scale = min(1.0, max(1, width) / w, max(1, height) / h)
    if scale == 1.0:
        return frame.copy()
    return cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_LINEAR)
