"""Profile stages separately on a saved camera still or explicit synthetic data.

Run camera_benchmark first for the physical raw baseline. This offline profile
does not certify acquisition FPS or mechanical stop capture.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .camera import crop_bounds
from .capture_performance import preview_image
from .frame_quality import FrameQualityAnalyzer
from .stationary_capture import StationaryViewCapture


def profile(frame, bounds, iterations=30, legacy=False, inspect=None):
    from PyQt6.QtGui import QImage, QPixmap
    from PyQt6.QtWidgets import QApplication
    application = QApplication.instance() or QApplication([])
    roi = crop_bounds(frame, bounds)
    controller, quality = StationaryViewCapture(), FrameQualityAnalyzer()
    previous = None
    def motion():
        nonlocal previous
        if legacy:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            band = gray[round(h * .18):max(round(h * .82), round(h * .18) + 1)]
            reduced = cv2.GaussianBlur(cv2.resize(band, (min(320, max(80, w)), 64), interpolation=cv2.INTER_AREA), (3, 3), 0)
            before, previous = previous, reduced
            if before is None:
                return False
            delta = reduced.astype(np.float32) - before.astype(np.float32)
            difference = float(np.mean(np.abs(delta - np.median(delta))))
            flow = cv2.calcOpticalFlowFarneback(before, reduced, None, .5, 3, 15, 2, 5, 1.2, 0)
            movement = float(np.quantile(cv2.magnitude(flow[..., 0], flow[..., 1]), .80))
            return difference >= controller.motion_threshold or movement >= controller.flow_threshold
        else:
            reduced = controller._motion_image(roi)
        return controller._moving(reduced)
    def preview():
        display = frame.copy() if legacy else preview_image(frame, 1100, 620)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB).copy()
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
        return QPixmap.fromImage(image).scaled(1100, 620)
    stages = {"roi_copy": lambda: crop_bounds(frame, bounds), "motion": motion,
              "quality": lambda: quality.analyze(roi), "preview_qt": preview}
    if inspect:
        stages["patchcore_inspection"] = lambda: inspect(roi)
    result = {"pipeline": "legacy serial stages" if legacy else "separate stages; bounded threaded acquisition",
              "image_resolution": [frame.shape[1], frame.shape[0]], "roi": list(bounds),
              "iterations": iterations, "opencv_threads": cv2.getNumThreads(), "stages": {}}
    for name, operation in stages.items():
        for _ in range(3): operation()
        samples = []
        for _ in range(iterations):
            start = time.perf_counter()
            operation()
            samples.append((time.perf_counter() - start) * 1000)
        result["stages"][name] = {"mean_ms": float(np.mean(samples)), "p95_ms": float(np.percentile(samples, 95))}
    # Retain QApplication throughout Qt profiling.
    assert application is not None
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"), default=(307, 799, 2957, 475))
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--opencv-threads", type=int, help="Control diagnostic CPU parallelism for comparable stage measurements")
    parser.add_argument("--legacy", action="store_true", help="Reproduce pre-change motion and full-resolution Qt stages")
    parser.add_argument("--model-id", help="Optionally profile the installed, calibrated production model; does not train or modify it")
    parser.add_argument("--output", type=Path, default=Path("pipeline-profile.json"))
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error("iterations must be positive")
    if args.opencv_threads is not None:
        if args.opencv_threads < 1:
            parser.error("opencv-threads must be positive")
        cv2.setNumThreads(args.opencv_threads)
    frame = (np.random.default_rng(72).integers(40, 180, (2160, 3840, 3), dtype=np.uint8)
             if args.synthetic else cv2.imread(str(args.image)))
    if frame is None:
        parser.error("Cannot read camera still")
    inspect, device = None, "not measured (no production model supplied)"
    if args.model_id:
        from .config import ModelRegistry
        from .inspector_factory import inspector_for_model
        model = ModelRegistry().get(args.model_id)
        inspector = inspector_for_model(model)
        inspector.validate_ready(model)
        device = inspector.runtime_device_name()
        inspect = lambda roi: inspector.inspect(model, roi, save_outputs=False, crop_to_component=False)
    result = profile(frame, tuple(args.roi), args.iterations, args.legacy, inspect)
    result.update(source="synthetic; no physical camera" if args.synthetic else str(args.image), inference_device=device)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
