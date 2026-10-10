"""Record six native stationary stills and stage metrics without the desktop.

Uses the selected production profile's quality thresholds and settling rules.
Optional production inference runs on its own thread. No machine outputs are sent.
Run camera_benchmark first to establish the raw driver baseline.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
from PyQt6.QtCore import QCoreApplication

from .camera import USBCamera
from .capture_performance import EvidenceQueue
from .config import ModelRegistry
from .stationary_capture import INSPECTION_ANGLES
from .stationary_workers import StationaryCameraWorker


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="BF-001")
    parser.add_argument("--roi", nargs=4, type=int, required=True, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--continuous", action="store_true", help="Also inspect qualified video frames between required stills")
    parser.add_argument("--output", type=Path, default=Path("camera-pipeline-audit"))
    args = parser.parse_args(argv)
    if args.continuous and not args.inference:
        parser.error("--continuous requires --inference")
    if args.seconds <= 0 or min(args.width, args.height, args.roi[2], args.roi[3]) <= 0:
        parser.error("Duration, resolution and ROI dimensions must be positive")
    if min(args.roi[:2]) < 0 or args.roi[0] + args.roi[2] > args.width or args.roi[1] + args.roi[3] > args.height:
        parser.error("ROI must fit the requested native camera resolution")
    application = QCoreApplication.instance() or QCoreApplication([])
    model = ModelRegistry().get(args.model_id)
    inspector, inference = None, None
    if args.inference:
        from .app import InspectionWorker
        from .inspector_factory import inspector_for_model
        inspector = inspector_for_model(model)
        inspector.validate_ready(model)
    args.output.mkdir(parents=True, exist_ok=True)
    worker = StationaryCameraWorker(USBCamera(args.index, args.width, args.height, model.camera_fps), model, tuple(args.roi), 1)
    captures, inspections, samples, inference_workers, video_checks = [], [], [], [], []
    pending = EvidenceQueue(max_items=12)
    errors = []
    worker.failed.connect(errors.append)
    started, sampled, acquisition_started = time.monotonic(), 0.0, None
    worker.start()
    try:
        while not errors and not worker.error_message:
            application.processEvents()
            snapshot = worker.snapshot()
            now = time.monotonic()
            if snapshot and acquisition_started is None:
                acquisition_started = now
                print("Camera ready: start the fit/home rotation, then six inspection stops", flush=True)
            if acquisition_started is None and now - started > 150:
                errors.append("Camera mode testing/startup exceeded 150 seconds")
                break
            if acquisition_started is not None and now - acquisition_started >= args.seconds:
                break
            if snapshot and (snapshot.frame.shape[1], snapshot.frame.shape[0]) != (args.width, args.height):
                errors.append("Negotiated resolution differs from requested resolution; remeasure and use matching native ROI")
                break
            for _track, capture in worker.take_captures():
                expected = INSPECTION_ANGLES[len(captures)] if len(captures) < 6 else None
                if capture.angle != expected:
                    raise RuntimeError("Captured angles are not in mechanical 60..360 order")
                path = args.output / f"{capture.angle:03d}.png"
                if not cv2.imwrite(str(path), capture.frame):
                    raise RuntimeError(f"Cannot save selected inspection still: {path}")
                captures.append({"angle": capture.angle, "valid": capture.valid, "sharpness": capture.sharpness,
                                 "burst_frames": capture.burst_frames, "reasons": capture.reasons,
                                 "resolution": [capture.frame.shape[1], capture.frame.shape[0]], "image": str(path)})
                print(f"{capture.angle} degrees: valid={capture.valid} sharpness={capture.sharpness:.2f} frames={capture.burst_frames}", flush=True)
                if inspector and capture.valid:
                    pending.append((1, capture))
            if inspector and inference is not None and not inference.isRunning():
                inference = None
            if inspector and inference is None:
                video = None
                if pending:
                    track, capture = pending.popleft()
                    inference = InspectionWorker(inspector, model, track, capture.frame, capture.angle)
                elif args.continuous and len(captures) < 6:
                    video = worker.take_video_evidence()
                    if video is not None:
                        inference = InspectionWorker(inspector, model, video.track_id, video.frame)
                if inference is not None:
                    results = video_checks if video is not None else inspections
                    inference_workers.append(inference)
                    inference.finished_result.connect(lambda _track, result, latency, results=results: results.append(
                        {"angle": result.view_angle, "status": result.status, "valid": result.view_valid,
                         "latency_ms": latency, "reasons": result.reason_codes}))
                    inference.failed.connect(errors.append)
                    inference.start()
            now = time.monotonic()
            if now - sampled >= 1:
                samples.append({"elapsed_s": now - started, **worker.performance()})
                sampled = now
            if len(captures) == 6 and not pending and (inference is None or not inference.isRunning()):
                application.processEvents()
                break
            time.sleep(.01)
    finally:
        worker.requestInterruption()
        # Keep worker references alive through driver release, even on a timeout.
        while not worker.wait(100):
            application.processEvents()
        if inference is not None:
            while not inference.wait(100):
                application.processEvents()
        application.processEvents()
        if worker.error_message and worker.error_message not in errors:
            errors.append(worker.error_message)
        report = {"source": "physical camera pipeline audit", "model_id": model.id,
                  "thresholds": {"settle_ms": model.stationary_settle_ms, "minimum_frames": model.stationary_min_burst_frames,
                                 "minimum_sharpness": max(model.minimum_sharpness, model.inference_min_sharpness),
                                 "max_glare_ratio": model.max_glare_ratio, "max_saturation_ratio": model.max_saturation_ratio},
                  "captures": captures, "inspections": inspections, "video_checks": video_checks, "samples": samples,
                  "startup_seconds": acquisition_started - started if acquisition_started else None,
                  "final": worker.performance(), "errors": errors,
                  "inference_device": inspector.runtime_device_name() if inspector else "inference disabled"}
        report["six_valid_views"] = (len(captures) == 6 and all(c["valid"] for c in captures) and not errors)
        report["inference_complete"] = (not inspector or len(inspections) == sum(c["valid"] for c in captures))
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["six_valid_views"] and report["inference_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
