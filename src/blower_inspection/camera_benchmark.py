"""Raw OpenCV benchmark: no Qt, optical flow, quality analysis or PyTorch.

Windows drivers expose no universal supported-format enumeration through OpenCV.
Probe requested combinations and report what was actually delivered instead.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import cv2
import numpy as np


def fourcc_name(value):
    code = int(value)
    return "".join(chr((code >> (8 * i)) & 255) for i in range(4)).strip("\x00") or "unknown"


def benchmark_case(index, backend, width, height, fourcc, fps=30, duration=10,
                   warmup=10, auto_exposure=None, *, factory=cv2.VideoCapture, clock=time.perf_counter):
    """Open/read/release on the calling thread; rates count successful reads."""
    report = {"requested": {"index": index, "backend": backend, "width": width,
                            "height": height, "fourcc": fourcc, "fps": fps}, "status": "error"}
    capture = None
    try:
        capture = factory(index, {"DSHOW": cv2.CAP_DSHOW, "MSMF": cv2.CAP_MSMF, "ANY": cv2.CAP_ANY}[backend])
        if not capture.isOpened():
            raise RuntimeError("Camera/backend did not open")
        properties = []
        if fourcc != "DEFAULT":
            properties.append(("fourcc", cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc)))
        if hasattr(cv2, "CAP_PROP_HW_ACCELERATION") and hasattr(cv2, "VIDEO_ACCELERATION_ANY"):
            properties.append(("hw_acceleration", cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY))
        properties.extend((("width", cv2.CAP_PROP_FRAME_WIDTH, width),
                           ("height", cv2.CAP_PROP_FRAME_HEIGHT, height), ("fps", cv2.CAP_PROP_FPS, fps),
                           ("buffer", cv2.CAP_PROP_BUFFERSIZE, 1)))
        if auto_exposure is not None:
            properties.append(("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure))
        report["property_set_accepted"] = {name: bool(capture.set(prop, value)) for name, prop, value in properties}
        for _ in range(warmup):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Warmup read failed")
        report["actual"] = {"backend": capture.getBackendName(),
                            "fourcc": fourcc_name(capture.get(cv2.CAP_PROP_FOURCC)),
                            "reported_width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                            "reported_height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                            "reported_fps": capture.get(cv2.CAP_PROP_FPS),
                            "exposure": capture.get(cv2.CAP_PROP_EXPOSURE),
                            "auto_exposure": capture.get(cv2.CAP_PROP_AUTO_EXPOSURE)}
        start = previous = clock()
        latencies, intervals, frame_positions, timestamps = [], [], [], []
        failures, frames = 0, 0
        while clock() - start < duration:
            before = clock()
            ok, frame = capture.read()
            after = clock()
            latencies.append((after - before) * 1000)
            if not ok or frame is None:
                failures += 1
                # A failing backend must not busy-loop for the entire sample.
                if failures >= 3:
                    break
                continue
            if frames:
                intervals.append(after - previous)
            previous = after
            frames += 1
            report["actual"].update(width=frame.shape[1], height=frame.shape[0])
            # Driver properties are often unsupported (zero); report that explicitly.
            frame_positions.append(capture.get(cv2.CAP_PROP_POS_FRAMES))
            timestamps.append(capture.get(cv2.CAP_PROP_POS_MSEC))
        elapsed = clock() - start
        increasing_positions = len(frame_positions) > 1 and all(b > a for a, b in zip(frame_positions, frame_positions[1:]))
        increasing_timestamps = len(timestamps) > 1 and all(b > a for a, b in zip(timestamps, timestamps[1:]))
        report.update(status="ok" if frames and failures == 0 else "read_failed", frames=frames,
                      elapsed_s=elapsed, delivered_fps=frames / max(elapsed, 1e-9), read_failures=failures,
                      read_latency_ms={"mean": float(np.mean(latencies)) if latencies else 0,
                                       "p50": float(np.percentile(latencies, 50)) if latencies else 0,
                                       "p95": float(np.percentile(latencies, 95)) if latencies else 0,
                                       "max": max(latencies, default=0)},
                      dropped_frame_evidence={
                          "delivery_gaps_over_1_5_requested_period": sum(interval > 1.5 / fps for interval in intervals),
                          "max_delivery_gap_ms": 1000 * max(intervals, default=0),
                          "driver_sequence_available": increasing_positions,
                          "driver_sequence_gaps": sum(max(0, round(b - a) - 1) for a, b in zip(frame_positions, frame_positions[1:])) if increasing_positions else None,
                          "driver_timestamps_available": increasing_timestamps,
                          "note": "Delivery gaps/read failures are evidence, not proof of sensor drops. No exact drop count without driver sequence/timestamps."})
        actual = report["actual"]
        report["resolution_matches_request"] = (actual.get("width"), actual.get("height")) == (width, height)
        report["format_matches_request"] = actual["fourcc"] == fourcc if fourcc != "DEFAULT" else None
    except Exception as exc:
        report["error"] = str(exc)
    finally:
        if capture is not None:
            capture.release()
    return report


def _child(connection, kwargs):
    try:
        connection.send(benchmark_case(**kwargs))
    finally:
        connection.close()


def isolated_case(kwargs, timeout):
    """Unsupported Windows formats can hang inside a driver; isolate each probe."""
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_child, args=(child, kwargs))
    process.start()
    child.close()
    try:
        if parent.poll(timeout):
            try:
                return parent.recv()
            except EOFError:
                return {"requested": kwargs, "status": "process_failed", "exit_code": process.exitcode}
        return {"requested": kwargs, "status": "timeout", "error": "Driver/open/read exceeded probe timeout; no delivered FPS assumed"}
    finally:
        if process.is_alive():
            process.terminate()
        process.join(5)
        parent.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--backends", nargs="+", choices=["DSHOW", "MSMF", "ANY"], default=["DSHOW", "MSMF"])
    parser.add_argument("--formats", nargs="+", default=["DEFAULT", "MJPG", "YUY2", "NV12"])
    parser.add_argument("--resolutions", nargs="+", default=["1920x1080", "3840x2160"])
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--auto-exposure", type=float, default=None, help="Leave driver default unless specified; production currently requests 0")
    parser.add_argument("--output", type=Path, default=Path("camera-benchmark.json"))
    args = parser.parse_args(argv)
    if args.fps <= 0 or args.seconds <= 0 or args.timeout <= 0 or args.warmup < 0:
        parser.error("FPS, duration and timeout must be positive; warmup must be nonnegative")
    if any(fmt != "DEFAULT" and len(fmt) != 4 for fmt in args.formats):
        parser.error("Formats must be DEFAULT or a four-character FOURCC (MJPG means MJPEG)")
    resolutions = []
    for size in args.resolutions:
        try:
            w, h = map(int, size.lower().split("x"))
            if w <= 0 or h <= 0:
                raise ValueError()
            resolutions.append((w, h))
        except ValueError:
            parser.error("Resolutions must be WIDTHxHEIGHT")
    reports = []
    for backend in args.backends:
        for width, height in resolutions:
            for fmt in args.formats:
                kwargs = dict(index=args.index, backend=backend, width=width, height=height,
                              fourcc=fmt, fps=args.fps, duration=args.seconds, warmup=args.warmup,
                              auto_exposure=args.auto_exposure)
                report = isolated_case(kwargs, max(args.timeout, args.seconds + 15))
                reports.append(report)
                actual = report.get("actual", {})
                measured_fps = f"{report['delivered_fps']:.2f}" if "delivered_fps" in report else "unmeasured"
                read_p95 = f"{report['read_latency_ms']['p95']:.1f}" if "read_latency_ms" in report else "unmeasured"
                print(f"{backend} {width}x{height} {fmt}: {report['status']} | actual "
                      f"{actual.get('width', '?')}x{actual.get('height', '?')} {actual.get('fourcc', '?')} "
                      f"{actual.get('backend', '?')} | delivered {measured_fps} FPS "
                      f"| read p95 {read_p95} ms", flush=True)
                args.output.write_text(json.dumps({"opencv": cv2.__version__, "platform": sys.platform,
                                                  "pipeline": "raw OpenCV only; no Qt/torch/quality/motion",
                                                  "results": reports}, indent=2), encoding="utf-8")
    return 0 if any(r["status"] == "ok" for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
