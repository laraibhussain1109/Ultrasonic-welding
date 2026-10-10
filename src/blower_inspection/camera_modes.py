"""CPU-only format negotiation and isolated Windows raw-delivery probes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import signal

import cv2


def configure_capture(capture, width, height, fps, fourcc="MJPG", auto_exposure=None):
    """Some drivers reset FOURCC on resolution/FPS changes: request it last too."""
    properties = []
    if fourcc != "DEFAULT":
        if len(fourcc) != 4:
            raise ValueError("Camera FOURCC must be DEFAULT or four characters, such as MJPG")
        code = cv2.VideoWriter_fourcc(*fourcc)
        properties.append(("fourcc_initial", cv2.CAP_PROP_FOURCC, code))
    properties.extend((("width", cv2.CAP_PROP_FRAME_WIDTH, width),
                       ("height", cv2.CAP_PROP_FRAME_HEIGHT, height),
                       ("fps", cv2.CAP_PROP_FPS, fps)))
    if fourcc != "DEFAULT":
        properties.append(("fourcc", cv2.CAP_PROP_FOURCC, code))
    properties.append(("buffer", cv2.CAP_PROP_BUFFERSIZE, 1))
    # 0 has backend-specific meaning and may select a long manual exposure.
    # Leave the driver setting untouched unless explicitly configured by operator.
    if auto_exposure is not None:
        properties.append(("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure))
    return {name: bool(capture.set(prop, value)) for name, prop, value in properties}


def probe_mode(index, backend, width, height, fourcc, fps, auto_exposure=None):
    """A fresh lightweight process bounds driver hangs; no live handle is shared."""
    with tempfile.TemporaryDirectory(prefix="neuroiris-camera-") as folder:
        output = Path(folder) / "probe.json"
        command = [sys.executable, "-m", "blower_inspection.camera_benchmark", "--index", str(index),
                   "--backends", backend, "--formats", fourcc, "--resolutions", f"{width}x{height}",
                   "--fps", str(fps), "--seconds", "2", "--warmup", "5", "--timeout", "8",
                   "--output", str(output)]
        if auto_exposure is not None:
            command.extend(("--auto-exposure", str(auto_exposure)))
        try:
            # The benchmark isolates its driver probe in a child. Its own 8 s
            # timeout releases/terminates that child before this outer deadline.
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                                       start_new_session=sys.platform != "win32")
            try:
                process.communicate(timeout=18)
            except subprocess.TimeoutExpired:
                # Terminate the isolated driver child as well as its launcher.
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   capture_output=True, timeout=5,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.kill()
                process.communicate(timeout=5)
                raise
            if not output.exists():
                return {"status": "error", "error": f"Camera probe exited {process.returncode}"}
            return json.loads(output.read_text(encoding="utf-8"))["results"][0]
        except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, IndexError) as exc:
            return {"status": "error", "error": str(exc)}


def choose_mode(index, width, height, fps, auto_exposure=None, *, probe=probe_mode):
    """Choose by successful native-size reads, never by CAP_PROP_FPS alone."""
    backends = [os.environ["NEUROIRIS_CAMERA_BACKEND"].upper()] if os.environ.get("NEUROIRIS_CAMERA_BACKEND") else ["DSHOW", "MSMF"]
    formats = [os.environ["NEUROIRIS_CAMERA_FOURCC"].upper()] if os.environ.get("NEUROIRIS_CAMERA_FOURCC") else ["MJPG", "DEFAULT"]
    reports, eligible = [], []
    trials = [(backend, fourcc, auto_exposure) for fourcc in formats for backend in backends]
    if auto_exposure is None and not (os.environ.get("NEUROIRIS_CAMERA_BACKEND") and os.environ.get("NEUROIRIS_CAMERA_FOURCC")):
        # Drivers may retain the old manual exposure setting between launches.
        # Compare automatic exposure too; only actual native reads rank a trial.
        for backend in backends:
            if backend in ("DSHOW", "MSMF"):
                trials.append((backend, formats[0], .75 if backend == "DSHOW" else 1.0))
    # Compare compressed modes first; avoid testing uncompressed modes once a
    # measured MJPEG stream is already close to the requested raw baseline.
    for backend, fourcc, exposure in trials:
        report = probe(index, backend, width, height, fourcc, fps, exposure)
        report = {**report, "probe_backend": backend, "probe_fourcc": fourcc, "probe_auto_exposure": exposure}
        reports.append(report)
        actual = report.get("actual", {})
        if (report.get("status") == "ok" and report.get("frames", 0) >= 2
                and (actual.get("width"), actual.get("height")) == (width, height)):
            eligible.append(report)
            if report["delivered_fps"] >= fps * .90 and (fourcc == "DEFAULT" or actual.get("fourcc") == fourcc):
                return report, reports
    if not eligible:
        raise RuntimeError(f"No camera mode delivered the requested {width}x{height} images. "
                           "Run python -m blower_inspection.camera_benchmark; inspection resolution was not reduced.")
    return max(eligible, key=lambda report: report["delivered_fps"]), reports
