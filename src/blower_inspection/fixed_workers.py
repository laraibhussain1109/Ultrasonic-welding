"""Camera, inspection, and disk workers that keep Qt responsive."""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor

import cv2
from PyQt6.QtCore import QThread, pyqtSignal

from .camera import USBCamera
from .fail_output import ESP32FailOutputBridge
from .fixed_settings import InspectionSettings
from .fixed_views import FixedViewController, FixedViewPipeline
from .inspection_storage import ResultStorage
from .machine_signals import MachineEvent, TCPMachineConnection
from .stationary_roi import InvalidView


class CameraWorker(QThread):
    preview = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, settings: InspectionSettings) -> None:
        super().__init__()
        self.settings = settings
        self.lock = threading.Lock()
        self.latest = None
        self.sequence = 0

    def frame_snapshot(self):
        with self.lock:
            return self.latest

    def run(self) -> None:
        c = self.settings.camera
        camera = USBCamera(c.index, c.width, c.height, c.fps)
        try:
            camera.open()
            camera.capture.set(cv2.CAP_PROP_EXPOSURE, c.exposure)
            camera.capture.set(cv2.CAP_PROP_GAIN, c.gain)
            preview_at = 0.0
            while not self.isInterruptionRequested():
                # Start timestamp is conservative: a buffered pre-stop exposure
                # cannot be counted simply because read() completed after stop.
                captured_at = time.monotonic()
                frame = camera.read()
                with self.lock:
                    self.sequence += 1
                    self.latest = (self.sequence, captured_at, frame)
                if captured_at - preview_at >= .1:
                    self.preview.emit(frame)
                    preview_at = captured_at
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            camera.close()


class SessionWorker(QThread):
    status = pyqtSignal(str, object)
    view_ready = pyqtSignal(object)
    part_ready = pyqtSignal(object)
    failed = pyqtSignal(str)
    ready = pyqtSignal()

    def __init__(self, settings: InspectionSettings, camera: CameraWorker) -> None:
        super().__init__()
        self.settings, self.camera = settings, camera
        self.events = queue.Queue(maxsize=100)

    def send_event(self, event: MachineEvent) -> None:
        if self.settings.machine.interface != "manual":
            raise ValueError("Manual controls are disabled with PLC input enabled")
        self.events.put_nowait(event)

    def run(self) -> None:
        controller = FixedViewController(self.settings)
        pipeline = FixedViewPipeline(self.settings)
        storage = ResultStorage(self.settings)
        connection = None
        inference = None
        epoch, job_epoch = 0, -1
        pending_save = None
        output_futures = []
        last_status = None
        last_sequence = -1
        last_signal = time.monotonic()
        bridge = ESP32FailOutputBridge()
        bridge.config = replace(bridge.config, enabled=self.settings.machine.output_enabled and bridge.config.enabled)
        model_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="patchcore-inference")
        disk_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inspection-storage")
        def output(future):
            if future is not None:
                output_futures.append(future)
        def fault(reason):
            controller.fault(reason)
            output(bridge.send_result(True))
            if controller.part is not None and storage.directory is not None:
                storage.update_part(controller.part)
            self.status.emit(controller.state, controller.part.metadata() if controller.part else {})
            self.failed.emit(reason)
        try:
            pipeline.ready()
            if self.settings.machine.interface == "tcp_json":
                connection = TCPMachineConnection(self.settings.machine.host, self.settings.machine.port)
                last_signal = time.monotonic()
            self.ready.emit()
            while not self.isInterruptionRequested():
                now = time.monotonic()
                events = []
                if connection:
                    events.extend(connection.poll())
                    if events:
                        last_signal = now
                    if controller.part is not None and controller.part.final_verdict == "INCOMPLETE" and now - last_signal > self.settings.machine.signal_timeout_s:
                        raise ConnectionError("PLC heartbeat lost; stopped state is no longer confirmed")
                while not self.events.empty():
                    events.append(self.events.get_nowait())
                for event in events:
                    if event.kind == "begin":
                        if pending_save is not None:
                            raise ValueError("Previous inspection is still saving")
                        epoch += 1
                        controller.begin(event.part_id, fit_check_complete=event.fit_check_complete, now=now)
                        storage.begin(controller.part)
                        output(bridge.reset())
                    elif event.kind == "stopped":
                        controller.stopped(event.angle, now=now)
                    elif event.kind == "moving":
                        epoch += 1
                        controller.moving()
                    elif event.kind == "abort":
                        raise RuntimeError("Machine/operator aborted inspection")
                for future in list(output_futures):
                    if future.done():
                        future.result()  # Failed output is a visible equipment fault.
                        output_futures.remove(future)
                if pending_save is not None and pending_save.done():
                    pending_save.result()
                    pending_save = None
                    part = controller.part
                    if connection:
                        connection.acknowledge({"event": "view_result", "angle": saved_view.angle,
                                                "verdict": saved_view.verdict, "view_count": len(part.views)})
                    self.view_ready.emit(saved_view)
                    if part.final_verdict in {"PASS", "FAIL"}:
                        if connection:
                            connection.acknowledge({"event": "part_result", **part.metadata()})
                        output(bridge.signal_pass() if part.final_verdict == "PASS" else bridge.send_result(True))
                        self.part_ready.emit(part.metadata())
                if inference is not None and inference.done():
                    try:
                        view = inference.result()
                        if epoch == job_epoch and controller.pending:
                            controller.accept(view, now=time.monotonic())
                            if view.verdict == "FAIL":
                                output(bridge.send_result(True))
                            saved_view = view
                            def save(view=view, part=controller.part):
                                storage.save_view(view)
                                storage.update_part(part)
                            pending_save = disk_pool.submit(save)
                    except InvalidView as exc:
                        if epoch == job_epoch:
                            controller.invalid(str(exc))
                    inference = None
                controller.tick(time.monotonic())
                if controller.part and controller.part.fault:
                    raise RuntimeError(controller.part.fault)
                sample = self.camera.frame_snapshot()
                if sample is not None and sample[0] != last_sequence and inference is None and pending_save is None:
                    last_sequence, captured_at, frame = sample
                    if controller.frame(frame, captured_at):
                        frames, angle = list(controller.burst), controller.angle
                        job_epoch = epoch
                        inference = model_pool.submit(pipeline.inspect_frames, frames, angle)
                status = (controller.state, len(controller.part.views) if controller.part else 0)
                if status != last_status:
                    self.status.emit(controller.state, controller.part.metadata() if controller.part else {})
                    last_status = status
                self.msleep(5)
            if controller.part is not None and controller.part.final_verdict == "INCOMPLETE":
                fault("Inspection stopped with fewer than six valid views")
        except Exception as exc:
            try:
                fault(str(exc))
            except Exception as storage_exc:
                self.failed.emit(f"{exc}; logging failed: {storage_exc}")
        finally:
            if connection:
                connection.close()
            model_pool.shutdown(wait=True, cancel_futures=True)
            disk_pool.shutdown(wait=True, cancel_futures=True)
            for future in output_futures:
                try:
                    future.result(timeout=2.0)
                except Exception as exc:
                    self.failed.emit(f"Pending ESP32 output failed during shutdown: {exc}")
            bridge.close()


class TaskWorker(QThread):
    result = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, function) -> None:
        super().__init__()
        self.function = function

    def run(self) -> None:
        try:
            self.result.emit(self.function(self.progress.emit))
        except Exception as exc:
            self.failed.emit(str(exc))
