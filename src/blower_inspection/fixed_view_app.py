"""Stationary six-view production, engineering, settings, and calibration UI."""
from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPushButton, QScrollArea, QSpinBox, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget)

from .anomaly_tolerance import heatmap_image, marked_image
from .auth import AuthStore, User
from .fixed_settings import InspectionSettings, ToleranceSettings
from .fixed_views import ANGLES, FixedViewPipeline, ViewInspectionResult
from .fixed_workers import CameraWorker, SessionWorker, TaskWorker
from .inspection_storage import ResultStorage, export_history, load_saved_view
from .machine_signals import MachineEvent
from .patchcore_spatial import SpatialPatchCore
from .tolerance_calibration import CalibrationSample, evaluate_samples

STYLE = """
QWidget { background: #101923; color: #d8e3ef; font-family: 'Segoe UI', Arial; font-size: 13px; }
QTabWidget::pane, QGroupBox { border: 1px solid #324659; border-radius: 5px; }
QGroupBox { margin-top: 16px; padding-top: 10px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; }
QPushButton { padding: 9px 14px; border: 1px solid #45637c; border-radius: 4px; background: #22364a; }
QPushButton:hover { background: #2c4f6c; } QPushButton:disabled { color: #657788; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit { background: #182634; padding: 5px; border: 1px solid #3a5064; }
QTableWidget { background: #162431; gridline-color: #334c60; }
QHeaderView::section { background: #243b50; padding: 7px; }
QTabBar::tab { padding: 12px 18px; background: #1c2c3d; }
QTabBar::tab:selected { background: #2a4f68; color: #65dfd0; }
"""


class ImageViewer(QLabel):
    def __init__(self):
        super().__init__("NO IMAGE")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(480, 300)
        self.frame = None

    def set_frame(self, frame):
        self.frame = frame
        self.render()

    def render(self):
        if self.frame is None:
            return
        frame = self.frame
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.setPixmap(QPixmap.fromImage(image).scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                       Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.render()


def spin(value, *, integer=False, maximum=1e9, minimum=0.0):
    widget = QSpinBox() if integer else QDoubleSpinBox()
    if integer:
        widget.setRange(int(minimum), int(min(maximum, 2_000_000_000)))
    else:
        widget.setDecimals(5)
        widget.setRange(float(minimum), float(maximum))
        widget.setSingleStep(.01)
    widget.setValue(value)
    return widget


class ToleranceControls(QWidget):
    changed = None

    def __init__(self, settings: ToleranceSettings, callback):
        super().__init__()
        self.inputs = {}
        self.callback = callback
        layout = QFormLayout(self)
        labels = {"heatmap_threshold": "1. Heatmap threshold (raw distance)",
                  "minimum_component_pixels": "2. Minimum region (px)", "pixel_threshold": "3. Pixel tolerance (px)",
                  "percentage_threshold": "3. Percentage tolerance (%)", "decision_mode": "Decision mode",
                  "ignore_top_percent": "Ignore top (%)", "ignore_bottom_percent": "Ignore bottom (%)",
                  "ignore_left_percent": "Ignore left (%)", "ignore_right_percent": "Ignore right (%)",
                  "opening_kernel": "Opening kernel (0 = OFF)", "closing_kernel": "Closing kernel (0 = OFF)"}
        for name, value in asdict(settings).items():
            if name == "decision_mode":
                widget = QComboBox()
                widget.addItems(["PIXEL", "PERCENTAGE", "BOTH", "EITHER"])
                widget.setCurrentText(value)
                widget.currentTextChanged.connect(callback)
            elif name in {"opening_kernel", "closing_kernel"}:
                widget = QComboBox()
                widget.addItems(["0", "3", "5"])
                widget.setCurrentText(str(value))
                widget.currentTextChanged.connect(callback)
            else:
                integer = isinstance(value, int)
                maximum = 100 if name.endswith("percent") or name == "percentage_threshold" else 2e9
                minimum = 1 if name in {"pixel_threshold", "minimum_component_pixels"} else .00001 if name == "percentage_threshold" else 0
                widget = spin(value, integer=integer, maximum=maximum, minimum=minimum)
                widget.valueChanged.connect(callback)
            self.inputs[name] = widget
            layout.addRow(labels[name], widget)

    def value(self) -> ToleranceSettings:
        result = {}
        for name, widget in self.inputs.items():
            result[name] = (int(widget.currentText()) if name.endswith("kernel") else widget.currentText()) if isinstance(widget, QComboBox) else widget.value()
        tolerance = ToleranceSettings(**result)
        InspectionSettings(tolerance=tolerance).validate()
        return tolerance


class FixedInspectionWindow(QMainWindow):
    def __init__(self, user: User, settings_path: str | Path = "config/fixed_inspection.json") -> None:
        super().__init__()
        self.user, self.settings_path = user, Path(settings_path)
        self.settings = InspectionSettings.load(settings_path)
        self.camera = None
        self.session = None
        self.session_is_ready = False
        self.tasks = []
        self.view_results = {}
        self.current_view = None
        self.part_metadata = {}
        self.samples = []
        self.calibration_rows = []
        self.calibration_errors = []
        self.cycle_started_at = None
        self.calibration_worker = None
        self.calibration_revision = 0
        self.setWindowTitle("NeuroIris • Fixed 6-view PatchCore inspection")
        self.resize(1520, 980)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self._production_page()
        self._engineering_page()
        self._settings_page()
        self._calibration_page()
        self.tabs.setTabEnabled(1, user.is_admin)
        self.tabs.setTabEnabled(2, user.is_admin)
        self.tabs.setTabEnabled(3, user.is_admin)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._cycle_time)
        self.timer.start(100)
        self.calibration_debounce = QTimer(self)
        self.calibration_debounce.setSingleShot(True)
        self.calibration_debounce.timeout.connect(self._calculate_calibration)
        self.statusBar().showMessage("Manual commissioning • hardware output disabled" if self.settings.machine.interface == "manual" else "PLC bridge configured • waiting for connection")

    def button(self, text, callback, layout):
        button = QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def _production_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        controls = QHBoxLayout()
        self.camera_button = self.button("OPEN CAMERA", self.open_camera, controls)
        self.start_button = self.button("ARM INSPECTION", self.arm, controls)
        self.stop_button = self.button("STOP / ABORT", self.stop, controls)
        self.button("EXPORT HISTORY CSV", self.export_csv, controls)
        layout.addLayout(controls)
        manual = QGroupBox("Manual commissioning — confirm the actual machine position")
        line = QHBoxLayout(manual)
        self.part_id = QLineEdit()
        self.part_id.setPlaceholderText("Part ID (automatic if blank)")
        self.fit_check = QCheckBox("PLC 360° fitting check completed")
        line.addWidget(self.part_id)
        line.addWidget(self.fit_check)
        self.begin_button = self.button("NEW PART", self.begin_part, line)
        self.moving_button = self.button("MOTOR MOVING", lambda: self.machine_event(MachineEvent("moving")), line)
        self.stopped_button = self.button("CONFIRM NEXT STOP", self.confirm_stop, line)
        manual.setVisible(self.settings.machine.interface == "manual")
        self.manual_box = manual
        layout.addWidget(manual)
        split = QSplitter()
        self.production_viewer = ImageViewer()
        split.addWidget(self.production_viewer)
        side = QWidget()
        info = QVBoxLayout(side)
        self.result_badge = QLabel("STANDBY")
        self.result_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_badge.setStyleSheet("font-size: 36px; font-weight: bold; padding: 25px; background: #233344")
        info.addWidget(self.result_badge)
        self.part_label = QLabel("PART: —")
        self.progress_label = QLabel("0 / 6 VALID VIEWS")
        self.progress_label.setStyleSheet("font-size: 24px; color: #65dfd0")
        self.angle_label = QLabel("Next position: 60°")
        self.cycle_label = QLabel("Cycle: —")
        for widget in (self.part_label, self.progress_label, self.angle_label, self.cycle_label):
            info.addWidget(widget)
        self.view_buttons = {}
        for index, angle in enumerate(ANGLES, 1):
            button = self.button(f"○ View {index} / {angle}° — WAITING", lambda checked=False, a=angle: self.select_view(a), info)
            self.view_buttons[angle] = button
        info.addStretch()
        self.production_detail = QLabel("Waiting for all six stopped positions")
        self.production_detail.setWordWrap(True)
        info.addWidget(self.production_detail)
        split.addWidget(side)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        layout.addWidget(split, 1)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(100)
        layout.addWidget(self.log)
        self.tabs.addTab(page, "PRODUCTION")

    def _engineering_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        line = QHBoxLayout()
        self.display_mode = QComboBox()
        self.display_mode.addItems(["ORIGINAL", "ROI", "HEATMAP", "OVERLAY", "BINARY MASK", "FILTERED MASK", "FINAL REGIONS"])
        self.display_mode.currentTextChanged.connect(self.render_engineering)
        self.show_yolo = QCheckBox("SHOW YOLO ROI")
        self.show_yolo.setChecked(self.settings.yolo.show_roi)
        self.show_yolo.toggled.connect(self.render_engineering)
        line.addWidget(self.display_mode)
        line.addWidget(self.show_yolo)
        self.button("LOAD SAVED VIEW", self.open_saved_view, line)
        self.button("INSPECT IMAGE", self.inspect_image, line)
        layout.addLayout(line)
        split = QSplitter()
        self.engineering_viewer = ImageViewer()
        split.addWidget(self.engineering_viewer)
        right = QWidget()
        form = QVBoxLayout(right)
        self.tuning = ToleranceControls(self.settings.tolerance, self.retune)
        form.addWidget(self.tuning)
        self.button("SAVE TOLERANCES FOR FUTURE CYCLES", self.save_tolerances, form)
        self.engineering_stats = QLabel("Select a captured view. Tuning uses its cached heatmap.")
        self.engineering_stats.setWordWrap(True)
        form.addWidget(self.engineering_stats)
        form.addStretch()
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        layout.addWidget(split)
        self.tabs.addTab(page, "ENGINEERING")

    def _settings_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        grid = QGridLayout(inner)
        self.settings_inputs = {}
        for section_index, (section, values) in enumerate(self.settings.to_dict().items()):
            group = QGroupBox(section.upper())
            form = QFormLayout(group)
            for key, value in values.items():
                if isinstance(value, bool):
                    widget = QCheckBox()
                    widget.setChecked(value)
                elif isinstance(value, (int, float)):
                    widget = spin(value, integer=isinstance(value, int), minimum=-100 if key in {"exposure", "gain", "class_id"} else 0)
                    if section == "machine" and key in {"views", "angle_step"}:
                        widget.setEnabled(False)
                elif key in {"interface", "decision_mode", "backbone"}:
                    widget = QComboBox()
                    widget.addItems({"interface": ["manual", "tcp_json"], "decision_mode": ["PIXEL", "PERCENTAGE", "BOTH", "EITHER"], "backbone": ["wide_resnet50_2", "resnet18"]}[key])
                    widget.setCurrentText(value)
                else:
                    widget = QLineEdit(json.dumps(value) if isinstance(value, (dict, tuple, list)) else value)
                self.settings_inputs[(section, key)] = widget
                if key.endswith("path") or key.endswith("dir"):
                    row = QHBoxLayout()
                    row.addWidget(widget, 1)
                    browse = QPushButton("Browse")
                    browse.clicked.connect(lambda checked=False, w=widget, directory=key.endswith("dir"): self.browse_setting(w, directory))
                    row.addWidget(browse)
                    form.addRow(key.replace("_", " "), row)
                else:
                    form.addRow(key.replace("_", " "), widget)
            grid.addWidget(group, section_index // 2, section_index % 2)
        scroll.setWidget(inner)
        layout.addWidget(scroll)
        self.settings_notice = QLabel("ROI/preprocessing changes require retraining. Tolerances affect future cycles only; active cycles keep their settings snapshot.")
        self.settings_notice.setWordWrap(True)
        layout.addWidget(self.settings_notice)
        controls = QHBoxLayout()
        self.button("VALIDATE & SAVE SETTINGS", self.save_settings, controls)
        self.button("TRAIN PATCHCORE FROM GOOD IMAGES", self.train, controls)
        self.train_angle = QComboBox()
        self.train_angle.addItems(["Shared model"] + [str(a) for a in ANGLES])
        controls.addWidget(self.train_angle)
        layout.addLayout(controls)
        self.tabs.addTab(page, "SETTINGS")

    def _calibration_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        controls = QHBoxLayout()
        self.button("ADD GOOD IMAGES", lambda: self.add_calibration("GOOD"), controls)
        self.button("ADD NG IMAGES", lambda: self.add_calibration("NG"), controls)
        self.button("LOAD LABELED MANIFEST", self.load_manifest, controls)
        self.button("EXPORT CALIBRATION CSV", self.export_calibration, controls)
        self.calibration_angle = QComboBox()
        self.calibration_angle.addItems([str(a) for a in ANGLES])
        controls.addWidget(QLabel("Angle for selected images:"))
        controls.addWidget(self.calibration_angle)
        layout.addLayout(controls)
        self.calibration_tuning = ToleranceControls(self.settings.tolerance, self.recalculate_calibration)
        layout.addWidget(self.calibration_tuning)
        self.calibration_table = QTableWidget(0, 6)
        self.calibration_table.setHorizontalHeaderLabels(["Image", "Actual", "Anomaly px", "Anomaly %", "Largest px", "Predicted"])
        self.calibration_table.cellClicked.connect(self.calibration_selected)
        layout.addWidget(self.calibration_table, 1)
        self.calibration_summary = QLabel("Load independent GOOD/NG samples; thresholds are never automatically selected.")
        self.calibration_summary.setWordWrap(True)
        layout.addWidget(self.calibration_summary)
        self.calibration_progress = QLabel("")
        layout.addWidget(self.calibration_progress)
        self.tabs.addTab(page, "TOLERANCE CALIBRATION")

    def browse_setting(self, widget, directory):
        path = QFileDialog.getExistingDirectory(self, "Select directory") if directory else QFileDialog.getOpenFileName(self, "Select model", "", "Models (*.pt *.pth);;All files (*)")[0]
        if path:
            widget.setText(path)

    def read_settings(self):
        data = self.settings.to_dict()
        for (section, key), widget in self.settings_inputs.items():
            old = data[section][key]
            if isinstance(widget, QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, QComboBox):
                value = widget.currentText()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                value = widget.value()
            else:
                value = json.loads(widget.text()) if isinstance(old, (tuple, list, dict)) else widget.text().strip()
            data[section][key] = value
        return InspectionSettings.from_dict(data)

    def save_settings(self):
        if self.session is not None and self.session.isRunning() or self.camera is not None and self.camera.isRunning() or any(t.isRunning() for t in self.tasks):
            self.error("Stop camera/inspection and wait for tasks before changing settings")
            return
        try:
            settings = self.read_settings()
            settings.save(self.settings_path)
            self.settings = settings
            self.manual_box.setVisible(settings.machine.interface == "manual")
            self.log.append("Settings saved; new cycles use the saved configuration.")
        except Exception as exc:
            self.error(str(exc))

    def save_tolerances(self):
        if self.session is not None and self.session.isRunning():
            self.error("Stop inspection before saving production tolerances")
            return
        try:
            self.settings = replace(self.settings, tolerance=self.tuning.value()).validate()
            self.settings.save(self.settings_path)
            self.log.append("Tolerances saved. Engineering previews do not change recorded verdicts.")
            for name, value in asdict(self.settings.tolerance).items():
                widget = self.settings_inputs[("tolerance", name)]
                if isinstance(widget, QComboBox):
                    widget.setCurrentText(value)
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(value))
                else:
                    widget.setValue(value)
        except Exception as exc:
            self.error(str(exc))

    def run_task(self, function, callback):
        if self.session is not None and self.session.isRunning() or any(t.isRunning() for t in self.tasks):
            self.error("Finish the active operation before starting another model/storage task")
            return
        task = TaskWorker(function)
        self.tasks.append(task)
        task.result.connect(callback)
        task.failed.connect(self.error)
        task.progress.connect(self.calibration_progress.setText)
        task.progress.connect(self.log.append)
        task.finished.connect(lambda: self.tasks.remove(task))
        task.start()

    def open_camera(self):
        if self.camera is not None and self.camera.isRunning():
            return
        self.camera = CameraWorker(self.settings)
        self.camera.preview.connect(self.preview)
        self.camera.failed.connect(self.camera_error)
        self.camera.start()

    def arm(self):
        if any(task.isRunning() for task in self.tasks) or self.session is not None and self.session.isRunning():
            return
        self.open_camera()
        self.session = SessionWorker(self.settings, self.camera)
        self.session_is_ready = False
        self.session.status.connect(self.session_status)
        self.session.view_ready.connect(self.receive_view)
        self.session.part_ready.connect(self.receive_part)
        self.session.failed.connect(self.error)
        self.session.ready.connect(self.inspection_ready)
        self.session.finished.connect(lambda: setattr(self, "session_is_ready", False))
        self.session.start()

    def inspection_ready(self):
        self.session_is_ready = True
        self.log.append("Inspection armed. Awaiting fit-check and stopped-position confirmations.")

    def machine_event(self, event):
        if self.session is None or not self.session.isRunning() or not self.session_is_ready:
            self.error("Arm inspection and wait for model readiness first")
            return
        try:
            self.session.send_event(event)
        except Exception as exc:
            self.error(str(exc))

    def begin_part(self):
        if not self.fit_check.isChecked():
            self.error("Confirm completion of the PLC 360° fitting rotation before inspection")
            return
        identifier = self.part_id.text().strip() or time.strftime("PART_%Y%m%d_%H%M%S")
        self.machine_event(MachineEvent("begin", part_id=identifier, fit_check_complete=True))
        self.fit_check.setChecked(False)

    def confirm_stop(self):
        angle = next((a for a in ANGLES if a not in self.view_results), None)
        if angle is not None:
            self.machine_event(MachineEvent("stopped", angle=angle))

    def stop(self):
        if self.session is not None:
            self.session.requestInterruption()
        if self.camera is not None:
            self.camera.requestInterruption()
        self.log.append("Stop requested; incomplete inspection will be recorded as a fault.")

    def camera_error(self, message):
        if self.session is not None:
            self.session.requestInterruption()
        self.error(message)

    def preview(self, frame):
        if self.current_view is None:
            self.production_viewer.set_frame(frame)

    def session_status(self, state, metadata):
        if metadata.get("part_id") != self.part_metadata.get("part_id") or metadata.get("start_time") != self.part_metadata.get("start_time"):
            self.view_results.clear()
            self.current_view = None
            self.cycle_started_at = time.monotonic() if metadata.get("part_id") else None
            for index, angle in enumerate(ANGLES, 1):
                self.view_buttons[angle].setText(f"○ View {index} / {angle}° — WAITING")
        self.part_metadata = metadata
        count = metadata.get("view_count", 0)
        self.part_label.setText("PART: " + metadata.get("part_id", "—"))
        self.progress_label.setText(f"{count} / 6 VALID VIEWS")
        self.angle_label.setText(f"Next position: {(count + 1) * 60}°" if count < 6 else "All six positions inspected")
        self.result_badge.setText(state)
        self.result_badge.setStyleSheet("font-size: 24px; font-weight: bold; padding: 20px; background: " + ("#66272b" if "FAIL" in state or "FAULT" in state else "#175245" if state == "PASS" else "#233344"))
        self.log.append(state)

    def receive_view(self, view):
        self.view_results[view.angle] = view
        self.current_view = view
        self.view_buttons[view.angle].setText(f"● View {view.view_number} / {view.angle}° — {view.verdict}")
        self.production_viewer.set_frame(marked_image(view.roi_image, view.decision))
        self.production_detail.setText(f"View {view.view_number}: {view.verdict}\nQuality {view.quality.score:.1f}%\nAnomaly {view.decision.filtered_anomaly_pixels:,} px ({view.decision.anomaly_percentage:.4f}%)\nLargest region {view.decision.largest_component_pixels:,} px")
        self.render_engineering()

    def receive_part(self, metadata):
        self.part_metadata = metadata
        self.result_badge.setText("FINAL " + metadata["final_verdict"])
        self.progress_label.setText("6 / 6 VALID VIEWS")
        failed = ", ".join(f"View {ANGLES.index(a)+1} / {a}°" for a in metadata["failed_views"])
        self.production_detail.setText("Rejected: " + failed if failed else "All six independent views passed")

    def select_view(self, angle):
        if angle in self.view_results:
            self.current_view = self.view_results[angle]
            self.production_viewer.set_frame(marked_image(self.current_view.roi_image, self.current_view.decision))
            self.render_engineering()

    def retune(self, *_):
        if not hasattr(self, "tuning"):
            return
        self.render_engineering()

    def render_engineering(self, *_):
        if self.current_view is None:
            return
        try:
            tuned = self.current_view.retune(self.tuning.value())
        except ValueError as exc:
            self.engineering_stats.setText(str(exc))
            return
        d = tuned.decision
        mode = self.display_mode.currentText()
        if mode == "ORIGINAL":
            image = tuned.original_image.copy()
            if self.show_yolo.isChecked():
                x, y, w, h = tuned.roi_bounds
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 220, 255), 2)
        elif mode == "ROI":
            image = tuned.roi_image
        elif mode in {"HEATMAP", "OVERLAY"}:
            image = heatmap_image(tuned.heatmap, self.settings.patchcore.heatmap_display_max)
            if mode == "OVERLAY":
                image = cv2.addWeighted(tuned.roi_image, .6, image, .4, 0)
        elif mode in {"BINARY MASK", "FILTERED MASK"}:
            image = (d.candidate_mask if mode == "BINARY MASK" else d.filtered_mask).astype(np.uint8) * 255
        else:
            image = marked_image(tuned.roi_image, d)
        self.engineering_viewer.set_frame(image)
        self.engineering_stats.setText(f"ENGINEERING PREVIEW: {tuned.verdict} • Recorded: {self.current_view.verdict}\n"
            f"View {tuned.view_number} / {tuned.angle}°\nQuality: {tuned.quality.score:.1f}% • Blur: {tuned.quality.blur_score:.1f}\n"
            f"YOLO: {tuned.yolo_confidence:.3f} • PatchCore score: {tuned.patchcore_score:.5f}\n"
            f"Heatmap threshold: {tuned.tolerance.heatmap_threshold:.5f}\nFixed scale: 0 to {self.settings.patchcore.heatmap_display_max:.5f}\n"
            f"Raw thresholded pixels: {d.raw_anomaly_pixels:,}\nAfter border exclusion: {d.valid_candidate_pixels:,}\n"
            f"Filtered anomaly: {d.filtered_anomaly_pixels:,} px\nValid ROI: {d.valid_roi_pixels:,} px\n"
            f"Anomaly area: {d.anomaly_percentage:.5f}%\nLargest region: {d.largest_component_pixels:,} px\n"
            f"Regions: {len(d.regions)} • Inference: {tuned.inference_ms:.1f} ms\nTuning uses the cached map and does not change stored production results.")

    def open_saved_view(self):
        directory = QFileDialog.getExistingDirectory(self, "Select saved view_60/120/... directory", self.settings.storage.results_dir)
        if directory:
            self.run_task(lambda progress: load_saved_view(directory), self.show_offline)

    def show_offline(self, view):
        self.current_view = view
        self.render_engineering()
        self.tabs.setCurrentIndex(1)

    def inspect_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select full-camera image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        angle = int(self.calibration_angle.currentText())
        def inspect(progress):
            pipeline = FixedViewPipeline(self.settings)
            view = pipeline.inspect_frames([cv2.imread(path)], angle)
            from .fixed_views import PartInspectionResult
            part = PartInspectionResult("OFFLINE_SINGLE_VIEW", view.timestamp, time.monotonic(), {angle: view})
            storage = ResultStorage(replace(self.settings, storage=replace(self.settings.storage, save_pass_images=True)))
            storage.begin(part)
            storage.save_view(view)
            storage.update_part(part)
            return view
        self.run_task(inspect, self.show_offline)

    def train(self):
        try:
            settings = self.read_settings()
            angle = int(self.train_angle.currentText()) if self.train_angle.currentText() != "Shared model" else None
            self.run_task(lambda progress: SpatialPatchCore(settings).train_fixed(angle, lambda update: progress(update.format())),
                          lambda path: self.log.append("PatchCore trained: " + str(path)))
        except Exception as exc:
            self.error(str(exc))

    def add_calibration(self, label):
        paths, _ = QFileDialog.getOpenFileNames(self, f"Select independent {label} full-camera images", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if paths:
            self.process_calibration([{ "image": p, "actual_class": label, "angle": int(self.calibration_angle.currentText())} for p in paths])

    def load_manifest(self):
        path, _ = QFileDialog.getOpenFileName(self, "Calibration JSON manifest", "", "JSON (*.json)")
        if not path:
            return
        try:
            entries = json.loads(Path(path).read_text(encoding="utf-8"))
            for entry in entries:
                key = "saved_view" if "saved_view" in entry else "image"
                value = Path(entry[key])
                if not value.is_absolute():
                    entry[key] = str(Path(path).parent / value)
            self.process_calibration(entries)
        except Exception as exc:
            self.error(str(exc))

    def process_calibration(self, entries):
        def process(progress):
            pipeline = FixedViewPipeline(self.settings)
            samples, errors = [], []
            for index, entry in enumerate(entries, 1):
                label = entry["actual_class"]
                if label not in {"GOOD", "NG"}:
                    raise ValueError("Manifest classes must be GOOD or NG")
                name = entry.get("image", entry.get("saved_view", ""))
                try:
                    view = load_saved_view(entry["saved_view"]) if "saved_view" in entry else pipeline.inspect_frames([cv2.imread(entry["image"])], int(entry.get("angle", 60)))
                    samples.append(CalibrationSample(name, label, view))
                except Exception as exc:
                    errors.append(f"{name}: INVALID / UNRUN — {exc}")
                progress(f"Calibration {index}/{len(entries)}")
            return samples, errors
        self.run_task(process, self.calibration_loaded)

    def calibration_loaded(self, result):
        samples, errors = result
        self.samples.extend(samples)
        self.calibration_errors.extend(errors)
        self.recalculate_calibration()

    def recalculate_calibration(self, *_):
        if not hasattr(self, "calibration_table"):
            return
        self.calibration_revision += 1
        self.calibration_debounce.start(40)

    def _calculate_calibration(self):
        if self.calibration_worker is not None and self.calibration_worker.isRunning():
            self.calibration_debounce.start(40)
            return
        try:
            tolerance = self.calibration_tuning.value()
            samples = list(self.samples)
            revision = self.calibration_revision
            task = TaskWorker(lambda progress: evaluate_samples(samples, tolerance))
            self.calibration_worker = task
            task.result.connect(lambda result: self._display_calibration(result, revision))
            task.failed.connect(self.error)
            task.start()
        except Exception as exc:
            self.calibration_summary.setText(str(exc))

    def _display_calibration(self, result, revision):
        if revision != self.calibration_revision:
            return
        try:
            rows, counts = result
            self.calibration_rows = rows
            self.calibration_table.setRowCount(len(rows))
            for i, row in enumerate(rows):
                values = [row["image"], row["actual_class"], row["filtered_anomaly_pixels"],
                          f"{row['anomaly_percentage']:.5f}", row["largest_component_pixels"], row["verdict"]]
                for j, value in enumerate(values):
                    self.calibration_table.setItem(i, j, QTableWidgetItem(str(value)))
            self.calibration_table.resizeColumnsToContents()
            fp, fn = counts["false_positive_rate"], counts["false_negative_rate"]
            rates = f"FPR: {fp*100:.2f}%" if fp is not None else "FPR: N/A (no GOOD samples)"
            rates += f" • FNR: {fn*100:.2f}%" if fn is not None else " • FNR: N/A (no NG samples)"
            self.calibration_summary.setText(f"GOOD passed {counts['GOOD_correctly_passed']} / falsely failed {counts['GOOD_incorrectly_failed']} • "
               f"NG failed {counts['NG_correctly_failed']} / falsely passed {counts['NG_incorrectly_passed']}\n{rates}\n"
               f"INVALID / UNRUN samples: {len(self.calibration_errors)}\n" + "\n".join(self.calibration_errors[-5:]))
        except Exception as exc:
            self.calibration_summary.setText(str(exc))

    def calibration_selected(self, row, column):
        if row < len(self.samples):
            self.current_view = self.samples[row].view
            for key, value in asdict(self.calibration_tuning.value()).items():
                widget = self.tuning.inputs[key]
                widget.setCurrentText(str(value)) if isinstance(widget, QComboBox) else widget.setValue(value)
            self.render_engineering()
            self.tabs.setCurrentIndex(1)

    def export_calibration(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export calibration", "calibration.csv", "CSV (*.csv)")
        if path:
            rows = list(self.calibration_rows)
            tolerance = asdict(self.calibration_tuning.value())
            errors = list(self.calibration_errors)
            def save(progress):
                fields = ["image", "actual_class", "filtered_anomaly_pixels", "valid_roi_pixels", "anomaly_percentage", "largest_component_pixels", "verdict"]
                with Path(path).open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fields, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(rows)
                Path(path).with_suffix(".settings.json").write_text(json.dumps({"tolerance": tolerance, "invalid_or_unrun": errors}, indent=2), encoding="utf-8")
                return path
            self.run_task(save, lambda p: self.log.append("Calibration exported: " + str(p)))

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export part history", "inspection_history.csv", "CSV (*.csv)")
        if path:
            self.run_task(lambda progress: export_history(self.settings.storage.results_dir, path), lambda count: self.log.append(f"Exported {count} history rows"))

    def _cycle_time(self):
        duration = self.part_metadata.get("cycle_time_s")
        if duration is not None:
            self.cycle_label.setText(f"Cycle: {duration:.2f} s")
        elif self.cycle_started_at is not None:
            self.cycle_label.setText(f"Cycle: {time.monotonic() - self.cycle_started_at:.2f} s")

    def error(self, message):
        self.log.append("ERROR: " + message)
        self.statusBar().showMessage(message)
        if self.session is not None and self.session.isRunning():
            self.result_badge.setText("SYSTEM FAULT")

    def closeEvent(self, event):
        workers = [self.session, self.camera, self.calibration_worker, *self.tasks]
        if any(worker is not None and worker.isRunning() for worker in workers):
            self.stop()
            self.statusBar().showMessage("Stopping workers; close again after they finish")
            event.ignore()
            return
        super().closeEvent(event)


def main() -> None:
    from .app import LoginDialog
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    login = LoginDialog(AuthStore())
    if login.exec() != QDialog.DialogCode.Accepted or login.user is None:
        return
    settings_path = "config/fixed_inspection.json"
    if "--settings" in sys.argv:
        settings_path = sys.argv[sys.argv.index("--settings") + 1]
    try:
        window = FixedInspectionWindow(login.user, settings_path)
    except Exception as exc:
        QMessageBox.critical(None, "Configuration error", str(exc))
        return
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
