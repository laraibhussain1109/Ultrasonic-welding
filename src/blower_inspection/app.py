"""PyQt6 industrial UI for blower fan inspection."""

from __future__ import annotations

import sys
import time
import json
import os
from dataclasses import replace
from pathlib import Path

import cv2
from PyQt6.QtCore import QThread, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QDoubleSpinBox,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from .auth import AuthStore, User
from .camera import (
    DEFAULT_COMPONENT_ROI_RATIOS,
    USBCamera,
    component_roi_bounds,
    crop_bounds,
    crop_component_roi,
    is_part_present,
    roi_ratios_from_bounds,
    save_capture,
)
from .config import ModelRegistry, PartModelConfig, ensure_model_folders
from .daily_stats import DailyStatistics, operating_day
from .fail_output import ESP32FailOutputBridge
from .frame_selection import RotationPhaseGate, SharpFrameSampler
from .frame_quality import FrameQualityAnalyzer
from .stationary_capture import INSPECTION_ANGLES, StationaryViewCapture
from .stationary_workers import StationaryCameraWorker, YoloPresenceWorker
from .capture_performance import EvidenceQueue, StageMeter, preview_image
from .inspector_factory import inspector_for_model
from .trainer import InspectionResult
from .tao_training import run_visual_changenet_task
from .yolo_tracking import SUPPORTED_COMPLETION_MODES, RotatingPartInspector, TrackedPart, YoloByteTrackDetector


QSS = """
* { font-family: 'Segoe UI', 'Rajdhani', Arial; color: #8fb7df; }
QWidget { background: #050a12; }
QFrame#topBar, QFrame#leftPanel, QFrame#rightPanel, QFrame#bottomPanel { background: #081323; border: 1px solid #0b314a; }
QLabel#brand { color: #00d9ff; font-size: 28px; font-weight: 800; letter-spacing: 5px; }
QLabel#subBrand, QLabel#sectionTitle { color: #38577e; font-size: 12px; font-weight: 700; letter-spacing: 7px; }
QLabel#metricValue { color: #00ffe1; font-size: 34px; font-weight: 900; }
QLabel#statusStandby { background: #1c1b2b; border: 1px solid #173a59; border-radius: 4px; color: #b7c5e8; font-size: 30px; font-weight: 900; letter-spacing: 12px; padding: 15px; }
QLabel#statusPass { background: #062319; border: 1px solid #00ff85; border-radius: 4px; color: #00ff85; font-size: 30px; font-weight: 900; letter-spacing: 12px; padding: 15px; }
QLabel#statusFail { background: #25080d; border: 1px solid #ff3030; border-radius: 4px; color: #ff4040; font-size: 30px; font-weight: 900; letter-spacing: 12px; padding: 15px; }
QLabel#statusNoPart { background: #211b08; border: 1px solid #f5c542; border-radius: 4px; color: #f5c542; font-size: 30px; font-weight: 900; letter-spacing: 8px; padding: 15px; }
QPushButton { background: #08172a; border: 1px solid #0b314a; border-radius: 5px; padding: 14px; color: #8fb7df; font-size: 14px; font-weight: 800; letter-spacing: 2px; }
QPushButton:hover { border-color: #00c8ff; color: #00d9ff; background: #092038; }
QPushButton#primary { border: 2px solid #00c8ff; color: #00d9ff; background: #08253a; }
QPushButton#danger { border: 1px solid #bf1020; color: #ff4040; }
QPushButton#train { border: 1px solid #f5aa00; color: #ffc14d; }
QComboBox, QLineEdit { background: #08172a; border: 1px solid #0b314a; border-radius: 5px; padding: 11px; color: #a7c7ef; }
QListWidget { background: #060e1b; border: 1px solid #0b314a; color: #8fb7df; }
QSlider::groove:horizontal { height: 7px; background: #10283d; border-radius: 3px; }
QSlider::handle:horizontal { background: #ffc400; border: 1px solid #ffb000; width: 12px; margin: -5px 0; border-radius: 6px; }
"""


class LoginDialog(QDialog):
    def __init__(self, auth: AuthStore) -> None:
        super().__init__()
        self.auth = auth
        self.user: User | None = None
        self.setWindowTitle("NeuroIris Login")
        self.setMinimumWidth(440)
        layout = QVBoxLayout(self)
        title = QLabel("NeuroIris")
        title.setObjectName("brand")
        subtitle = QLabel("STAVATECH TECHNOLOGIES PVT. LTD.")
        subtitle.setObjectName("subBrand")
        form = QFormLayout()
        self.username = QLineEdit("admin")
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        self.password.setText("admin123")
        form.addRow("USERNAME", self.username)
        form.addRow("PASSWORD", self.password)
        login = QPushButton("LOGIN")
        login.setObjectName("primary")
        login.clicked.connect(self._login)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(20)
        layout.addLayout(form)
        layout.addWidget(login)

    def _login(self) -> None:
        user = self.auth.authenticate(self.username.text(), self.password.text())
        if user is None:
            QMessageBox.critical(self, "Login failed", "Invalid username or password")
            return
        self.user = user
        self.accept()


class TrainWorker(QThread):
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, inspector, model: PartModelConfig) -> None:
        super().__init__()
        self.inspector = inspector
        self.model = model

    def run(self) -> None:
        try:
            output = self.inspector.train(
                self.model, progress_callback=lambda update: self.progress.emit(update.format())
            )
            tao_production = (
                self.model.production_algorithm != "patchcore_geometry"
                and self.model.algorithm == "nvidia_tao"
            )
            action = "CALIBRATED" if tao_production else "TRAINED"
            self.finished_ok.emit(f"{action} {self.model.id}: {output}")
        except Exception as exc:
            self.failed.emit(f"TRAINING FAILED {self.model.id}: {exc}")


class InspectionWorker(QThread):
    finished_result = pyqtSignal(int, object, float)
    failed = pyqtSignal(str)

    def __init__(self, inspector, model: PartModelConfig, track_id: int, frame, view_angle: int | None = None) -> None:
        super().__init__()
        self.inspector = inspector
        self.model = model
        self.track_id = track_id
        self.frame = frame.copy()
        self.view_angle = view_angle

    def run(self) -> None:
        start = time.perf_counter()
        try:
            result = self.inspector.inspect(self.model, self.frame, save_outputs=False, crop_to_component=False)
            result = replace(result, view_angle=self.view_angle)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished_result.emit(self.track_id, result, (time.perf_counter() - start) * 1000.0)


class TaoExportWorker(QThread):
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, model: PartModelConfig) -> None:
        super().__init__()
        self.model = model

    def run(self) -> None:
        spec = Path(f"specs/visual_changenet/{self.model.id.lower()}_segmentation.yaml")
        results = Path(f"data/results/{self.model.id}/tao")
        output = self.model.tao_model_file or self.model.model_file
        try:
            run_visual_changenet_task(
                "export", spec, results_dir=results, export_file=output,
                progress_callback=lambda update: self.progress.emit(update.format()),
            )
            self.finished_ok.emit(f"EXPORTED {self.model.id}: {output}")
        except Exception as exc:
            self.failed.emit(f"EXPORT FAILED {self.model.id}: {exc}")


class InspectionWindow(QWidget):
    def __init__(self, user: User) -> None:
        super().__init__()
        self.user = user
        self.registry = ModelRegistry()
        ensure_model_folders(self.registry)
        self.inspector = inspector_for_model(self.registry.active())
        self.fail_output = ESP32FailOutputBridge()
        active_model = self.registry.active()
        self.camera = USBCamera(width=active_model.camera_width, height=active_model.camera_height, fps=active_model.camera_fps)
        self.frame = None
        self.inspection_running = False
        self.inspection_generation = 0
        self.camera_worker: StationaryCameraWorker | None = None
        self.presence_worker: YoloPresenceWorker | None = None
        self.camera_sequence = 0
        self.presence_poll_at = 0.0
        self.locked_absent_since: float | None = None
        self.locked_absence_checks = 0
        self.inference_worker: InspectionWorker | None = None
        self.last_inference_at = 0.0
        self.inference_interval_s = 0.05
        self.latest_annotated_frame = None
        self.current_display_frame = None
        self.live_roi_bounds = None
        self.locked_roi_confidence = 0.0
        self.locked_track_id = 1
        self.locked_part_present = False
        self.locked_missing_frames = 0
        self.locked_presence_poll = 0
        self.locked_last_yolo_present = False
        self.raw_frame = None
        self.fps_frame_count = 0
        self.fps_started_at = time.perf_counter()
        self.tolerance_percent = 5.0
        self.frame_sampler: SharpFrameSampler | None = None
        self.rotation_phase_gate: RotationPhaseGate | None = None
        self.pending_sharp_frames = {}
        self.stationary_captures: dict[int, StationaryViewCapture] = {}
        self.stationary_queue = EvidenceQueue(max_items=12)
        self.preview_meter = StageMeter()
        self.inference_latency_ms = 0.0
        self._camera_starting = False
        self.active_fail_asserted = False
        self.daily_statistics = DailyStatistics(memory_only=active_model.runtime_storage_mode == "memory")
        self.stats = self.daily_statistics.counts()
        self.part_detector: YoloByteTrackDetector | None = None
        self.rotating_parts: RotatingPartInspector | None = None
        self.started_at = time.time()
        self.train_worker: TrainWorker | None = None
        self.export_worker: TaoExportWorker | None = None
        self.setWindowTitle(f"NeuroIris Blower Fan Inspection - {user.username} ({user.role})")
        self.resize(1884, 940)
        self._build_ui()
        self.clock = QTimer(self)
        self.clock.timeout.connect(self._tick)
        self.clock.start(500)
        self.live_timer = QTimer(self)
        self.live_timer.timeout.connect(self._process_live_frame)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._top_bar())
        main = QHBoxLayout()
        main.setSpacing(0)
        main.addWidget(self._left_panel(), 0)
        main.addWidget(self._center_panel(), 1)
        main.addWidget(self._right_panel(), 0)
        root.addLayout(main, 1)

    def _top_bar(self) -> QFrame:
        bar = QFrame(objectName="topBar")
        layout = QHBoxLayout(bar)
        brand_box = QVBoxLayout()
        brand = QLabel("◉  NeuroIris")
        brand.setObjectName("brand")
        sub = QLabel("STAVATECH TECHNOLOGIES PVT. LTD.")
        sub.setObjectName("subBrand")
        brand_box.addWidget(brand)
        brand_box.addWidget(sub)
        layout.addLayout(brand_box)
        layout.addStretch(1)
        self.online_label = QLabel("● OFFLINE")
        self.speed_top = QLabel("SURFACE SPEED:  1.0 m/s")
        self.tolerance_top = QLabel("TOLERANCE:  5%")
        self.fps_top = QLabel("FPS:  -")
        self.latency_top = QLabel("LATENCY:  - ms")
        self.time_label = QLabel("--:--:--")
        for widget in [self.online_label, self.speed_top, self.tolerance_top, self.fps_top, self.latency_top, self.time_label]:
            widget.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
            layout.addWidget(widget)
            layout.addSpacing(25)
        return bar

    def _left_panel(self) -> QFrame:
        panel = QFrame(objectName="leftPanel")
        panel.setFixedWidth(385)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(23, 20, 23, 20)
        layout.addWidget(self._section("SYSTEM CONTROL"))
        self.model_combo = QComboBox()
        self.model_by_label = {}
        self._refresh_models()
        self.model_combo.currentTextChanged.connect(lambda _text: self._apply_selected_camera_settings())
        layout.addWidget(self.model_combo)
        start = QPushButton("▶   START INSPECTION")
        start.setObjectName("primary")
        start.clicked.connect(self.start_inspection)
        layout.addWidget(start)
        calibrate = QPushButton("⊕   CALIBRATE")
        calibrate.clicked.connect(self.load_image)
        layout.addWidget(calibrate)
        roi_button = QPushButton("▣   SET PART ROI")
        roi_button.clicked.connect(self.set_part_roi)
        layout.addWidget(roi_button)
        detector_button = QPushButton("⌕   YOLO PART MODEL")
        detector_button.clicked.connect(self.set_yolo_model)
        layout.addWidget(detector_button)
        camera_button = QPushButton("⚙   CAMERA FPS / RESOLUTION")
        camera_button.clicked.connect(self.set_camera_settings)
        layout.addWidget(camera_button)
        stop = QPushButton("▪   STOP")
        stop.setObjectName("danger")
        stop.clicked.connect(self.stop_camera)
        layout.addWidget(stop)
        reset = QPushButton("↻   RESET STATS")
        reset.clicked.connect(self.reset_stats)
        layout.addWidget(reset)
        layout.addSpacing(26)
        layout.addWidget(self._section("TOLERANCE SETTING"))
        layout.addWidget(QLabel("DEFECT SIZE THRESHOLD — LOWER = STRICTER"))
        tolerance_grid = QGridLayout()
        for index, value in enumerate([1, 3, 5, 8]):
            button = QPushButton(f"{value}%")
            if value == 5:
                button.setStyleSheet("border-color:#d29b00;color:#f5c542;")
            button.clicked.connect(lambda _checked=False, v=value: self.set_tolerance(v))
            tolerance_grid.addWidget(button, 0, index)
        manual = QPushButton("MANUAL")
        manual.clicked.connect(self.show_manual_tolerance)
        tolerance_grid.addWidget(manual, 1, 0, 1, 2)
        self.manual_tolerance = QDoubleSpinBox()
        self.manual_tolerance.setRange(0.00, 50.00)
        self.manual_tolerance.setDecimals(2)
        self.manual_tolerance.setSingleStep(0.25)
        self.manual_tolerance.setSuffix(" %")
        self.manual_tolerance.setValue(5.0)
        self.manual_tolerance.setVisible(False)
        self.manual_tolerance.valueChanged.connect(self.set_tolerance)
        tolerance_grid.addWidget(self.manual_tolerance, 1, 2, 1, 2)
        layout.addLayout(tolerance_grid)
        layout.addSpacing(25)
        layout.addWidget(self._section("SURFACE SPEED"))
        self.speed_value = QLabel("1.0")
        self.speed_value.setObjectName("metricValue")
        layout.addWidget(self.speed_value)
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 30)
        self.speed_slider.setValue(10)
        self.speed_slider.valueChanged.connect(lambda value: self.speed_value.setText(f"{value / 10:.1f}"))
        layout.addWidget(self.speed_slider)
        layout.addWidget(QLabel("FIXED LINE RATE — OPTIMISED @ 30 FPS"))
        layout.addStretch(1)
        if self.user.is_admin:
            export = QPushButton("⬡   EXPORT TAO MODEL (ENGINEERING)")
            export.setObjectName("train")
            export.clicked.connect(self.export_selected)
            layout.addWidget(export)
            train = QPushButton("◆   TRAIN PATCHCORE MODEL")
            train.setObjectName("train")
            train.clicked.connect(self.train_selected)
            layout.addWidget(train)
        return panel

    def _center_panel(self) -> QFrame:
        center = QFrame()
        layout = QVBoxLayout(center)
        layout.setContentsMargins(0, 0, 0, 0)
        self.viewer = QLabel("NO CAMERA FRAME")
        self.viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viewer.setStyleSheet("background:#000000; border:2px solid #00bdea; color:#1c4c65; font-size:24px; letter-spacing:6px;")
        self.viewer.setMinimumSize(640, 360)
        self.viewer.setScaledContents(False)
        self.viewer.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        layout.addWidget(self.viewer, 1)
        bottom = QFrame(objectName="bottomPanel")
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.addWidget(QLabel("ANOMALY SCORE"))
        self.score_slider = QSlider(Qt.Orientation.Horizontal)
        self.score_slider.setEnabled(False)
        self.score_slider.setRange(0, 1000)
        bottom_layout.addWidget(self.score_slider, 1)
        self.score_label = QLabel("0.000")
        self.score_label.setObjectName("metricValue")
        bottom_layout.addWidget(self.score_label)
        layout.addWidget(bottom)
        return center

    def _right_panel(self) -> QFrame:
        panel = QFrame(objectName="rightPanel")
        panel.setFixedWidth(410)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(23, 13, 18, 18)
        self.status_badge = QLabel("STANDBY")
        self.status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_badge.setObjectName("statusStandby")
        layout.addWidget(self.status_badge)
        layout.addSpacing(20)
        layout.addWidget(self._section("SESSION STATISTICS"))
        stats_grid = QGridLayout()
        self.inspected_value = self._metric_card("INSPECTED", "0")
        self.pass_rate_value = self._metric_card("PASS RATE", "-")
        self.passed_value = self._metric_card("PASSED", "0")
        self.failed_value = self._metric_card("FAILED", "0")
        stats_grid.addWidget(self.inspected_value, 0, 0)
        stats_grid.addWidget(self.pass_rate_value, 0, 1)
        stats_grid.addWidget(self.passed_value, 1, 0)
        stats_grid.addWidget(self.failed_value, 1, 1)
        layout.addLayout(stats_grid)
        layout.addSpacing(25)
        layout.addWidget(self._section("LAST RESULT"))
        self.last_result = QLabel("FRAME:  -\nSCORE:  -\nCOVERAGE:  -\nLATENCY:  -")
        self.last_result.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        layout.addWidget(self.last_result)
        layout.addSpacing(25)
        layout.addWidget(self._section("INSPECTION LOG"))
        self.log = QListWidget()
        layout.addWidget(self.log, 1)
        return panel

    def _section(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionTitle")
        return label

    def _metric_card(self, title: str, value: str) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("QFrame { background:#081323; border:1px solid #0b314a; border-radius:5px; padding:12px; }")
        layout = QVBoxLayout(frame)
        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        value_label = QLabel(value)
        value_label.setObjectName("metricValue")
        frame.value_label = value_label
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        return frame

    def selected_model(self) -> PartModelConfig:
        return self.model_by_label[self.model_combo.currentText()]

    def inspection_model(self) -> PartModelConfig:
        model = self.selected_model()
        # Component-size filtering is separate from the operator's percentage.
        # The percentage measures anomalous pixels / inspectable heatmap pixels.
        return replace(
            model,
            heatmap_tolerance_percent=self.tolerance_percent,
            max_bad_sector_ratio=max(0.0001, self.tolerance_percent / 100.0),
        )

    def _refresh_models(self, selected_id: str | None = None) -> None:
        current_id = selected_id
        if current_id is None and getattr(self, "model_combo", None) is not None and self.model_combo.currentText():
            current_id = self.selected_model().id
        self.model_by_label = {f"{m.id}  |  {m.name}": m for m in self.registry.all()}
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(self.model_by_label.keys())
        if current_id is not None:
            for index, model in enumerate(self.model_by_label.values()):
                if model.id == current_id:
                    self.model_combo.setCurrentIndex(index)
                    break
        self.model_combo.blockSignals(False)
        self._apply_selected_camera_settings()

    def _apply_selected_camera_settings(self) -> None:
        if not getattr(self, "model_by_label", None) or self.inspection_running:
            return
        model = self.selected_model()
        self.inspector = inspector_for_model(model)
        self.camera = USBCamera(width=model.camera_width, height=model.camera_height, fps=model.camera_fps)

    def load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load inspection image", str(Path.cwd()), "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            QMessageBox.critical(self, "Image error", f"Unable to read {path}")
            return
        frame = crop_component_roi(frame, roi_ratios=self.selected_model().roi_ratios)
        self.frame = frame
        self.show_frame(frame)
        self.log.addItem(f"LOADED ROI {Path(path).name}")

    def set_part_roi(self) -> None:
        model = self.selected_model()
        ratios = model.roi_ratios or DEFAULT_COMPONENT_ROI_RATIOS
        if model.roi_ratios is None and self.raw_frame is not None:
            ratios = roi_ratios_from_bounds(self.raw_frame, component_roi_bounds(self.raw_frame))

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Set ROI for {model.id}")
        layout = QFormLayout(dialog)
        fields: list[QDoubleSpinBox] = []
        for label, value in zip(("X %", "Y %", "WIDTH %", "HEIGHT %"), ratios):
            field = QDoubleSpinBox()
            field.setRange(0.0, 100.0)
            field.setDecimals(2)
            field.setSingleStep(0.5)
            field.setValue(value * 100.0)
            layout.addRow(label, field)
            fields.append(field)
        save = QPushButton("SAVE ROI AS MODEL DEFAULT")
        save.clicked.connect(dialog.accept)
        layout.addRow(save)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        x, y, w, h = (field.value() / 100.0 for field in fields)
        w = max(0.01, min(w, 1.0 - x))
        h = max(0.01, min(h, 1.0 - y))
        updated = self.registry.update_model_settings(model.id, roi_ratios=(x, y, w, h))
        self._refresh_models(updated.id)
        self.live_roi_bounds = None
        self.log.addItem(f"SAVED ROI DEFAULT {updated.id}: x={x:.3f} y={y:.3f} w={w:.3f} h={h:.3f}")

    def set_camera_settings(self) -> None:
        model = self.selected_model()
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Camera settings for {model.id}")
        layout = QFormLayout(dialog)
        width = QSpinBox()
        width.setRange(320, 8192)
        width.setSingleStep(160)
        width.setValue(model.camera_width)
        height = QSpinBox()
        height.setRange(240, 8192)
        height.setSingleStep(90)
        height.setValue(model.camera_height)
        fps = QSpinBox()
        fps.setRange(1, 240)
        fps.setValue(model.camera_fps)
        layout.addRow("WIDTH", width)
        layout.addRow("HEIGHT", height)
        layout.addRow("FPS", fps)
        save = QPushButton("SAVE CAMERA DEFAULT")
        save.clicked.connect(dialog.accept)
        layout.addRow(save)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        was_running = self.inspection_running
        if was_running:
            self.stop_camera()
        updated = self.registry.update_model_settings(
            model.id,
            camera_width=width.value(),
            camera_height=height.value(),
            camera_fps=fps.value(),
        )
        self._refresh_models(updated.id)
        self.log.addItem(f"SAVED CAMERA DEFAULT {updated.id}: {updated.camera_width}x{updated.camera_height}@{updated.camera_fps}")

    def set_yolo_model(self) -> None:
        model = self.selected_model()
        path, _ = QFileDialog.getOpenFileName(self, "Select YOLO best.pt", str(model.yolo_model_path or Path.cwd()), "PyTorch model (*.pt)")
        if not path:
            return
        updated = self.registry.update_model_settings(model.id, yolo_model_path=path)
        self._refresh_models(updated.id)
        self.log.addItem(f"SAVED YOLO PART DETECTOR {updated.id}: {path}")

    def start_inspection(self) -> None:
        if self._camera_starting:
            return
        if self.inspection_running:
            return
        if self.camera_worker is not None and self.camera_worker.isRunning():
            self.log.addItem("WAITING FOR PREVIOUS CAMERA READ TO STOP")
            return
        model = self.selected_model()
        if model.yolo_model_path is None:
            QMessageBox.critical(self, "YOLO model required", "Select your trained YOLO best.pt with YOLO PART MODEL before starting live inspection.")
            return
        if model.inspection_completion_mode not in SUPPORTED_COMPLETION_MODES:
            module = sys.modules.get(RotatingPartInspector.__module__)
            loaded_from = Path(getattr(module, "__file__", "unknown")).resolve()
            QMessageBox.critical(
                self,
                "Inspection runtime is out of date",
                f"The configured completion mode {model.inspection_completion_mode!r} is not supported by "
                f"the loaded runtime:\n{loaded_from}\n\nClose every running NeuroIris/Python process, "
                "activate the intended virtual environment, then reinstall this checkout with:\n"
                'python -m pip install -e ".[industrial,dev]"\n\n'
                "Verify it with: blower-inspection doctor",
            )
            return
        # Applying camera settings also rebuilds the model-specific inspector.
        # Do this before readiness validation so the validated TAO session is
        # the same instance used for logging and inference.
        try:
            self._apply_selected_camera_settings()
        except Exception as exc:
            QMessageBox.critical(self, "Camera settings error", str(exc))
            return
        if model.production_algorithm == "patchcore_geometry":
            try:
                self.inspector.validate_ready(model)
            except Exception as exc:
                QMessageBox.critical(self, "PatchCore model not ready", str(exc))
                return
        if model.production_algorithm != "patchcore_geometry" and model.algorithm == "nvidia_tao":
            try:
                self.inspector.validate_ready(model)
            except Exception as exc:
                QMessageBox.critical(self, "TAO model not ready", str(exc))
                return
        try:
            self.part_detector = YoloByteTrackDetector(model.yolo_model_path, model.yolo_confidence)
            self.rotating_parts = RotatingPartInspector(
                model.inspection_lost_timeout_s,
                model.counting_line_ratio,
                model.counting_direction,
                model.minimum_rotation_views,
                model.weak_candidate_required_views,
                model.inspection_completion_mode,
                model.counting_axis,
                fixed_view_angles=(INSPECTION_ANGLES if model.stationary_six_view_capture else None),
            )
            self.frame_sampler = SharpFrameSampler(
                model.capture_burst_frames, model.minimum_sharpness
            )
            self.rotation_phase_gate = RotationPhaseGate(
                model.minimum_rotation_descriptor_distance,
                maximum_history=max(24, model.minimum_rotation_views * 2),
            )
            self._camera_starting = True
            worker = StationaryCameraWorker(self.camera, model, None, self.locked_track_id)
            self.camera_worker = worker
            worker.finished.connect(lambda: self._release_camera_worker(worker))
            worker.start()
            self._fresh_camera_frame()
            if model.lock_roi_after_confirmation and not self._confirm_and_lock_roi(model):
                self._stop_camera_worker()
                self.part_detector = None
                self.rotating_parts = None
                return
        except Exception as exc:
            self._stop_camera_worker()
            QMessageBox.critical(self, "Camera error", str(exc))
            return
        finally:
            self._camera_starting = False
        self.inspection_running = True
        self.inspection_generation += 1
        self.online_label.setText("● ONLINE")
        self.status_badge.setObjectName("statusStandby")
        self.status_badge.setText("RUNNING")
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)
        self.fps_frame_count = 0
        self.fps_started_at = time.perf_counter()
        self.preview_meter = StageMeter()
        self.inference_latency_ms = 0.0
        self.last_inference_at = 0.0
        self.latest_annotated_frame = None
        self.current_display_frame = None
        if not model.lock_roi_after_confirmation:
            self.live_roi_bounds = None
        self.raw_frame = None
        self.pending_sharp_frames = {}
        self.stationary_captures.clear()
        self.stationary_queue.clear()
        self.camera_sequence = 0
        self.presence_poll_at = 0.0
        self.locked_absent_since = None
        self.locked_absence_checks = 0
        try:
            device_name = self.inspector.runtime_device_name()
            runtime_summary = getattr(self.inspector, "runtime_summary", lambda: device_name)()
        except Exception as exc:
            device_name = f"unavailable ({exc})"
            runtime_summary = f"TAO runtime diagnostics unavailable: {exc}"
        if self.fail_output.config.enabled:
            # ESP32FailOutputBridge is deliberately asynchronous and exposes
            # send_result/reset rather than the synchronous connect/set_fail
            # API of the legacy serial driver. Queue a safe PASS state without
            # blocking camera startup.
            self.fail_output.reset()
            self.active_fail_asserted = False
            esp32_status = f"ESP32 CONFIGURED {self.fail_output.config.base_url}"
        else:
            esp32_status = "ESP32 DISABLED"
        self.log.addItem(
            f"LIVE INSPECTION STARTED {self.selected_model().id} | "
            f"CAMERA {self.camera.width}x{self.camera.height} reported FPS={self.camera.format_info.get('reported_fps') or 'unknown'} | DEVICE {device_name} | {esp32_status}"
        )
        self.log.addItem(runtime_summary)
        generation = self.inspection_generation
        worker.failed.connect(lambda message: self._handle_live_error(message)
                              if generation == self.inspection_generation else None)
        if worker.error_message:
            self._handle_live_error(worker.error_message)
            return
        if model.stationary_six_view_capture and self.live_roi_bounds is not None:
            worker.set_bounds(self.live_roi_bounds)
            self.log.addItem(f"INDEPENDENT CAMERA CAPTURE | burst target={model.capture_burst_frames} "
                             f"minimum={model.stationary_min_burst_frames} window={model.stationary_burst_window_ms} ms")
        self.live_timer.start(30 if self.camera_worker is not None else 1)

    def _fresh_camera_frame(self):
        """Wait for a new reader-owned snapshot, keeping startup dialogs responsive."""
        worker = self.camera_worker
        if worker is None:
            raise RuntimeError("Camera acquisition thread is not running")
        previous = worker.snapshot()
        sequence = previous.sequence if previous else 0
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if worker.error_message:
                raise RuntimeError(worker.error_message)
            QApplication.processEvents()
            if self.camera_worker is not worker:
                raise RuntimeError("Camera startup was stopped")
            current = worker.wait_snapshot(sequence, .02)
            if current is not None and current.sequence > sequence:
                return current.frame
        raise RuntimeError("No fresh camera frame delivered within 15 seconds")

    def _release_camera_worker(self, worker) -> None:
        if self.camera_worker is worker:
            self.camera_worker = None
            worker.deleteLater()

    def _stop_camera_worker(self) -> bool:
        worker = self.camera_worker
        if worker is None:
            self.camera.close()
            return True
        worker.requestInterruption()
        if worker.wait(1000):
            self._release_camera_worker(worker)
            return True
        self.log.addItem("WAITING FOR CAMERA READ TO STOP")
        return False

    def _confirm_and_lock_roi(self, model: PartModelConfig) -> bool:
        """Use YOLO once and require operator approval of a fixed live ROI."""
        assert self.part_detector is not None
        while True:
            # Discard initial auto-exposure frames before presenting the box.
            raw_frame = None
            for _ in range(3):
                raw_frame = self._fresh_camera_frame()
            assert raw_frame is not None
            self.raw_frame = raw_frame
            try:
                if hasattr(self.inspector, "detect_inspection_roi"):
                    detected = self.inspector.detect_inspection_roi(raw_frame, model, self.part_detector)
                else:
                    detected = self.part_detector.detect_best(raw_frame)
            except ValueError as exc:
                choice = QMessageBox.warning(
                    self, "ROI not found", f"{exc}\n\nRetry the camera frame?",
                    QMessageBox.StandardButton.Retry | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Retry,
                )
                if choice == QMessageBox.StandardButton.Retry:
                    continue
                return False

            preview = raw_frame.copy()
            x, y, w, h = detected.bounds
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 255), 3)
            cv2.putText(preview, f"PROPOSED FIXED ROI {detected.confidence:.0%}",
                        (x, max(24, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
            self.show_frame(preview)
            QApplication.processEvents()
            choice = QMessageBox.question(
                self,
                "Confirm fixed inspection ROI",
                "Is the yellow box the correct complete blower ROI?\n\n"
                "YES: lock this exact ROI for the entire inspection.\n"
                "NO: capture again.\nCANCEL: do not start inspection.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes,
            )
            if choice == QMessageBox.StandardButton.No:
                continue
            if choice == QMessageBox.StandardButton.Cancel:
                return False
            self.live_roi_bounds = detected.bounds
            self.locked_roi_confidence = detected.confidence
            self.locked_track_id = 1
            self.locked_part_present = True
            self.locked_last_yolo_present = True
            self.locked_missing_frames = 0
            self.log.addItem(
                f"LOCKED ROI {model.id} | x={x} y={y} w={w} h={h} | YOLO={detected.confidence:.0%}"
            )
            return True

    def _locked_roi_tracks(self, raw_frame) -> list[TrackedPart]:
        """Return the immutable ROI, with debounced YOLO-confirmed presence."""
        if self.live_roi_bounds is None:
            return []
        model = self.selected_model()
        if self.camera_worker is not None:
            now = time.monotonic()
            if self.presence_worker is None and now - self.presence_poll_at >= .5:
                self.presence_poll_at = now
                self._start_presence_check(raw_frame, model)
            present = (self.locked_last_yolo_present or self.locked_absent_since is None
                       or self.locked_absence_checks < 2 or now - self.locked_absent_since < .5)
            if present and not self.locked_part_present:
                self.locked_track_id += 1
            self.locked_part_present = present
            self.camera_worker.set_part(self.locked_track_id, present)
            return ([TrackedPart(self.locked_track_id, self.live_roi_bounds, self.locked_roi_confidence)]
                    if present else [])
        self.locked_presence_poll += 1
        # Keep the crop immutable but periodically ask YOLO whether a blower is
        # still present. Polling once per burst avoids adding detector latency to
        # every camera frame.
        if self.locked_presence_poll == 1 or self.locked_presence_poll % model.capture_burst_frames == 0:
            try:
                assert self.part_detector is not None
                detection = self.part_detector.detect_best(raw_frame)
                self.locked_last_yolo_present = detection.confidence >= model.yolo_presence_confidence
            except ValueError:
                self.locked_last_yolo_present = False
        yolo_present = self.locked_last_yolo_present
        if yolo_present:
            if not self.locked_part_present:
                self.locked_track_id += 1
            self.locked_part_present = True
            self.locked_missing_frames = 0
        else:
            self.locked_missing_frames += 1
            # A rotating, manually focused blower may look texture-poor for a
            # few consecutive blurred frames. Require roughly half a second of
            # absence before ending the fixed-nest part session.
            missing_limit = max(model.capture_burst_frames, model.camera_fps // 2)
            if self.locked_missing_frames >= missing_limit:
                self.locked_part_present = False
                return []
        return [TrackedPart(self.locked_track_id, self.live_roi_bounds, self.locked_roi_confidence)]

    def _start_presence_check(self, raw_frame, model: PartModelConfig) -> None:
        worker = YoloPresenceWorker(self.part_detector, raw_frame, model.yolo_presence_confidence)
        self.presence_worker = worker
        generation = self.inspection_generation
        def received(present, observed_at):
            if generation != self.inspection_generation or not self.inspection_running:
                return
            self.locked_last_yolo_present = present
            if present:
                self.locked_absent_since, self.locked_absence_checks = None, 0
            else:
                if self.locked_absent_since is None:
                    self.locked_absent_since = observed_at
                self.locked_absence_checks += 1
        def finished():
            if self.presence_worker is worker:
                self.presence_worker = None
            worker.deleteLater()
        worker.result.connect(received)
        worker.failed.connect(lambda message: self._handle_live_error(message)
                              if generation == self.inspection_generation else None)
        worker.finished.connect(finished)
        worker.start()

    def _process_live_frame(self) -> None:
        if not self.inspection_running:
            return
        try:
            acquisition = self.camera_worker
            if acquisition is not None and getattr(acquisition, "error_message", None):
                raise RuntimeError(acquisition.error_message)
            snapshot = acquisition.snapshot() if acquisition is not None else None
            if acquisition is not None:
                for track_id, selected in acquisition.take_captures():
                    self.stationary_queue.append((track_id, selected))
                    self.log.addItem(f"CAPTURE {selected.angle}° | track={track_id} | burst={selected.burst_frames}")
                if snapshot is None:
                    return
                if snapshot.sequence == self.camera_sequence:
                    self._dispatch_stationary_view()
                    return
                self.camera_sequence = snapshot.sequence
                if hasattr(acquisition, "mark_previewed"):
                    acquisition.mark_previewed(snapshot.sequence)
                raw_frame = snapshot.frame
            else:
                raise RuntimeError("Camera acquisition thread is not running")
            self.raw_frame = raw_frame
            assert self.part_detector is not None and self.rotating_parts is not None
            tracks = (self._locked_roi_tracks(raw_frame) if self.live_roi_bounds is not None
                      else self.part_detector.track(raw_frame))
            removed_parts = self.rotating_parts.observe_tracks(
                tracks, raw_frame.shape[1], frame_height=raw_frame.shape[0],
                pending_track_ids=self._pending_stationary_track_ids(),
            )
            for completed_part in removed_parts:
                self._handle_completed_part(completed_part)
            if removed_parts and self.active_fail_asserted:
                # A latched reject remains asserted while YOLO sees the part;
                # confirmed departure is the reset boundary. Do not queue this
                # reset after a PASS: it would immediately follow the dedicated
                # pass-pulse request and can hide the pulse on some controllers.
                self.fail_output.reset()
                self.active_fail_asserted = False
            elif not tracks and self.active_fail_asserted and not self._pending_stationary_track_ids():
                # minimum_views may already have finalized and removed its
                # session. A qualified YOLO NO PART still releases the latch.
                self.fail_output.reset()
                self.active_fail_asserted = False
            preview = preview_image(raw_frame, self.viewer.width(), self.viewer.height())
            sx, sy = preview.shape[1] / raw_frame.shape[1], preview.shape[0] / raw_frame.shape[0]
            preview_tracks = [replace(part, bounds=tuple(round(value * (sx if i % 2 == 0 else sy))
                                                       for i, value in enumerate(part.bounds))) for part in tracks]
            frame = self._draw_tracks(
                preview,
                preview_tracks,
                self.rotating_parts.counting_line_ratio,
                self.rotating_parts.counting_direction,
                self.rotating_parts.counting_axis,
                {part.track_id: self.rotating_parts.defect_sections(part.track_id) for part in tracks},
                self.selected_model().patchcore_section_count,
                defect_angles={part.track_id: self.rotating_parts.defect_section_angles(part.track_id) for part in tracks},
            )
        except Exception as exc:
            self._handle_live_error(f"Camera frame error: {exc}")
            return
        self.frame = frame
        # Never pin the last inference overlay over the live camera stream.
        # Results are still displayed when they arrive, but the following
        # camera frame must replace them so removal/motion remains visible.
        if tracks:
            self.show_frame(frame)
        self.fps_frame_count += 1
        if self.selected_model().stationary_six_view_capture:
            try:
                if acquisition is None or self.live_roi_bounds is None:
                    self._capture_stationary_views(raw_frame, tracks)
                elif not self.active_fail_asserted and self.inference_worker is None and tracks:
                    self.status_badge.setObjectName("statusStandby")
                    self.status_badge.setText(snapshot.state)
                    self.status_badge.style().unpolish(self.status_badge)
                    self.status_badge.style().polish(self.status_badge)
                self._dispatch_stationary_view()
            except Exception as exc:
                self._handle_live_error(f"Stationary capture error: {exc}")
                return
            if not tracks:
                self._handle_no_part_frame(frame)
            return
        if not tracks:
            if self.frame_sampler is not None:
                self.frame_sampler.discard_missing(set())
            self._handle_no_part_frame(frame)
            return
        assert self.frame_sampler is not None
        active_ids = {part.track_id for part in tracks}
        self.frame_sampler.discard_missing(active_ids)
        if self.rotation_phase_gate is not None:
            self.rotation_phase_gate.discard_missing(active_ids)
        self.pending_sharp_frames = {
            track_id: selected
            for track_id, selected in self.pending_sharp_frames.items()
            if track_id in active_ids
        }
        for part in tracks:
            selected = self.frame_sampler.offer(
                part.track_id, crop_bounds(raw_frame, part.bounds)
            )
            if selected is not None:
                if self.rotation_phase_gate is None or self.rotation_phase_gate.accept(part.track_id, selected.frame):
                    self.pending_sharp_frames[part.track_id] = selected
                elif self.inference_worker is None:
                    self.status_badge.setObjectName("statusStandby")
                    self.status_badge.setText("WAITING FOR ROTATION")
                    self.status_badge.style().unpolish(self.status_badge)
                    self.status_badge.style().polish(self.status_badge)
        now = time.perf_counter()
        if (
            self.inference_worker is None
            and now - self.last_inference_at >= self.inference_interval_s
        ):
            self.last_inference_at = now
            # Give each simultaneously tracked part one exact-crop inspection
            # before spending more frames on any already observed rotation.
            assert self.rotating_parts is not None
            candidates = [part for part in tracks if part.track_id in self.pending_sharp_frames
                          and self.rotating_parts.accepts_inspection(part.track_id)]
            if not candidates:
                return
            part = max(
                candidates,
                key=lambda candidate: (
                    self.rotating_parts.needs_completion_inspection(candidate.track_id),
                    self.rotating_parts.needs_initial_inspection(candidate.track_id),
                    candidate.confidence,
                ),
            )
            self.inference_worker = InspectionWorker(
                self.inspector,
                self.inspection_model(),
                part.track_id,
                self.pending_sharp_frames.pop(part.track_id).frame,
            )
            self._start_inference_worker(self.inference_worker)

    def _pending_stationary_track_ids(self) -> set[int]:
        pending = {track_id for track_id, _capture in self.stationary_queue}
        if self.camera_worker is not None and hasattr(self.camera_worker, "pending_track_ids"):
            pending |= self.camera_worker.pending_track_ids()
        if self.inference_worker is not None:
            pending.add(self.inference_worker.track_id)
        return pending

    def _capture_stationary_views(self, raw_frame, tracks: list[TrackedPart]) -> None:
        model = self.selected_model()
        for part in tracks:
            if not self.rotating_parts.accepts_inspection(part.track_id):
                continue
            capture = self.stationary_captures.get(part.track_id)
            if capture is None:
                capture = StationaryViewCapture(
                    burst_frames=model.capture_burst_frames, settle_ms=model.stationary_settle_ms,
                    motion_threshold=model.stationary_motion_threshold,
                    flow_threshold=model.stationary_flow_threshold,
                    skip_fit_rotation=model.skip_initial_fit_rotation,
                    minimum_burst_frames=model.stationary_min_burst_frames,
                    burst_window_ms=model.stationary_burst_window_ms,
                    quality=FrameQualityAnalyzer(max(model.minimum_sharpness, model.inference_min_sharpness),
                                               model.max_glare_ratio, model.max_saturation_ratio),
                )
                self.stationary_captures[part.track_id] = capture
            selected = capture.offer(crop_bounds(raw_frame, part.bounds))
            if selected is not None:
                # One queued still per stop; a slow inference must never replace
                # an earlier side with a newer side.
                self.stationary_queue.append((part.track_id, selected))
                self.log.addItem(f"CAPTURE {selected.angle}° | track={part.track_id} | burst={selected.burst_frames}")
            if not self.active_fail_asserted and self.inference_worker is None:
                self.status_badge.setObjectName("statusStandby")
                self.status_badge.setText(capture.state)
                self.status_badge.style().unpolish(self.status_badge)
                self.status_badge.style().polish(self.status_badge)
        retained = {part.track_id for part in tracks} | self._pending_stationary_track_ids()
        self.stationary_captures = {track_id: capture for track_id, capture in self.stationary_captures.items()
                                    if track_id in retained}

    def _dispatch_stationary_view(self) -> None:
        if self.inference_worker is not None or not self.stationary_queue:
            return
        track_id, selected = self.stationary_queue.popleft()
        if not self.rotating_parts.accepts_inspection(track_id):
            return
        if not selected.valid:
            result = InspectionResult("VIEW INVALID", 0, 0, 0, [], display_image=selected.frame,
                                      view_valid=False, reason_codes=selected.reasons,
                                      view_angle=selected.angle, view_quality_score=selected.sharpness)
            self._handle_inspection_result(track_id, result, 0)
            return
        self.inference_worker = InspectionWorker(self.inspector, self.inspection_model(), track_id,
                                                selected.frame, selected.angle)
        self._start_inference_worker(self.inference_worker)

    def _start_inference_worker(self, worker: InspectionWorker) -> None:
        generation = self.inspection_generation
        worker.finished_result.connect(
            lambda track_id, result, latency: self._handle_inspection_result(track_id, result, latency)
            if generation == self.inspection_generation else None
        )
        worker.failed.connect(
            lambda message: self._handle_live_error(f"Inspection error: {message}")
            if generation == self.inspection_generation else None
        )
        worker.finished.connect(lambda: self._clear_inference_worker(worker))
        worker.start()

    def _handle_inspection_result(self, track_id: int, result, latency_ms: float) -> None:
        if not self.inspection_running:
            return
        self.inference_latency_ms = latency_ms
        if result.is_no_part:
            self._handle_no_part_result(result, latency_ms)
            return
        completed_part = None
        latched_failure = result.status == "FAIL"
        view_progress = (0, 0, self.selected_model().minimum_rotation_views)
        if self.rotating_parts is not None:
            section_count = self.selected_model().patchcore_section_count
            defect_sections = tuple(result.candidate_sections)
            if result.status == "FAIL" and not defect_sections:
                image_width = result.display_image.shape[1] if result.display_image is not None else 0
                if image_width and result.defect_boxes:
                    defect_sections = tuple(sorted({
                        min(section_count - 1, max(0, int((x + width / 2) * section_count / image_width)))
                        for x, _y, width, _height in result.defect_boxes
                    }))
                else:
                    # A scalar-only legacy failure has no safe localization.
                    # Highlight the full component rather than lose the latched
                    # warning while it rotates.
                    defect_sections = tuple(range(section_count))
            completed_part = self.rotating_parts.record_inspection(
                track_id, is_pass=result.is_pass, anomaly_score=result.anomaly_score,
                view_valid=result.view_valid, immediate_failure=result.status == "FAIL",
                provisional_candidate=result.status == "CANDIDATE",
                geometry_score=result.geometry_score or 0.0,
                tao_score=result.tao_score, reason_codes=result.reason_codes,
                candidate_sections=defect_sections,
                view_angle=result.view_angle,
            )
            latched_failure = self.rotating_parts.latched_failure(track_id) or latched_failure
            if latched_failure and not self.active_fail_asserted:
                self.fail_output.send_result(True)
                self.active_fail_asserted = True
            view_progress = ((completed_part.frames_inspected, completed_part.valid_views,
                              self.rotating_parts.minimum_rotation_views) if completed_part is not None
                             else self.rotating_parts.view_progress(track_id))
        self.score_slider.setValue(int(result.anomaly_score * 1000))
        self.score_label.setText(f"{result.anomaly_score:.3f}")
        if latched_failure:
            badge_object, badge_text = "statusFail", "FAIL LATCHED"
        elif result.status == "PASS":
            # A good view is provisional until YOLO confirms part departure.
            # Do not show the final green PASS treatment while the part remains.
            badge_object, badge_text = "statusStandby", "VIEW OK — CHECKING"
        elif result.status == "VIEW INVALID":
            badge_object, badge_text = "statusStandby", "VIEW INVALID"
        else:
            badge_object, badge_text = "statusStandby", "INSPECTING"
        self.status_badge.setObjectName(badge_object)
        self.status_badge.setText(badge_text)
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)
        if result.display_image is not None:
            self.latest_annotated_frame = result.display_image
            self.show_frame(result.display_image)
        reason_text = ", ".join(result.reason_codes) or "NORMAL"
        geometry_components = result.geometry_components
        geometry_detail = (f"PITCH: {geometry_components.get('pitch', 0):.2f}   "
                           f"CONT: {geometry_components.get('continuity', 0):.2f}   "
                           f"BROKEN: {geometry_components.get('broken', 0):.2f}")
        self.last_result.setText(
            f"TRACK: {track_id}   VIEW SCORE: {result.anomaly_score:.2f}\n"
            f"PATCHCORE: {result.anomaly_score:.2f}   GEOMETRY: {result.geometry_score or 0:.2f}\n"
            f"GLARE: {result.glare_score or 0:.2f}   REGISTRATION: {result.registration_score or 0:.2f}\n"
            f"VIEW QUALITY: {result.view_quality_score if result.view_quality_score is not None else 1.0:.2f}\n"
            f"{geometry_detail}\n"
            f"VIEWS: {view_progress[1]}/{view_progress[2]} valid ({view_progress[0]} attempted)\n"
            f"ANGLE: {result.view_angle if result.view_angle is not None else '-'}°   "
            f"AREA: {result.anomaly_percentage:.2f}% / {self.tolerance_percent:.2f}% tolerance\n"
            f"REASON: {reason_text}\nLATENCY: {latency_ms:.1f} ms"
        )
        self.latency_top.setText(f"LATENCY:  {latency_ms:.0f} ms")
        self.log.addItem(f"VIEW {result.status} | track={track_id} | angle={result.view_angle}° | score={result.anomaly_score:.3f}")
        if completed_part is not None:
            self._handle_completed_part(completed_part)

    def _handle_completed_part(self, part) -> None:
        self.stats = self.daily_statistics.record(part.status)
        self.update_stats()
        self.status_badge.setObjectName("statusPass" if part.status == "PASS" else "statusFail")
        self.status_badge.setText(part.status)
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)
        if part.status == "PASS":
            # This is the first point at which a good view becomes a final part
            # verdict. Generate exactly one 0.5-second ESP32 output pulse now.
            self.fail_output.signal_pass()
        else:
            self.fail_output.send_result(True)
            self.active_fail_asserted = True
        self.log.addItem(f"FINAL {part.status} | track={part.track_id} | views={part.frames_inspected} | "
                        f"failed angles={part.failed_angles} | sections={part.defect_sections} | worst={part.worst_score:.3f}")

    @staticmethod
    def _draw_tracks(
        frame, tracks: list[TrackedPart], counting_line_ratio: float, counting_direction: str,
        counting_axis: str = "x", defect_sections: dict[int, tuple[int, ...]] | None = None,
        section_count: int = 1,
        defect_angles: dict[int, dict[int, tuple[int, ...]]] | None = None,
    ):
        display = frame.copy()
        if counting_axis == "y":
            line_y = int(round(display.shape[0] * counting_line_ratio))
            cv2.line(display, (0, line_y), (display.shape[1] - 1, line_y), (0, 255, 255), 2)
            marker = "v" if counting_direction == "top_to_bottom" else "^"
            cv2.putText(display, f"COUNT LINE {marker}", (8, max(20, line_y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        else:
            line_x = int(round(display.shape[1] * counting_line_ratio))
            cv2.line(display, (line_x, 0), (line_x, display.shape[0] - 1), (0, 255, 255), 2)
            marker = ">" if counting_direction == "left_to_right" else "<"
            cv2.putText(display, f"COUNT LINE {marker}", (max(4, line_x - 135), 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        for track in tracks:
            x, y, w, h = track.bounds
            sections = (defect_sections or {}).get(track.track_id, ())
            for section in sections:
                sx0 = x + round(w * section / max(section_count, 1))
                sx1 = x + round(w * (section + 1) / max(section_count, 1))
                overlay = display.copy()
                cv2.rectangle(overlay, (sx0, y), (sx1, y + h), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.28, display, 0.72, 0, display)
                cv2.rectangle(display, (sx0, y), (sx1, y + h), (0, 0, 255), 3)
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 217, 255), 2)
            angles = sorted({angle for values in (defect_angles or {}).get(track.track_id, {}).values() for angle in values})
            angle_hint = f" @ {','.join(str(angle) for angle in angles)} DEG" if angles else ""
            label = (f"DEFECT SECTION {','.join(str(value + 1) for value in sections)}{angle_hint} - ROTATE TO VERIFY"
                     if sections else f"PART {track.track_id} {track.confidence:.0%}")
            colour = (0, 0, 255) if sections else (0, 217, 255)
            cv2.putText(display, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
        return display

    def _handle_no_part_frame(self, frame) -> None:
        result = InspectionResult(
            status="NO PART",
            anomaly_score=0.0,
            defect_area_px=0,
            bad_sector_ratio=0.0,
            bad_sectors=[],
            display_image=frame,
        )
        self._handle_no_part_result(result, 0.0)

    def _handle_no_part_result(self, result, latency_ms: float) -> None:
        self.status_badge.setObjectName("statusNoPart")
        self.status_badge.setText("NO PART")
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)
        if result.display_image is not None:
            self.latest_annotated_frame = result.display_image
            self.show_frame(result.display_image)
        self.score_slider.setValue(0)
        self.score_label.setText("0.000")
        self.last_result.setText(
            f"FRAME:  -\nSCORE:  -\nCOVERAGE:  -\nBOXES:  -\n"
            f"BAD SECTORS:  -\nSECTOR RATIO:  -\nLATENCY:  {latency_ms:.1f} ms"
        )
        self.latency_top.setText(f"LATENCY:  {latency_ms:.0f} ms")
        self.log.addItem(f"NO PART | {self.selected_model().id}")

    def _send_fail_output(self, result) -> None:
        self.fail_output.send_result(not result.is_pass)

    def _clear_inference_worker(self, worker: InspectionWorker | None = None) -> None:
        worker = worker or self.inference_worker
        if worker is not None:
            worker.deleteLater()
        if self.inference_worker is worker:
            self.inference_worker = None

    def _handle_live_error(self, message: str) -> None:
        if not self.inspection_running:
            return
        self.stop_camera()
        # A runtime/model/calibration fault is never equivalent to a good part.
        # Stop acquisition and assert the reject/inhibit output until an
        # operator explicitly restarts a healthy inspection session.
        self.fail_output.send_result(True)
        self.status_badge.setObjectName("statusFail")
        self.status_badge.setText("SYSTEM FAULT")
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)
        self.log.addItem(f"INSPECTION INHIBITED | {message}")
        QMessageBox.critical(self, "Live inspection stopped", message)

    def inspect_current(self) -> None:
        self.start_inspection()

    def show_frame(self, frame) -> None:
        started = time.monotonic()
        self.current_display_frame = preview_image(frame, self.viewer.width(), self.viewer.height())
        self._paint_frame_to_viewer()
        self.preview_meter.record(time.monotonic() - started)

    def _paint_frame_to_viewer(self) -> None:
        if self.current_display_frame is None:
            return
        frame = self.current_display_frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        rgb = rgb.copy()
        h, w, ch = rgb.shape
        qimage = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
        target_size = self.viewer.contentsRect().size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            target_size = self.viewer.size()
        pixmap = QPixmap.fromImage(qimage).scaled(target_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        self.viewer.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._paint_frame_to_viewer()


    def closeEvent(self, event) -> None:
        self.inspection_running = False
        self.inspection_generation += 1
        self.live_timer.stop()
        if not self._stop_camera_worker():
            event.ignore()
            return
        for worker in (self.presence_worker, self.inference_worker):
            if worker is not None and worker.isRunning() and not worker.wait(1000):
                self.log.addItem("WAITING FOR INSPECTION WORKER TO STOP")
                event.ignore()
                return
        self.fail_output.reset()
        self.active_fail_asserted = False
        self.fail_output.close()
        super().closeEvent(event)

    def train_selected(self) -> None:
        if not self.user.is_admin:
            QMessageBox.warning(self, "Permission denied", "Training is available to admin users only.")
            return
        if self.inspection_running:
            QMessageBox.warning(self, "Inspection active", "Stop inspection before training this model.")
            return
        if self.train_worker is not None and self.train_worker.isRunning():
            QMessageBox.information(self, "Training active", "PatchCore training is already running.")
            return
        model = self.selected_model()
        tao_production = model.production_algorithm != "patchcore_geometry" and model.algorithm == "nvidia_tao"
        if tao_production and not model.model_file.is_file():
            path, _ = QFileDialog.getOpenFileName(
                self,
                f"Select NVIDIA TAO ONNX export for {model.id}",
                str(model.model_file.parent),
                "ONNX models (*.onnx)",
            )
            if not path:
                QMessageBox.information(
                    self,
                    "TAO model required",
                    "Calibration was cancelled. Export the trained model from NVIDIA TAO as ONNX, "
                    "then select that file here. Good images calibrate its production thresholds; "
                    "this desktop application does not replace TAO training.",
                )
                return
            model = self.registry.update_model_settings(model.id, model_file=Path(path).resolve())
            self._refresh_models(selected_id=model.id)
            self.inspector = inspector_for_model(model)
        operation = "TAO CALIBRATION" if tao_production else "PATCHCORE TRAINING"
        self.log.addItem(f"{operation} STARTED {model.id}")
        self.train_worker = TrainWorker(inspector_for_model(model), model)
        self.train_worker.finished_ok.connect(lambda message: self.log.addItem(message))
        self.train_worker.progress.connect(self._show_training_progress)
        self.train_worker.failed.connect(lambda message: QMessageBox.critical(self, "Training failed", message))
        self.train_worker.start()

    def export_selected(self) -> None:
        if not self.user.is_admin:
            QMessageBox.warning(self, "Permission denied", "TAO export is available to admin users only.")
            return
        model = self.selected_model()
        self.log.addItem(f"TAO EXPORT STARTED {model.id}")
        self.export_worker = TaoExportWorker(model)
        self.export_worker.progress.connect(self._show_training_progress)
        self.export_worker.finished_ok.connect(lambda message: self.log.addItem(message))
        self.export_worker.failed.connect(lambda message: QMessageBox.critical(self, "Export failed", message))
        self.export_worker.start()

    def _show_training_progress(self, message: str) -> None:
        """Keep the newest worker update visible to the operator in real time."""
        self.log.addItem(message)
        self.log.scrollToBottom()

    def stop_camera(self) -> None:
        self.inspection_generation += 1
        if self.rotating_parts is not None:
            for part in self.rotating_parts.flush():
                self._handle_completed_part(part)
        self.part_detector = None
        self.rotating_parts = None
        self.inspection_running = False
        self.live_timer.stop()
        self._stop_camera_worker()
        self.fail_output.reset()
        self.active_fail_asserted = False
        self.latest_annotated_frame = None
        self.current_display_frame = None
        self.live_roi_bounds = None
        self.locked_roi_confidence = 0.0
        self.locked_part_present = False
        self.locked_missing_frames = 0
        self.locked_presence_poll = 0
        self.locked_last_yolo_present = False
        self.raw_frame = None
        self.pending_sharp_frames = {}
        self.stationary_captures.clear()
        self.stationary_queue.clear()
        if self.frame_sampler is not None:
            self.frame_sampler.clear()
        self.frame_sampler = None
        if self.rotation_phase_gate is not None:
            self.rotation_phase_gate.clear()
        self.rotation_phase_gate = None
        self.viewer.clear()
        self.viewer.setText("NO CAMERA FRAME")
        self.fps_top.setText("FPS:  -")
        self.online_label.setText("● OFFLINE")
        self.status_badge.setObjectName("statusStandby")
        self.status_badge.setText("STANDBY")
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)

    def reset_stats(self) -> None:
        if self.daily_statistics.memory_only:
            self.daily_statistics.clear()
        self.stats = self.daily_statistics.counts()
        self.update_stats()
        message = "TEMPORARY COUNTERS RESET" if self.daily_statistics.memory_only else f"DAILY COUNTERS RETAINED ({operating_day()} 07:00–07:00)"
        self.log.addItem(message)

    def update_stats(self) -> None:
        inspected = self.stats["inspected"]
        pass_rate = "-" if inspected == 0 else f"{100 * self.stats['passed'] / inspected:.1f}%"
        self.inspected_value.value_label.setText(str(inspected))
        self.passed_value.value_label.setText(str(self.stats["passed"]))
        self.failed_value.value_label.setText(str(self.stats["failed"]))
        self.pass_rate_value.value_label.setText(pass_rate)

    def show_manual_tolerance(self) -> None:
        self.manual_tolerance.setVisible(True)
        self.manual_tolerance.setFocus()
        self.manual_tolerance.selectAll()

    def set_tolerance(self, value: float) -> None:
        self.tolerance_percent = max(0.0, min(float(value), 50.0))
        formatted = f"{self.tolerance_percent:.2f}".rstrip("0").rstrip(".")
        self.tolerance_top.setText(f"TOLERANCE:  {formatted}%")
        self.log.addItem(f"TOLERANCE SET TO {formatted}%")

    def _tick(self) -> None:
        self.time_label.setText(time.strftime("%H:%M:%S"))
        self.speed_top.setText(f"SURFACE SPEED:  {self.speed_slider.value() / 10:.1f} m/s")
        if self.inspection_running:
            snapshot = self.camera_worker.snapshot() if self.camera_worker is not None else None
            if snapshot is not None:
                fps = snapshot.fps
            else:
                elapsed = max(time.perf_counter() - self.fps_started_at, 0.001)
                fps = self.fps_frame_count / elapsed
            self.fps_top.setText(f"FPS:  {fps:.1f}")
            if self.camera_worker is not None and hasattr(self.camera_worker, "performance"):
                performance = self.camera_worker.performance()
                performance["gui"] = self.preview_meter.snapshot()
                performance["inference_ms"] = self.inference_latency_ms
                performance["ui_inspection_queue"] = len(self.stationary_queue)
                fmt = performance["format"]
                text = (f"RAW {performance['raw']['fps']:.1f} PROC {performance['processing']['fps']:.1f} "
                        f"GUI {performance['gui']['fps']:.1f} FPS\n"
                        f"READ {performance['raw']['mean_ms']:.1f} ms | INFER {self.inference_latency_ms:.1f} ms\n"
                        f"DROP {performance['evidence_dropped']} | PREVIEW OVERWRITTEN {performance['preview_overwritten']}\n"
                        f"{fmt.get('width', '?')}x{fmt.get('height', '?')} {fmt.get('fourcc', '?')} {fmt.get('backend', '?')}")
                base = self.last_result.text().split("\nRAW ")[0]
                self.last_result.setText(base + "\n" + text)
                details = json.dumps(performance, indent=2)
                self.fps_top.setToolTip(details)
                self.last_result.setToolTip(details)
                log_path = os.environ.get("NEUROIRIS_PERFORMANCE_LOG")
                if log_path:
                    try:
                        with Path(log_path).open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps({"monotonic": time.monotonic(), **performance}) + "\n")
                    except OSError as exc:
                        self.log.addItem(f"PERFORMANCE LOG ERROR | {exc}")


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    auth = AuthStore()
    login = LoginDialog(auth)
    if login.exec() != QDialog.DialogCode.Accepted or login.user is None:
        return
    window = InspectionWindow(login.user)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
