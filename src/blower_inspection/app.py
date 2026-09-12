"""PyQt6 industrial UI for blower fan inspection."""

from __future__ import annotations

import sys
import time
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
from .inspector_factory import inspector_for_model
from .trainer import InspectionResult
from .tao_training import run_visual_changenet_task
from .yolo_tracking import RotatingPartInspector, TrackedPart, YoloByteTrackDetector


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

    def __init__(self, inspector, model: PartModelConfig) -> None:
        super().__init__()
        self.inspector = inspector
        self.model = model

    def run(self) -> None:
        try:
            output = self.inspector.train(self.model)
            action = "CALIBRATED" if self.model.algorithm == "nvidia_tao" else "TRAINED"
            self.finished_ok.emit(f"{action} {self.model.id}: {output}")
        except Exception as exc:
            self.failed.emit(f"TRAINING FAILED {self.model.id}: {exc}")


class InspectionWorker(QThread):
    finished_result = pyqtSignal(int, object, float)
    failed = pyqtSignal(str)

    def __init__(self, inspector, model: PartModelConfig, track_id: int, frame) -> None:
        super().__init__()
        self.inspector = inspector
        self.model = model
        self.track_id = track_id
        self.frame = frame.copy()

    def run(self) -> None:
        start = time.perf_counter()
        try:
            result = self.inspector.inspect(self.model, self.frame, save_outputs=False, crop_to_component=False)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished_result.emit(self.track_id, result, (time.perf_counter() - start) * 1000.0)


class TaoExportWorker(QThread):
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, model: PartModelConfig) -> None:
        super().__init__()
        self.model = model

    def run(self) -> None:
        spec = Path(f"specs/visual_changenet/{self.model.id.lower()}_segmentation.yaml")
        results = Path(f"data/results/{self.model.id}/tao")
        try:
            run_visual_changenet_task(
                "export", spec, results_dir=results, export_file=self.model.model_file
            )
            self.finished_ok.emit(f"EXPORTED {self.model.id}: {self.model.model_file}")
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
        self.inference_worker: InspectionWorker | None = None
        self.last_inference_at = 0.0
        self.inference_interval_s = 0.05
        self.latest_annotated_frame = None
        self.current_display_frame = None
        self.live_roi_bounds = None
        self.raw_frame = None
        self.fps_frame_count = 0
        self.fps_started_at = time.perf_counter()
        self.tolerance_percent = 5
        self.daily_statistics = DailyStatistics()
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
        for index, value in enumerate([1, 5, 8, 10, 13, 15, 18, 20]):
            button = QPushButton(f"{value}%")
            if value == 5:
                button.setStyleSheet("border-color:#d29b00;color:#f5c542;")
            button.clicked.connect(lambda _checked=False, v=value: self.set_tolerance(v))
            tolerance_grid.addWidget(button, index // 4, index % 4)
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
            export = QPushButton("⬡   EXPORT TAO MODEL")
            export.setObjectName("train")
            export.clicked.connect(self.export_selected)
            layout.addWidget(export)
            train = QPushButton("◆   CALIBRATE TAO MODEL")
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
        # The tolerance buttons are operator-facing strictness controls.  Lower
        # percentages must reject smaller detected regions/sectors, while higher
        # percentages allow larger confirmed defects before rejecting the part.
        strictness_scale = max(self.tolerance_percent, 1) / 5.0
        tolerance_area = max(1, int(model.min_defect_area_px * strictness_scale))
        tolerance_threshold = max(model.anomaly_threshold, (0.65 + 0.30 * (self.tolerance_percent / 20.0)) * 10.0)
        return replace(
            model,
            anomaly_threshold=tolerance_threshold,
            min_defect_area_px=tolerance_area,
            max_bad_sector_ratio=max(0.01, self.tolerance_percent / 100.0),
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
        if self.inspection_running:
            return
        model = self.selected_model()
        if model.yolo_model_path is None:
            QMessageBox.critical(self, "YOLO model required", "Select your trained YOLO best.pt with YOLO PART MODEL before starting live inspection.")
            return
        if model.algorithm == "nvidia_tao":
            try:
                self.inspector.validate_ready(model)
            except Exception as exc:
                QMessageBox.critical(self, "TAO model not ready", str(exc))
                return
        try:
            self._apply_selected_camera_settings()
            self.part_detector = YoloByteTrackDetector(model.yolo_model_path, model.yolo_confidence)
            self.rotating_parts = RotatingPartInspector(
                model.inspection_lost_timeout_s,
                model.counting_line_ratio,
                model.counting_direction,
            )
            self.camera.open()
        except Exception as exc:
            QMessageBox.critical(self, "Camera error", str(exc))
            return
        self.inspection_running = True
        self.online_label.setText("● ONLINE")
        self.status_badge.setObjectName("statusStandby")
        self.status_badge.setText("RUNNING")
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)
        self.fps_frame_count = 0
        self.fps_started_at = time.perf_counter()
        self.last_inference_at = 0.0
        self.latest_annotated_frame = None
        self.current_display_frame = None
        self.live_roi_bounds = None
        self.raw_frame = None
        try:
            device_name = self.inspector.runtime_device_name()
        except Exception as exc:
            device_name = f"unavailable ({exc})"
        if self.fail_output.config.enabled:
            # ESP32FailOutputBridge is deliberately asynchronous and exposes
            # send_result/reset rather than the synchronous connect/set_fail
            # API of the legacy serial driver. Queue a safe PASS state without
            # blocking camera startup.
            self.fail_output.reset()
            esp32_status = f"ESP32 CONFIGURED {self.fail_output.config.base_url}"
        else:
            esp32_status = "ESP32 DISABLED"
        self.log.addItem(
            f"LIVE INSPECTION STARTED {self.selected_model().id} | "
            f"CAMERA {self.camera.width}x{self.camera.height}@{self.camera.fps} | DEVICE {device_name} | {esp32_status}"
        )
        self.live_timer.start(1)

    def _process_live_frame(self) -> None:
        if not self.inspection_running:
            return
        try:
            raw_frame = self.camera.read()
            self.raw_frame = raw_frame
            assert self.part_detector is not None and self.rotating_parts is not None
            tracks = self.part_detector.track(raw_frame)
            self.rotating_parts.observe_tracks(tracks, raw_frame.shape[1])
            frame = self._draw_tracks(
                raw_frame,
                tracks,
                self.rotating_parts.counting_line_ratio,
                self.rotating_parts.counting_direction,
            )
        except Exception as exc:
            self._handle_live_error(f"Camera frame error: {exc}")
            return
        self.frame = frame
        if self.latest_annotated_frame is None:
            self.show_frame(frame)
        self.fps_frame_count += 1
        if not tracks:
            self._handle_no_part_frame(frame)
            return
        now = time.perf_counter()
        if (
            self.inference_worker is None
            and now - self.last_inference_at >= self.inference_interval_s
        ):
            self.last_inference_at = now
            # Give each simultaneously tracked part one exact-crop inspection
            # before spending more frames on any already observed rotation.
            assert self.rotating_parts is not None
            part = max(
                tracks,
                key=lambda candidate: (
                    self.rotating_parts.needs_completion_inspection(candidate.track_id),
                    self.rotating_parts.needs_initial_inspection(candidate.track_id),
                    candidate.confidence,
                ),
            )
            self.inference_worker = InspectionWorker(
                self.inspector, self.inspection_model(), part.track_id, crop_bounds(raw_frame, part.bounds)
            )
            self.inference_worker.finished_result.connect(self._handle_inspection_result)
            self.inference_worker.failed.connect(lambda message: self._handle_live_error(f"Inspection error: {message}"))
            self.inference_worker.finished.connect(self._clear_inference_worker)
            self.inference_worker.start()

    def _handle_inspection_result(self, track_id: int, result, latency_ms: float) -> None:
        if not self.inspection_running:
            return
        if result.is_no_part:
            self._handle_no_part_result(result, latency_ms)
            return
        completed_part = None
        latched_failure = not result.is_pass
        if self.rotating_parts is not None:
            completed_part = self.rotating_parts.record_inspection(
                track_id, is_pass=result.is_pass, anomaly_score=result.anomaly_score
            )
            latched_failure = self.rotating_parts.latched_failure(track_id) or latched_failure
        self.score_slider.setValue(int(result.anomaly_score * 1000))
        self.score_label.setText(f"{result.anomaly_score:.3f}")
        self.status_badge.setObjectName("statusFail" if latched_failure else "statusStandby")
        self.status_badge.setText("FAIL LATCHED" if latched_failure else "INSPECTING")
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)
        if result.display_image is not None:
            self.latest_annotated_frame = result.display_image
            self.show_frame(result.display_image)
        bad_sector_text = ",".join(str(sector) for sector in result.bad_sectors[:8]) if result.bad_sectors else "-"
        if len(result.bad_sectors) > 8:
            bad_sector_text += ",..."
        self.last_result.setText(
            f"TRACK:  {track_id} (rotation in progress)\nSCORE:  {result.anomaly_score:.3f}\n"
            f"COVERAGE:  {result.defect_area_px}px\nBOXES:  {len(result.defect_boxes)}\n"
            f"BAD SECTORS:  {bad_sector_text}\nSECTOR RATIO:  {result.bad_sector_ratio:.2%}\n"
            f"LATENCY:  {latency_ms:.1f} ms"
        )
        self.latency_top.setText(f"LATENCY:  {latency_ms:.0f} ms")
        self.log.addItem(f"VIEW {result.status} | track={track_id} | score={result.anomaly_score:.3f}")
        if completed_part is not None:
            self._handle_completed_part(completed_part)

    def _handle_completed_part(self, part) -> None:
        self.stats = self.daily_statistics.record(part.status)
        self.update_stats()
        self.status_badge.setObjectName("statusPass" if part.status == "PASS" else "statusFail")
        self.status_badge.setText(part.status)
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)
        self.fail_output.send_result(part.status == "FAIL")
        self.log.addItem(f"FINAL {part.status} | track={part.track_id} | views={part.frames_inspected} | worst={part.worst_score:.3f}")

    @staticmethod
    def _draw_tracks(
        frame, tracks: list[TrackedPart], counting_line_ratio: float, counting_direction: str
    ):
        display = frame.copy()
        line_x = int(round(display.shape[1] * counting_line_ratio))
        cv2.line(display, (line_x, 0), (line_x, display.shape[0] - 1), (0, 255, 255), 2)
        cv2.putText(
            display,
            "COUNT LINE " + (">" if counting_direction == "left_to_right" else "<"),
            (max(4, line_x - 135), 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        for track in tracks:
            x, y, w, h = track.bounds
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 217, 255), 2)
            cv2.putText(display, f"PART {track.track_id} {track.confidence:.0%}", (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 217, 255), 2)
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

    def _clear_inference_worker(self) -> None:
        if self.inference_worker is not None:
            self.inference_worker.deleteLater()
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
        self.current_display_frame = frame.copy()
        self._paint_frame_to_viewer()

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
        self.fail_output.reset()
        self.fail_output.close()
        super().closeEvent(event)

    def train_selected(self) -> None:
        if not self.user.is_admin:
            QMessageBox.warning(self, "Permission denied", "Training is available to admin users only.")
            return
        model = self.selected_model()
        if model.algorithm == "nvidia_tao" and not model.model_file.is_file():
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
        self.log.addItem(f"TAO CALIBRATION STARTED {model.id}")
        self.train_worker = TrainWorker(self.inspector, model)
        self.train_worker.finished_ok.connect(lambda message: self.log.addItem(message))
        self.train_worker.failed.connect(lambda message: QMessageBox.critical(self, "Training failed", message))
        self.train_worker.start()

    def export_selected(self) -> None:
        if not self.user.is_admin:
            QMessageBox.warning(self, "Permission denied", "TAO export is available to admin users only.")
            return
        model = self.selected_model()
        if model.algorithm != "nvidia_tao":
            QMessageBox.warning(self, "Wrong model type", f"{model.id} is not a TAO model.")
            return
        self.log.addItem(f"TAO EXPORT STARTED {model.id}")
        self.export_worker = TaoExportWorker(model)
        self.export_worker.finished_ok.connect(lambda message: self.log.addItem(message))
        self.export_worker.failed.connect(lambda message: QMessageBox.critical(self, "Export failed", message))
        self.export_worker.start()

    def stop_camera(self) -> None:
        if self.rotating_parts is not None:
            for part in self.rotating_parts.flush():
                self._handle_completed_part(part)
        self.part_detector = None
        self.rotating_parts = None
        self.inspection_running = False
        self.live_timer.stop()
        self.camera.close()
        self.fail_output.reset()
        self.latest_annotated_frame = None
        self.current_display_frame = None
        self.live_roi_bounds = None
        self.raw_frame = None
        self.viewer.clear()
        self.viewer.setText("NO CAMERA FRAME")
        self.fps_top.setText("FPS:  -")
        self.online_label.setText("● OFFLINE")
        self.status_badge.setObjectName("statusStandby")
        self.status_badge.setText("STANDBY")
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)

    def reset_stats(self) -> None:
        self.stats = self.daily_statistics.counts()
        self.update_stats()
        self.log.addItem(f"DAILY COUNTERS RETAINED ({operating_day()} 07:00–07:00)")

    def update_stats(self) -> None:
        inspected = self.stats["inspected"]
        pass_rate = "-" if inspected == 0 else f"{100 * self.stats['passed'] / inspected:.1f}%"
        self.inspected_value.value_label.setText(str(inspected))
        self.passed_value.value_label.setText(str(self.stats["passed"]))
        self.failed_value.value_label.setText(str(self.stats["failed"]))
        self.pass_rate_value.value_label.setText(pass_rate)

    def set_tolerance(self, value: int) -> None:
        self.tolerance_percent = value
        self.tolerance_top.setText(f"TOLERANCE:  {value}%")
        self.log.addItem(f"TOLERANCE SET TO {value}%")

    def _tick(self) -> None:
        self.time_label.setText(time.strftime("%H:%M:%S"))
        self.speed_top.setText(f"SURFACE SPEED:  {self.speed_slider.value() / 10:.1f} m/s")
        if self.inspection_running:
            elapsed = max(time.perf_counter() - self.fps_started_at, 0.001)
            self.fps_top.setText(f"FPS:  {self.fps_frame_count / elapsed:.1f}")


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
