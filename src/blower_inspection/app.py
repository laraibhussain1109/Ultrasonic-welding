"""PyQt6 industrial UI for blower fan inspection."""

from __future__ import annotations

import sys
import time
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
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from .anomaly_models import HybridPatchcorePadimInspector
from .auth import AuthStore, User
from .camera import USBCamera, save_capture
from .config import ModelRegistry, PartModelConfig, ensure_model_folders


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

    def __init__(self, inspector: HybridPatchcorePadimInspector, model: PartModelConfig) -> None:
        super().__init__()
        self.inspector = inspector
        self.model = model

    def run(self) -> None:
        try:
            output = self.inspector.train(self.model)
            self.finished_ok.emit(f"TRAINED {self.model.id}: {output}")
        except Exception as exc:
            self.failed.emit(f"TRAINING FAILED {self.model.id}: {exc}")


class InspectionWindow(QWidget):
    def __init__(self, user: User) -> None:
        super().__init__()
        self.user = user
        self.registry = ModelRegistry()
        ensure_model_folders(self.registry)
        self.inspector = HybridPatchcorePadimInspector()
        self.camera = USBCamera(width=3840, height=2160)
        self.frame = None
        self.stats = {"inspected": 0, "passed": 0, "failed": 0}
        self.started_at = time.time()
        self.train_worker: TrainWorker | None = None
        self.setWindowTitle(f"NeuroIris Blower Fan Inspection - {user.username} ({user.role})")
        self.resize(1884, 940)
        self._build_ui()
        self.clock = QTimer(self)
        self.clock.timeout.connect(self._tick)
        self.clock.start(500)

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
        self.model_by_label = {f"{m.id}  |  {m.name}": m for m in self.registry.all()}
        self.model_combo.addItems(self.model_by_label.keys())
        layout.addWidget(self.model_combo)
        start = QPushButton("▶   START INSPECTION")
        start.setObjectName("primary")
        start.clicked.connect(self.inspect_current)
        layout.addWidget(start)
        calibrate = QPushButton("⊕   CALIBRATE")
        calibrate.clicked.connect(self.load_image)
        layout.addWidget(calibrate)
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
            train = QPushButton("◆   TRAIN SELECTED MODEL")
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
        self.viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
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

    def load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load inspection image", str(Path.cwd()), "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            QMessageBox.critical(self, "Image error", f"Unable to read {path}")
            return
        self.frame = frame
        self.show_frame(frame)
        self.log.addItem(f"LOADED {Path(path).name}")

    def inspect_current(self) -> None:
        if self.frame is None:
            try:
                self.camera.open()
                self.frame = self.camera.read()
                self.online_label.setText("● ONLINE")
            except Exception as exc:
                QMessageBox.critical(self, "Camera error", str(exc))
                return
        start = time.perf_counter()
        try:
            result = self.inspector.inspect(self.selected_model(), self.frame)
        except Exception as exc:
            QMessageBox.critical(self, "Inspection error", str(exc))
            return
        latency_ms = (time.perf_counter() - start) * 1000.0
        self.stats["inspected"] += 1
        self.stats["passed" if result.is_pass else "failed"] += 1
        self.update_stats()
        self.score_slider.setValue(int(result.anomaly_score * 1000))
        self.score_label.setText(f"{result.anomaly_score:.3f}")
        self.status_badge.setObjectName("statusPass" if result.is_pass else "statusFail")
        self.status_badge.setText(result.status)
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)
        self.last_result.setText(
            f"FRAME:  {self.stats['inspected']}\nSCORE:  {result.anomaly_score:.3f}\n"
            f"COVERAGE:  {result.defect_area_px}px\nLATENCY:  {latency_ms:.1f} ms"
        )
        self.latency_top.setText(f"LATENCY:  {latency_ms:.0f} ms")
        self.log.addItem(f"{result.status} | {self.selected_model().id} | score={result.anomaly_score:.3f}")
        if result.overlay_path:
            overlay = cv2.imread(str(result.overlay_path))
            if overlay is not None:
                self.show_frame(overlay)

    def show_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        h, w, ch = rgb.shape
        qimage = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage).scaled(self.viewer.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.viewer.setPixmap(pixmap)

    def train_selected(self) -> None:
        if not self.user.is_admin:
            QMessageBox.warning(self, "Permission denied", "Training is available to admin users only.")
            return
        model = self.selected_model()
        self.log.addItem(f"TRAINING STARTED {model.id}")
        self.train_worker = TrainWorker(self.inspector, model)
        self.train_worker.finished_ok.connect(lambda message: self.log.addItem(message))
        self.train_worker.failed.connect(lambda message: QMessageBox.critical(self, "Training failed", message))
        self.train_worker.start()

    def stop_camera(self) -> None:
        self.camera.close()
        self.online_label.setText("● OFFLINE")
        self.status_badge.setObjectName("statusStandby")
        self.status_badge.setText("STANDBY")
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)

    def reset_stats(self) -> None:
        self.stats = {"inspected": 0, "passed": 0, "failed": 0}
        self.update_stats()
        self.log.clear()

    def update_stats(self) -> None:
        inspected = self.stats["inspected"]
        pass_rate = "-" if inspected == 0 else f"{100 * self.stats['passed'] / inspected:.1f}%"
        self.inspected_value.value_label.setText(str(inspected))
        self.passed_value.value_label.setText(str(self.stats["passed"]))
        self.failed_value.value_label.setText(str(self.stats["failed"]))
        self.pass_rate_value.value_label.setText(pass_rate)

    def set_tolerance(self, value: int) -> None:
        self.tolerance_top.setText(f"TOLERANCE:  {value}%")
        self.log.addItem(f"TOLERANCE SET TO {value}%")

    def _tick(self) -> None:
        self.time_label.setText(time.strftime("%H:%M:%S"))
        self.speed_top.setText(f"SURFACE SPEED:  {self.speed_slider.value() / 10:.1f} m/s")


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
