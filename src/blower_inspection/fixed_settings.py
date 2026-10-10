"""Validated, persistent settings for the stationary six-view PatchCore workflow."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass(frozen=True)
class CameraSettings:
    index: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 30
    exposure: float = -6.0
    gain: float = 0.0
    burst_frames: int = 15


@dataclass(frozen=True)
class QualitySettings:
    minimum_blur_score: float = 60.0
    maximum_dark_percent: float = 55.0
    maximum_saturated_percent: float = 12.0
    minimum_mean_intensity: float = 10.0
    maximum_mean_intensity: float = 245.0
    blur_weight: float = 1.0
    exposure_penalty: float = 1.0


@dataclass(frozen=True)
class YoloSettings:
    model_path: str = "data/models/yolo/best.pt"
    confidence: float = 0.70
    iou: float = 0.45
    padding_x_percent: float = 4.0
    padding_y_percent: float = 4.0
    class_id: int = -1
    show_roi: bool = True


@dataclass(frozen=True)
class PatchCoreSettings:
    model_path: str = "data/models/BF-001/patchcore_fixed.pt"
    normal_image_dir: str = "data/training/BF-001/normal"
    input_width: int = 640
    input_height: int = 256
    backbone: str = "wide_resnet50_2"
    embedding_layers: tuple[str, ...] = ("layer2", "layer3")
    memory_bank_size: int = 8192
    coreset_ratio: float = 0.08
    max_training_images: int = 300
    angle_specific: bool = False
    angle_model_paths: dict[str, str] = field(default_factory=dict)
    alignment_enabled: bool = False
    alignment_max_translation_percent: float = 2.0
    alignment_min_correlation: float = 0.8
    heatmap_display_max: float = 1.0

    def path_for_angle(self, angle: int) -> Path:
        if not self.angle_specific:
            return Path(self.model_path)
        value = self.angle_model_paths.get(str(angle))
        if not value:
            raise ValueError(f"No angle-specific PatchCore checkpoint configured for {angle}°")
        return Path(value)


@dataclass(frozen=True)
class ToleranceSettings:
    heatmap_threshold: float = 0.43
    minimum_component_pixels: int = 500
    pixel_threshold: int = 2500
    percentage_threshold: float = 0.15
    decision_mode: str = "BOTH"
    ignore_top_percent: float = 5.0
    ignore_bottom_percent: float = 5.0
    ignore_left_percent: float = 5.0
    ignore_right_percent: float = 5.0
    opening_kernel: int = 0
    closing_kernel: int = 0


@dataclass(frozen=True)
class MachineSettings:
    views: int = 6
    angle_step: int = 60
    settle_delay_s: float = 0.15
    acquisition_time_s: float = 0.65
    minimum_burst_frames: int = 10
    position_timeout_s: float = 2.0
    cycle_timeout_s: float = 60.0
    signal_timeout_s: float = 1.0
    interface: str = "manual"
    host: str = "127.0.0.1"
    port: int = 8765
    output_enabled: bool = False


@dataclass(frozen=True)
class StorageSettings:
    results_dir: str = "data/results/fixed_inspection"
    save_pass_images: bool = False
    save_fail_images: bool = True
    save_heatmaps: bool = True


@dataclass(frozen=True)
class InspectionSettings:
    camera: CameraSettings = field(default_factory=CameraSettings)
    quality: QualitySettings = field(default_factory=QualitySettings)
    yolo: YoloSettings = field(default_factory=YoloSettings)
    patchcore: PatchCoreSettings = field(default_factory=PatchCoreSettings)
    tolerance: ToleranceSettings = field(default_factory=ToleranceSettings)
    machine: MachineSettings = field(default_factory=MachineSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)

    def validate(self) -> "InspectionSettings":
        c, q, y, p, t, m = self.camera, self.quality, self.yolo, self.patchcore, self.tolerance, self.machine
        for section in (c, q, y, p, t, m):
            for item in fields(section):
                value = getattr(section, item.name)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
                    raise ValueError(f"{item.name} must be finite")
                expected = getattr(type(section)(), item.name)
                if isinstance(expected, float) and (not isinstance(value, (int, float)) or isinstance(value, bool)):
                    raise ValueError(f"{item.name} must be numeric")
                if isinstance(expected, bool) and not isinstance(value, bool):
                    raise ValueError(f"{item.name} must be boolean")
                if isinstance(expected, int) and not isinstance(expected, bool) and (not isinstance(value, int) or isinstance(value, bool)):
                    raise ValueError(f"{item.name} must be an integer")
                if isinstance(expected, str) and not isinstance(value, str):
                    raise ValueError(f"{item.name} must be text")
        if not isinstance(p.angle_model_paths, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in p.angle_model_paths.items()):
            raise ValueError("Angle model paths must map angle strings to filesystem paths")
        if min(c.width, c.height, c.fps) < 1 or c.index < 0 or not 10 <= c.burst_frames <= 30:
            raise ValueError("Camera dimensions/FPS must be positive; burst_frames must be 10–30")
        if min(p.input_width, p.input_height) < 32 or p.input_width % 8 or p.input_height % 8:
            raise ValueError("PatchCore dimensions must be multiples of 8, at least 32")
        if p.backbone not in {"wide_resnet50_2", "resnet18"} or not p.embedding_layers or not set(p.embedding_layers) <= {"layer1", "layer2", "layer3", "layer4"}:
            raise ValueError("Unsupported PatchCore backbone/layers")
        if not 0 < p.coreset_ratio <= 1 or p.memory_bank_size < 1 or p.max_training_images < 20:
            raise ValueError("Invalid PatchCore memory-bank/training limits")
        if not 0 < y.confidence <= 1 or not 0 < y.iou <= 1 or y.class_id < -1:
            raise ValueError("YOLO confidence and IoU must be in (0, 1]")
        percentages = (q.maximum_dark_percent, q.maximum_saturated_percent, t.percentage_threshold,
                       t.ignore_top_percent, t.ignore_bottom_percent, t.ignore_left_percent, t.ignore_right_percent,
                       y.padding_x_percent, y.padding_y_percent, p.alignment_max_translation_percent)
        if any(not 0 <= v <= 100 for v in percentages):
            raise ValueError("Percentage values must be between 0 and 100")
        if t.ignore_top_percent + t.ignore_bottom_percent >= 100 or t.ignore_left_percent + t.ignore_right_percent >= 100:
            raise ValueError("Ignored borders must leave a nonempty ROI")
        if min(t.heatmap_threshold, q.minimum_blur_score, q.blur_weight, q.exposure_penalty) < 0 or t.pixel_threshold < 1 or t.minimum_component_pixels < 1 or t.percentage_threshold <= 0:
            raise ValueError("Tolerance and quality thresholds are invalid")
        if t.decision_mode not in {"PIXEL", "PERCENTAGE", "BOTH", "EITHER"}:
            raise ValueError("Decision mode must be PIXEL, PERCENTAGE, BOTH, or EITHER")
        if any(k not in {0, 3, 5} for k in (t.opening_kernel, t.closing_kernel)):
            raise ValueError("Morphology must be OFF (0), 3, or 5; aggressive kernels are unsupported")
        if not 0 <= q.minimum_mean_intensity < q.maximum_mean_intensity <= 255:
            raise ValueError("Invalid mean-intensity limits")
        if p.heatmap_display_max <= 0 or not 0 < p.alignment_min_correlation <= 1:
            raise ValueError("Invalid heatmap scale/alignment correlation")
        if m.views != 6 or m.angle_step != 60:
            raise ValueError("This machine requires six fixed positions at 60° intervals")
        if m.settle_delay_s < 0 or m.acquisition_time_s <= 0 or m.position_timeout_s <= m.settle_delay_s + m.acquisition_time_s or m.cycle_timeout_s <= 0 or m.signal_timeout_s <= 0:
            raise ValueError("Invalid settle, acquisition, position, cycle, or signal timeout")
        if not 2 <= m.minimum_burst_frames <= c.burst_frames:
            raise ValueError("Minimum burst frames must be 2 through burst_frames")
        if m.interface not in {"manual", "tcp_json"} or not m.host or not 1 <= m.port <= 65535:
            raise ValueError("Machine interface must be manual or tcp_json with a valid host/port")
        if m.interface == "manual" and m.output_enabled:
            raise ValueError("Manual commissioning cannot enable hardware PASS/FAIL output")
        if not self.storage.results_dir or not y.model_path or not p.model_path or not p.normal_image_dir:
            raise ValueError("Model, training, and results paths must not be empty")
        if p.angle_specific:
            for angle in range(60, 361, 60):
                p.path_for_angle(angle)
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "InspectionSettings":
        types = {"camera": CameraSettings, "quality": QualitySettings, "yolo": YoloSettings,
                 "patchcore": PatchCoreSettings, "tolerance": ToleranceSettings,
                 "machine": MachineSettings, "storage": StorageSettings}
        unknown = set(data) - set(types)
        if unknown:
            raise ValueError(f"Unknown settings sections: {sorted(unknown)}")
        sections = {}
        for name, section_type in types.items():
            values = dict(data.get(name, {}))
            if name == "patchcore" and "embedding_layers" in values:
                values["embedding_layers"] = tuple(values["embedding_layers"])
            sections[name] = section_type(**values)
        return cls(**sections).validate()

    @classmethod
    def load(cls, path: str | Path = "config/fixed_inspection.json") -> "InspectionSettings":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path = "config/fixed_inspection.json") -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        temp.replace(path)
