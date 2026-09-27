"""Native-resolution tiled ViT surface inspection with structural fusion.

The surface path deliberately never resizes a complete elongated blower ROI to a
square.  Native pixels are split into overlapping square tiles; DINOv2 patch
tokens supply spatial descriptors to a normal memory, while a PCA subspace
reconstructor supplies independent "can normal features reconstruct this?"
evidence.  Geometry remains a separate, aspect-preserving lower-resolution path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Iterable

import cv2
import numpy as np

from .camera import crop_component_roi
from .config import PartModelConfig
from .frame_quality import FrameQualityAnalyzer
from .geometry_inspector import (
    FinGeometryInspector, glare_evidence, inspection_band_mask, support_rib_mask,
)
from .trainer import InspectionResult, broken_fin_mask, list_images
from .surface_decision import surface_decision_state
from .vit_tokens import spatial_patch_tokens
from .yolo_tracking import YoloByteTrackDetector

SURFACE_MODEL_VERSION = 1
PREPROCESSING_VERSION = "native-tile-rgb-imagenet-v1"


@dataclass(frozen=True)
class Tile:
    image: np.ndarray
    x: int
    y: int
    valid_width: int
    valid_height: int


@dataclass(frozen=True)
class ComponentScore:
    bounds: tuple[int, int, int, int]
    area: int
    mean: float
    peak: float
    percentile: float
    compactness: float
    score: float


@dataclass(frozen=True)
class SurfaceStatistics:
    global_score: float
    local_score: float
    peak_score: float
    topk_score: float
    components: tuple[ComponentScore, ...] = field(default_factory=tuple)


def tile_positions(length: int, tile_size: int, overlap: float) -> list[int]:
    """Return starts that cover every pixel and anchor the last tile at the edge."""
    if length <= 0 or tile_size <= 0 or not 0 <= overlap < 1:
        raise ValueError("length/tile_size must be positive and overlap must be in [0, 1)")
    if length <= tile_size:
        return [0]
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    starts = list(range(0, length - tile_size + 1, stride))
    end = length - tile_size
    if starts[-1] != end:
        starts.append(end)
    return starts


def generate_tiles(image: np.ndarray, tile_size: int, overlap: float) -> list[Tile]:
    """Generate native-pixel tiles; only a boundary tile is reflect padded."""
    if image.size == 0:
        raise ValueError("Cannot tile an empty ROI")
    height, width = image.shape[:2]
    output: list[Tile] = []
    for y in tile_positions(height, tile_size, overlap):
        for x in tile_positions(width, tile_size, overlap):
            crop = image[y:min(y + tile_size, height), x:min(x + tile_size, width)]
            valid_h, valid_w = crop.shape[:2]
            if valid_h != tile_size or valid_w != tile_size:
                crop = cv2.copyMakeBorder(crop, 0, tile_size - valid_h, 0, tile_size - valid_w,
                                          cv2.BORDER_REFLECT_101)
            output.append(Tile(crop, x, y, valid_w, valid_h))
    return output


def blending_window(height: int, width: int) -> np.ndarray:
    """Positive Hann-like weights prevent both seams and zero-weight borders."""
    wy = np.hanning(max(height, 3))[:height]
    wx = np.hanning(max(width, 3))[:width]
    return np.maximum(np.outer(wy, wx), 0.05).astype(np.float32)


def merge_tile_maps(tile_maps: Iterable[np.ndarray], tiles: list[Tile], shape: tuple[int, int]) -> np.ndarray:
    """Blend tile-local maps back into exact native ROI coordinates."""
    maps = list(tile_maps)
    if len(maps) != len(tiles):
        raise ValueError("Every tile must have exactly one anomaly map")
    total = np.zeros(shape, np.float32)
    weights = np.zeros(shape, np.float32)
    for score_map, tile in zip(maps, tiles):
        local = cv2.resize(np.asarray(score_map, np.float32), (tile.image.shape[1], tile.image.shape[0]),
                           interpolation=cv2.INTER_CUBIC)[:tile.valid_height, :tile.valid_width]
        weight = blending_window(tile.image.shape[0], tile.image.shape[1])[:tile.valid_height, :tile.valid_width]
        ys, xs = slice(tile.y, tile.y + tile.valid_height), slice(tile.x, tile.x + tile.valid_width)
        total[ys, xs] += local * weight
        weights[ys, xs] += weight
    return total / np.maximum(weights, 1e-6)


def prepare_vit_rgb(image: np.ndarray, input_size: int) -> np.ndarray:
    """Convert a native tile to the fixed square tensor geometry expected by ViT.

    Tile placement and anomaly-map reconstruction continue using the untouched
    native tile. This function is shared by training and inference so the model
    can never see 768px during one phase and 518px during the other.
    """
    if input_size <= 0 or image.size == 0:
        raise ValueError("ViT input size and tile content must be non-empty")
    rgb = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3
           else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    if rgb.shape[:2] != (input_size, input_size):
        interpolation = cv2.INTER_AREA if max(rgb.shape[:2]) > input_size else cv2.INTER_CUBIC
        rgb = cv2.resize(rgb, (input_size, input_size), interpolation=interpolation)
    return rgb


def map_roi_box_to_frame(box: tuple[int, int, int, int], roi_bounds: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Translate a native ROI-local box into full camera-frame coordinates."""
    x, y, width, height = box
    roi_x, roi_y, roi_width, roi_height = roi_bounds
    x = min(max(0, x), roi_width)
    y = min(max(0, y), roi_height)
    return roi_x + x, roi_y + y, min(width, roi_width - x), min(height, roi_height - y)


def score_anomaly_map(score_map: np.ndarray, authority: np.ndarray | None = None, *,
                      component_threshold: float, topk_fraction: float = .0005) -> tuple[SurfaceStatistics, np.ndarray]:
    """Score global and minute/local evidence without an area rejection floor."""
    values = np.asarray(score_map, np.float32)
    if authority is None:
        authority = np.ones(values.shape, np.float32)
    if authority.shape != values.shape:
        raise ValueError("authority and score map shapes differ")
    valid = authority > 0
    evidence = values * authority
    selected = evidence[valid]
    if selected.size == 0:
        return SurfaceStatistics(0, 0, 0, 0), np.zeros(values.shape, bool)
    count = max(1, int(np.ceil(selected.size * topk_fraction)))
    top = np.partition(selected, selected.size - count)[-count:]
    mask = (evidence >= component_threshold) & valid
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    components: list[ComponentScore] = []
    for label in range(1, labels_count):
        x, y, width, height, area = map(int, stats[label])
        component_values = evidence[labels == label]
        perimeter_mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(perimeter_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
        compactness = float(4 * np.pi * area / max(perimeter * perimeter, 1.0))
        mean, peak = float(component_values.mean()), float(component_values.max())
        high = float(np.quantile(component_values, .95))
        # Intensity remains authoritative for a one-patch pit; area adds support
        # but can never veto an extreme local response.
        component_score = max(peak, high, mean * (1.0 + min(np.log1p(area) / 10.0, .5)))
        components.append(ComponentScore((x, y, width, height), area, mean, peak, high,
                                         compactness, float(component_score)))
    components.sort(key=lambda item: item.score, reverse=True)
    return SurfaceStatistics(float(np.quantile(selected, .995)), float(np.quantile(selected, .9995)),
                             float(selected.max()), float(top.mean()), tuple(components)), mask


def robust_limit(values: list[float], multiplier: float) -> float:
    array = np.asarray(values, np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return max(float(np.quantile(array, .995)), median + multiplier * 1.4826 * max(mad, 1e-6))


def largest_component_area(mask: np.ndarray) -> int:
    """Return the largest localized region without averaging it over the ROI."""
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return max((int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)), default=0)


class TiledViTSurfaceInspector:
    """DINOv2 memory + normal-subspace reconstruction + geometry backend."""

    def __init__(self, device: str | None = None) -> None:
        self.device_name = device
        self._model: Any = None
        self._torch: Any = None
        self._checkpoint_cache: tuple[float, dict[str, Any]] | None = None
        self._search_cache: tuple[int, Any, Any, Any, Any] | None = None

    @staticmethod
    def _path(config: PartModelConfig) -> Path:
        return config.surface_model_file or config.model_file

    def _device(self, config: PartModelConfig | None = None):
        torch = self._require("torch", "Install the industrial dependencies with: pip install -e '.[industrial]'")
        if self._torch is None:
            self._torch = torch
        if self.device_name:
            device = torch.device(self.device_name)
        elif torch.cuda.is_available():
            # CUDA is always selected ahead of CPU. CUDA index zero is the
            # production default; an explicit constructor device can override it.
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        if config is not None and config.surface_require_gpu and device.type != "cuda":
            cuda_build = getattr(getattr(torch, "version", None), "cuda", None) or "CPU-only"
            raise RuntimeError(
                "CUDA GPU REQUIRED FOR SURFACE INSPECTION — PyTorch cannot access CUDA. "
                f"Installed PyTorch build: {cuda_build}. Install a CUDA-enabled PyTorch build "
                "supported by the installed NVIDIA driver, then verify torch.cuda.is_available()."
            )
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return device

    @staticmethod
    def _require(name: str, hint: str):
        try:
            return __import__(name, fromlist=["*"])
        except ImportError as exc:
            raise RuntimeError(f"Required surface-inspection dependency '{name}' is missing. {hint}") from exc

    def _build_model(self, config: PartModelConfig):
        if self._model is not None:
            return self._model
        timm = self._require("timm", "Install the industrial dependencies so DINOv2 can be downloaded and cached.")
        weights = config.vit_weights_path or Path("data/models/backbones") / f"{config.vit_backbone}.pth"
        if weights.is_file():
            model = timm.create_model(config.vit_backbone, pretrained=False, num_classes=0,
                                      img_size=config.vit_input_size)
            state = self._torch.load(weights, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            state = {key.removeprefix("module."): value for key, value in state.items()}
            missing, unexpected = model.load_state_dict(state, strict=False)
            if len(missing) > 8 or unexpected:
                raise RuntimeError(f"DINOv2 weight/backbone mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}")
        elif config.vit_auto_download:
            try:
                # timm uses the model's official pretrained configuration and
                # Hugging Face cache. Persist our own atomic copy so subsequent
                # starts work offline and never redownload the checkpoint.
                model = timm.create_model(config.vit_backbone, pretrained=True, num_classes=0,
                                          img_size=config.vit_input_size)
                weights.parent.mkdir(parents=True, exist_ok=True)
                temporary = weights.with_suffix(weights.suffix + ".part")
                self._torch.save(model.state_dict(), temporary)
                temporary.replace(weights)
            except Exception as exc:
                raise RuntimeError(
                    f"Automatic DINOv2 download failed for {config.vit_backbone!r}. "
                    "Check internet/proxy access to Hugging Face, or copy a compatible checkpoint to "
                    f"{weights}. Original error: {exc}"
                ) from exc
        else:
            raise RuntimeError(
                f"DINOv2 weights are missing at {weights} and vit_auto_download is disabled."
            )
        model.eval().to(self._device(config))
        self._model = model
        return model

    def _tokens(self, images: list[np.ndarray], config: PartModelConfig) -> tuple[np.ndarray, tuple[int, int]]:
        device = self._device(config)
        torch, model = self._torch, self._build_model(config)
        arrays = []
        for image in images:
            # The native 768px tile remains unchanged for ROI/map coordinates,
            # but the pretrained fixed-size ViT-S/14 patch embed expects 518px.
            # Explicitly resizing here keeps training and inference identical
            # and prevents timm's "Input height ... doesn't match model (518)".
            rgb = prepare_vit_rgb(image, config.vit_input_size)
            array = rgb.astype(np.float32) / 255.0
            array = (array - np.asarray(config.surface_normalization_mean, np.float32)) / np.asarray(config.surface_normalization_std, np.float32)
            arrays.append(array.transpose(2, 0, 1))
        tensor = torch.from_numpy(np.stack(arrays)).pin_memory() if device.type == "cuda" else torch.from_numpy(np.stack(arrays))
        model_grid = getattr(model.patch_embed, "grid_size", None)
        if model_grid is None:
            patch = getattr(model.patch_embed, "patch_size", (14, 14))
            patch_h, patch_w = ((int(patch[0]), int(patch[1]))
                                if isinstance(patch, (tuple, list)) else (int(patch), int(patch)))
            grid = (config.vit_input_size // patch_h, config.vit_input_size // patch_w)
        else:
            grid = (tuple(int(value) for value in model_grid)
                    if isinstance(model_grid, (tuple, list))
                    else (int(model_grid), int(model_grid)))
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            output = model.forward_features(tensor.to(device, non_blocking=True))
            tokens = spatial_patch_tokens(output, model, grid)
        return tokens.float().cpu().numpy(), grid

    @staticmethod
    def _fit_reconstructor(features: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
        mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
        centered = features - mean
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        return mean, vt[:min(rank, vt.shape[0])].astype(np.float32)

    def _branch_maps(self, tokens: np.ndarray, grid: tuple[int, int], checkpoint: dict[str, Any],
                     config: PartModelConfig) -> tuple[list[np.ndarray], list[np.ndarray]]:
        torch, device = self._torch, self._device(config)
        flat = torch.as_tensor(tokens.reshape(-1, tokens.shape[-1]), dtype=torch.float32,
                               device=device)
        cache_key = id(checkpoint)
        if self._search_cache is None or self._search_cache[0] != cache_key:
            memory = torch.as_tensor(checkpoint["memory_bank"], dtype=torch.float32, device=device)
            memory_norm = (memory * memory).sum(1).unsqueeze(0)
            mean = torch.as_tensor(checkpoint["reconstruction_mean"], dtype=torch.float32, device=device)
            basis = torch.as_tensor(checkpoint["reconstruction_basis"], dtype=torch.float32, device=device)
            self._search_cache = cache_key, memory, memory_norm, mean, basis
        _, memory, memory_norm, mean, basis = self._search_cache
        # Bounded GPU chunks avoid a patches x memory allocation exceeding 12 GB.
        nearest = []
        for start in range(0, len(flat), 2048):
            query = flat[start:start + 2048]
            distance = ((query * query).sum(1, keepdim=True) + memory_norm
                        - 2 * query @ memory.T).clamp_min_(0)
            nearest.append(distance.min(1).values.sqrt_())
        memory_score = torch.cat(nearest).reshape(tokens.shape[0], *grid)
        centered = flat - mean
        reconstruction = centered @ basis.T @ basis + mean
        residual = torch.mean((flat - reconstruction) ** 2, dim=1).sqrt_().reshape(tokens.shape[0], *grid)
        return list(memory_score.cpu().numpy()), list(residual.cpu().numpy())

    @staticmethod
    def _metadata(config: PartModelConfig) -> dict[str, Any]:
        return {"version": SURFACE_MODEL_VERSION, "preprocessing_version": PREPROCESSING_VERSION,
                "backbone": config.vit_backbone, "tile_size": config.surface_tile_size,
                "vit_input_size": config.vit_input_size,
                "tile_overlap": config.surface_tile_overlap,
                "surface_near_candidate_ratio": config.surface_near_candidate_ratio,
                "geometry_surface_support_threshold": config.geometry_surface_support_threshold,
                "fine_break_candidate_area_px": config.fine_break_candidate_area_px,
                "fine_break_strong_area_px": config.fine_break_strong_area_px,
                "normalization_mean": list(config.surface_normalization_mean),
                "normalization_std": list(config.surface_normalization_std),
                "camera_resolution": [config.camera_width, config.camera_height]}

    def _validate(self, config: PartModelConfig, checkpoint: dict[str, Any]) -> None:
        expected = self._metadata(config)
        actual = checkpoint.get("metadata", {})
        mismatches = [key for key, value in expected.items() if actual.get(key) != value]
        if mismatches:
            raise RuntimeError("MODEL CONFIGURATION MISMATCH — RETRAIN REQUIRED: " + ", ".join(mismatches))

    def runtime_device_name(self) -> str:
        device = self._device()
        if device.type == "cuda":
            return f"{device} ({self._torch.cuda.get_device_name(device.index or 0)})"
        return str(device)

    def runtime_summary(self) -> str:
        return "DINOv2 tiled memory + PCA feature reconstruction + calibrated fin geometry"

    def validate_ready(self, config: PartModelConfig) -> None:
        checkpoint = self._load(config)
        self._validate(config, checkpoint)
        # Readiness includes the actual feature extractor. This verifies CUDA
        # and performs the one-time automatic pretrained-weight download.
        self._build_model(config)

    def train(self, config: PartModelConfig, progress_callback=None) -> Path:
        paths = list_images(config.normal_image_dir)
        if len(paths) < 6:
            raise ValueError("At least six diverse normal rotational views are required")
        detector = YoloByteTrackDetector(config.yolo_model_path, config.yolo_confidence) if config.yolo_model_path else None
        quality = FrameQualityAnalyzer(config.training_min_sharpness, config.max_glare_ratio, config.max_saturation_ratio)
        accepted: list[tuple[Path, np.ndarray, dict[str, Any]]] = []
        rejected: list[dict[str, Any]] = []
        for index, path in enumerate(paths, 1):
            frame = cv2.imread(str(path))
            if frame is None:
                rejected.append({"path": str(path), "reasons": ["UNREADABLE"]})
                continue
            roi = detector.exact_crop(frame) if detector else crop_component_roi(frame, roi_ratios=config.roi_ratios)
            result = quality.analyze(roi)
            row = {"path": str(path), "sharpness": result.sharpness, "saturation": result.saturation_ratio,
                   "glare": result.glare_ratio, "dark": result.dark_ratio, "reasons": list(result.reasons)}
            (accepted if result.valid else rejected).append((path, roi, row) if result.valid else row)
            if progress_callback:
                progress_callback(type("Progress", (), {"format": lambda self, i=index: f"Quality {i}/{len(paths)}"})())
        if len(accepted) < 6:
            report_path = self._path(config).with_suffix(".training_report.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps({"accepted": [item[2] for item in accepted],
                                               "rejected": rejected,
                                               "error": "INSUFFICIENT_DIVERSE_NORMAL_VIEWS"}, indent=2) + "\n")
            raise ValueError(f"Only {len(accepted)} quality/diverse images remain; see the training report")
        split = max(1, int(round(len(accepted) * .2)))
        calibration, training = accepted[::max(1, len(accepted) // split)][:split], []
        calibration_paths = {item[0] for item in calibration}
        training = [item for item in accepted if item[0] not in calibration_paths]
        all_tokens: list[np.ndarray] = []
        for _path, roi, _row in training:
            tiles = generate_tiles(roi, config.surface_tile_size, config.surface_tile_overlap)
            for start in range(0, len(tiles), config.surface_tile_batch_size):
                tokens, _grid = self._tokens([tile.image for tile in tiles[start:start + config.surface_tile_batch_size]], config)
                all_tokens.append(tokens.reshape(-1, tokens.shape[-1]))
        features = np.concatenate(all_tokens)
        rng = np.random.default_rng(42)
        if len(features) > config.surface_memory_bank_size:
            features = features[rng.choice(len(features), config.surface_memory_bank_size, replace=False)]
        mean, basis = self._fit_reconstructor(features, config.reconstruction_rank)
        provisional = {"memory_bank": features, "reconstruction_mean": mean, "reconstruction_basis": basis}
        samples: dict[str, list[float]] = {key: [] for key in (
            "global", "local", "peak", "topk", "memory", "reconstruction", "fine_break"
        )}
        geometry_images = [item[1] for item in accepted]
        geometry = FinGeometryInspector.calibrate(geometry_images, band_top=config.inspection_band_top_ratio,
                                                  band_bottom=config.inspection_band_bottom_ratio)
        for _path, roi, _row in calibration:
            memory_map, reconstruction_map, _ = self._surface_maps(roi, config, provisional)
            for name, branch in (("memory", memory_map), ("reconstruction", reconstruction_map)):
                samples[name].append(float(np.quantile(branch, .9995)))
            combined = self._normalize_branches(memory_map, reconstruction_map, None)
            stats, _ = score_anomaly_map(combined, self._authority(roi, config), component_threshold=float(np.quantile(combined, .999)))
            for name in ("global", "local", "peak", "topk"):
                samples[name].append(getattr(stats, f"{name}_score"))
            fine_break = broken_fin_mask(roi, roi.shape[:2])
            fine_break &= inspection_band_mask(roi.shape[:2], config.surface_band_top_ratio,
                                                config.surface_band_bottom_ratio)
            samples["fine_break"].append(float(largest_component_area(fine_break)))
        thresholds = {name: {"candidate": robust_limit(values, 6), "strong": robust_limit(values, 10)}
                      for name, values in samples.items()}
        thresholds["fine_break"]["candidate"] = max(
            thresholds["fine_break"]["candidate"], float(config.fine_break_candidate_area_px)
        )
        thresholds["fine_break"]["strong"] = max(
            thresholds["fine_break"]["strong"],
            thresholds["fine_break"]["candidate"] * 1.5,
            float(config.fine_break_strong_area_px),
        )
        checkpoint = {"memory_bank": self._torch.as_tensor(features),
                      "reconstruction_mean": self._torch.as_tensor(mean),
                      "reconstruction_basis": self._torch.as_tensor(basis),
                      "metadata": self._metadata(config), "thresholds": thresholds,
                      "geometry": geometry, "trained_at": datetime.now().isoformat()}
        path = self._path(config)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._torch.save(checkpoint, path)
        report = {"accepted": [item[2] for item in accepted], "rejected": rejected,
                  "training_count": len(training), "calibration_count": len(calibration),
                  "tile_size": config.surface_tile_size, "tile_overlap": config.surface_tile_overlap,
                  "thresholds": thresholds, "rotation_coverage_views": len(accepted)}
        path.with_suffix(".training_report.json").write_text(json.dumps(report, indent=2) + "\n")
        return path

    def _load(self, config: PartModelConfig) -> dict[str, Any]:
        path = self._path(config)
        if not path.is_file():
            raise FileNotFoundError(f"Surface model has not been trained: {path}")
        stamp = path.stat().st_mtime
        if self._checkpoint_cache and self._checkpoint_cache[0] == stamp:
            return self._checkpoint_cache[1]
        self._device(config)
        checkpoint = self._torch.load(path, map_location="cpu", weights_only=True)
        self._checkpoint_cache = stamp, checkpoint
        return checkpoint

    def _surface_maps(self, roi: np.ndarray, config: PartModelConfig, checkpoint: dict[str, Any]):
        tiles = generate_tiles(roi, config.surface_tile_size, config.surface_tile_overlap)
        memory_maps, reconstruction_maps = [], []
        for start in range(0, len(tiles), config.surface_tile_batch_size):
            batch = tiles[start:start + config.surface_tile_batch_size]
            tokens, grid = self._tokens([tile.image for tile in batch], config)
            memory, reconstruction = self._branch_maps(tokens, grid, checkpoint, config)
            memory_maps.extend(memory)
            reconstruction_maps.extend(reconstruction)
        shape = roi.shape[:2]
        return merge_tile_maps(memory_maps, tiles, shape), merge_tile_maps(reconstruction_maps, tiles, shape), tiles

    @staticmethod
    def _normalize_branches(memory: np.ndarray, reconstruction: np.ndarray, thresholds: dict | None) -> np.ndarray:
        if thresholds:
            m = memory / max(thresholds["memory"]["candidate"], 1e-6)
            r = reconstruction / max(thresholds["reconstruction"]["candidate"], 1e-6)
        else:
            m = memory / max(float(np.quantile(memory, .999)), 1e-6)
            r = reconstruction / max(float(np.quantile(reconstruction, .999)), 1e-6)
        # Max retains independent evidence; agreement gets an explicit boost.
        return (np.maximum(m, r) + .35 * np.minimum(m, r)).astype(np.float32)

    @staticmethod
    def _authority(roi: np.ndarray, config: PartModelConfig) -> np.ndarray:
        authority = np.ones(roi.shape[:2], np.float32)
        top, bottom = int(roi.shape[0] * config.surface_band_top_ratio), int(roi.shape[0] * config.surface_band_bottom_ratio)
        authority[:top] = 0
        authority[bottom:] = 0
        ribs, _ = support_rib_mask(roi)
        authority[ribs] *= config.surface_rib_authority
        return authority

    def inspect(self, config: PartModelConfig, image: np.ndarray, *, save_outputs: bool = True,
                crop_to_component: bool = True) -> InspectionResult:
        started = time.perf_counter()
        roi = crop_component_roi(image, roi_ratios=config.roi_ratios) if crop_to_component else image.copy()
        checkpoint = self._load(config)
        self._validate(config, checkpoint)
        quality = FrameQualityAnalyzer(config.inference_min_sharpness, 1.0, config.max_saturation_ratio).analyze(roi)
        if not quality.valid:
            return InspectionResult("VIEW INVALID", 0, 0, 0, [], display_image=roi,
                                    reason_codes=quality.reasons, view_valid=False,
                                    view_quality_score=quality.sharpness)
        tile_started = time.perf_counter()
        memory_map, reconstruction_map, tiles = self._surface_maps(roi, config, checkpoint)
        surface_ms = (time.perf_counter() - tile_started) * 1000
        thresholds = checkpoint["thresholds"]
        combined = self._normalize_branches(memory_map, reconstruction_map, thresholds)
        authority = self._authority(roi, config)
        sensitivity = max(.5, min(1.5, config.surface_sensitivity))
        candidate_line = 1.0 / sensitivity
        strong_line = max(thresholds["peak"]["strong"] / max(thresholds["peak"]["candidate"], 1e-6), 1.25) / sensitivity
        stats, mask = score_anomaly_map(combined, authority, component_threshold=candidate_line)
        memory_peak = float(np.quantile(memory_map, .9995))
        reconstruction_peak = float(np.quantile(reconstruction_map, .9995))
        memory_candidate = memory_peak >= thresholds["memory"]["candidate"] / sensitivity
        reconstruction_candidate = reconstruction_peak >= thresholds["reconstruction"]["candidate"] / sensitivity
        local_evidence = max(stats.local_score, stats.peak_score, stats.topk_score,
                             stats.components[0].score if stats.components else 0)
        geometry_started = time.perf_counter()
        scale = min(1.0, config.geometry_max_width / max(roi.shape[1], 1))
        geometry_image = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else roi
        geometry = FinGeometryInspector(checkpoint["geometry"], band_top=config.inspection_band_top_ratio,
                                        band_bottom=config.inspection_band_bottom_ratio,
                                        candidate_threshold=config.geometry_candidate_threshold).inspect(geometry_image)
        geometry_mask = cv2.resize(geometry.defect_mask.astype(np.uint8), (roi.shape[1], roi.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
        fine_break_mask = broken_fin_mask(roi, roi.shape[:2])
        fine_break_mask &= inspection_band_mask(roi.shape[:2], config.surface_band_top_ratio,
                                                config.surface_band_bottom_ratio)
        fine_break_area = largest_component_area(fine_break_mask)
        fine_thresholds = thresholds["fine_break"]
        fine_break_candidate = fine_break_area >= fine_thresholds["candidate"] / sensitivity
        fine_break_strong = fine_break_area >= fine_thresholds["strong"] / sensitivity
        surface_candidate, strong, geometry_corroborated = surface_decision_state(
            local_score=local_evidence,
            candidate_line=candidate_line,
            strong_line=strong_line,
            memory_candidate=memory_candidate,
            reconstruction_candidate=reconstruction_candidate,
            geometry_score=geometry.score,
            geometry_support_threshold=config.geometry_surface_support_threshold,
            near_candidate_ratio=config.surface_near_candidate_ratio,
            fine_break_candidate=fine_break_candidate,
            fine_break_strong=fine_break_strong,
        )
        geometry_ms = (time.perf_counter() - geometry_started) * 1000
        glare = glare_evidence(roi, roi.shape[:2], top=config.surface_band_top_ratio,
                               bottom=config.surface_band_bottom_ratio)
        reasons: list[str] = []
        if memory_candidate: reasons.append("SURFACE_MEMORY_ANOMALY")
        if reconstruction_candidate: reasons.append("SURFACE_RECONSTRUCTION_ANOMALY")
        if strong: reasons.append("STRONG_LOCAL_SURFACE_ANOMALY")
        if geometry_corroborated: reasons.append("SURFACE_GEOMETRY_CORROBORATION")
        if fine_break_candidate: reasons.append("BROKEN_FIN_CANDIDATE")
        if fine_break_strong: reasons.append("BROKEN_FIN")
        if geometry.broken_fin_score >= config.geometry_fail_threshold: reasons.append("BROKEN_FIN")
        if geometry.missing_fin_score >= config.geometry_fail_threshold: reasons.append("MISSING_FIN")
        if geometry.tilted_fin_score >= config.geometry_fail_threshold: reasons.append("TILTED_FIN")
        geometry_fail = geometry.score >= config.geometry_fail_threshold
        if geometry_fail: reasons.append("GEOMETRY_DEFORMATION")
        # Glare is evidence for temporal confirmation, never a subtraction or PASS override.
        if surface_candidate and glare.score >= config.glare_threshold: reasons.append("LIKELY_GLARE")
        immediate = geometry_fail or strong
        status = "FAIL" if immediate else ("CANDIDATE" if surface_candidate else "PASS")
        near_mask = ((combined * authority) >= candidate_line * config.surface_near_candidate_ratio)
        surface_mask = mask | (near_mask if geometry_corroborated else False)
        if fine_break_candidate:
            surface_mask |= fine_break_mask
        confirmed = (surface_mask if status != "PASS" else np.zeros_like(mask)) | geometry_mask
        display = roi.copy() if roi.ndim == 3 else cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(confirmed.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            boxes.append((x, y, width, height))
            cv2.drawContours(display, [contour], -1, (0, 0, 255), max(2, roi.shape[1] // 1200))
        section_count = config.patchcore_section_count
        sections = tuple(sorted({min(section_count - 1, int((x + width / 2) * section_count / roi.shape[1]))
                                 for x, _y, width, _height in boxes}))
        geometry_components = {"pitch": geometry.pitch_score, "continuity": geometry.continuity_score,
                               "broken": geometry.broken_fin_score,
                               "memory": memory_peak, "reconstruction": reconstruction_peak,
                               "global": stats.global_score, "local": stats.local_score,
                               "peak": stats.peak_score, "topk": stats.topk_score,
                               "fine_break_area": float(fine_break_area)}
        latencies = {"surface_tiles": surface_ms, "geometry": geometry_ms,
                     "total": (time.perf_counter() - started) * 1000, "tile_count": float(len(tiles))}
        if config.engineering_compare_legacy:
            comparison_started = time.perf_counter()
            try:
                from .patchcore_inspector import PatchCoreInspector
                legacy = PatchCoreInspector(device=self.device_name).inspect(
                    config, roi, save_outputs=False, crop_to_component=False
                )
                latencies["legacy_patchcore"] = (time.perf_counter() - comparison_started) * 1000
                geometry_components["legacy_patchcore"] = legacy.anomaly_score
                reasons.append(f"COMPARE_LEGACY_{legacy.status}")
            except Exception as exc:
                # Comparison is deliberately non-authoritative; report an
                # unavailable baseline without hiding the production verdict.
                latencies["legacy_patchcore"] = (time.perf_counter() - comparison_started) * 1000
                reasons.append("COMPARE_LEGACY_UNAVAILABLE")
        reasons = list(dict.fromkeys(reasons))
        report_path = overlay_path = None
        if save_outputs:
            config.result_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            overlay_path = config.result_dir / f"{stamp}_{status.lower()}_surface.png"
            report_path = config.result_dir / f"{stamp}_{status.lower()}_surface.json"
            cv2.imwrite(str(overlay_path), display)
            report_path.write_text(json.dumps({"status": status, "reasons": reasons,
                "statistics": asdict(stats), "memory_score": memory_peak,
                "reconstruction_score": reconstruction_peak, "glare_score": glare.score,
                "geometry_score": geometry.score, "latencies_ms": latencies,
                "roi_dimensions": list(roi.shape[:2][::-1]), "tile_count": len(tiles)}, indent=2) + "\n")
        return InspectionResult(status, local_evidence, int(confirmed.sum()), 0, [],
                                overlay_path, report_path, display, boxes,
                                geometry_score=geometry.score, periodicity_score=geometry.periodicity_score,
                                glare_score=glare.score, reason_codes=tuple(reasons), view_valid=True,
                                view_quality_score=quality.sharpness, latencies_ms=latencies,
                                geometry_components=geometry_components,
                                candidate_sections=sections)
