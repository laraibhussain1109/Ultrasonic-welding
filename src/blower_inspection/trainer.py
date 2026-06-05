"""Normal-image training and inspection algorithms.

The implementation is designed for a fixed industrial camera and nest. It learns a
per-model normal appearance template, then reports abnormal pixels and bad fin
sectors at inspection time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import PartModelConfig

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MODEL_VERSION = 1


@dataclass(frozen=True)
class InspectionResult:
    status: str
    anomaly_score: float
    defect_area_px: int
    bad_sector_ratio: float
    bad_sectors: list[int]
    overlay_path: Path | None = None
    report_path: Path | None = None

    @property
    def is_pass(self) -> bool:
        return self.status == "PASS"


def list_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file())


def read_gray(path: str | Path, image_size: tuple[int, int] | None = None) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    if image_size is not None:
        image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def normalize_gray(image: np.ndarray, image_size: tuple[int, int] | None = (1024, 1024)) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image_size is not None:
        image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
    return cv2.GaussianBlur(image, (3, 3), 0)


def align_to_reference(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Align image to reference using ECC translation/Euclidean motion.

    If alignment fails, the original image is returned. This keeps inspection
    usable while flagging positional errors as anomalies.
    """
    if image.shape != reference.shape:
        image = cv2.resize(image, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
        cv2.findTransformECC(reference, image, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5)
        return cv2.warpAffine(
            image,
            warp,
            (reference.shape[1], reference.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )
    except cv2.error:
        return image


def fan_ring_mask(shape: tuple[int, int], inner_ratio: float, outer_ratio: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_radius = min(h, w) / 2.0
    return (radius >= inner_ratio * max_radius) & (radius <= outer_ratio * max_radius)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cleaned.astype(bool)


def sector_statistics(mask: np.ndarray, score_map: np.ndarray, expected_fins: int) -> tuple[float, list[int]]:
    h, w = mask.shape
    yy, xx = np.indices((h, w))
    angle = (np.arctan2(yy - h / 2.0, xx - w / 2.0) + 2 * np.pi) % (2 * np.pi)
    sector_index = np.floor(angle / (2 * np.pi / expected_fins)).astype(np.int32)
    bad_sectors: list[int] = []
    sector_scores: list[float] = []
    for sector in range(expected_fins):
        sector_pixels = mask & (sector_index == sector)
        if not np.any(sector_pixels):
            sector_scores.append(0.0)
            continue
        sector_score = float(np.percentile(score_map[sector_pixels], 95))
        sector_scores.append(sector_score)
    if not sector_scores:
        return 0.0, []
    median = float(np.median(sector_scores))
    mad = float(np.median(np.abs(np.asarray(sector_scores) - median))) + 1e-6
    for sector, score in enumerate(sector_scores):
        if score > median + 6.0 * mad and score > 2.0:
            bad_sectors.append(sector)
    return len(bad_sectors) / max(expected_fins, 1), bad_sectors


class NormalTemplateTrainer:
    """Train and run normal-only blower fan anomaly inspection."""

    def __init__(self, image_size: tuple[int, int] = (1024, 1024)) -> None:
        self.image_size = image_size

    def train(self, config: PartModelConfig) -> Path:
        image_paths = list_images(config.normal_image_dir)
        if len(image_paths) < 3:
            raise ValueError(
                f"Need at least 3 normal images in {config.normal_image_dir}; found {len(image_paths)}"
            )
        images = [read_gray(path, self.image_size) for path in image_paths]
        reference = images[0]
        aligned = [reference]
        for image in images[1:]:
            aligned.append(align_to_reference(image, reference))
        stack = np.stack(aligned, axis=0)
        mean = stack.mean(axis=0).astype(np.float32)
        std = np.maximum(stack.std(axis=0), 0.025).astype(np.float32)
        ring_mask = fan_ring_mask(mean.shape, config.inner_radius_ratio, config.outer_radius_ratio)
        config.model_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            config.model_file,
            version=np.array([MODEL_VERSION], dtype=np.int32),
            mean=mean,
            std=std,
            ring_mask=ring_mask.astype(np.uint8),
            image_size=np.array(self.image_size, dtype=np.int32),
            trained_at=np.array([datetime.now().isoformat()]),
            source_count=np.array([len(image_paths)], dtype=np.int32),
            config_json=np.array([json.dumps(asdict(config), default=str)]),
        )
        return config.model_file

    def inspect(self, config: PartModelConfig, image: np.ndarray, *, save_outputs: bool = True) -> InspectionResult:
        if not config.model_file.exists():
            raise FileNotFoundError(f"Model has not been trained: {config.model_file}")
        loaded = np.load(config.model_file, allow_pickle=False)
        mean = loaded["mean"].astype(np.float32)
        std = loaded["std"].astype(np.float32)
        ring_mask = loaded["ring_mask"].astype(bool)
        normalized = normalize_gray(image, (mean.shape[1], mean.shape[0]))
        aligned = align_to_reference(normalized, mean)
        score_map = np.abs(aligned - mean) / std
        candidate_mask = (score_map > config.anomaly_threshold) & ring_mask
        candidate_mask = clean_mask(candidate_mask)
        defect_area = int(candidate_mask.sum())
        masked_scores = score_map[ring_mask]
        anomaly_score = float(np.percentile(masked_scores, 99.5)) if masked_scores.size else 0.0
        bad_sector_ratio, bad_sectors = sector_statistics(ring_mask, score_map, config.expected_fins)
        status = "FAIL" if (
            defect_area >= config.min_defect_area_px or bad_sector_ratio >= config.max_bad_sector_ratio
        ) else "PASS"
        overlay_path = report_path = None
        if save_outputs:
            overlay_path, report_path = self._save_outputs(config, aligned, score_map, candidate_mask, status, anomaly_score, defect_area, bad_sector_ratio, bad_sectors)
        return InspectionResult(
            status=status,
            anomaly_score=anomaly_score,
            defect_area_px=defect_area,
            bad_sector_ratio=bad_sector_ratio,
            bad_sectors=bad_sectors,
            overlay_path=overlay_path,
            report_path=report_path,
        )

    def _save_outputs(
        self,
        config: PartModelConfig,
        aligned: np.ndarray,
        score_map: np.ndarray,
        defect_mask: np.ndarray,
        status: str,
        anomaly_score: float,
        defect_area: int,
        bad_sector_ratio: float,
        bad_sectors: list[int],
    ) -> tuple[Path, Path]:
        config.result_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base = config.result_dir / f"{stamp}_{status.lower()}"
        display = cv2.cvtColor((aligned * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        heat = cv2.applyColorMap(np.clip(score_map * 24, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        blended = cv2.addWeighted(display, 0.65, heat, 0.35, 0)
        blended[defect_mask] = (0, 0, 255)
        overlay_path = base.with_suffix(".png")
        cv2.imwrite(str(overlay_path), blended)
        report = {
            "model_id": config.id,
            "status": status,
            "anomaly_score": anomaly_score,
            "defect_area_px": defect_area,
            "bad_sector_ratio": bad_sector_ratio,
            "bad_sectors": bad_sectors,
            "created_at": datetime.now().isoformat(),
            "overlay_path": str(overlay_path),
        }
        report_path = base.with_suffix(".json")
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        return overlay_path, report_path
