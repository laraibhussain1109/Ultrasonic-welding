"""Prepare exact YOLO-cropped known-good datasets for anomaly training."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .trainer import IMAGE_EXTENSIONS
from .yolo_tracking import YoloByteTrackDetector


class CropDetector(Protocol):
    def exact_crop(self, frame: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class DatasetPreparationResult:
    discovered: int
    written: int
    skipped: int
    failed: int
    manifest: Path


def prepare_yolo_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    yolo_model: str | Path | None = None,
    *,
    confidence: float = 0.40,
    recursive: bool = True,
    replace: bool = False,
    detector: CropDetector | None = None,
) -> DatasetPreparationResult:
    """Crop all source images with YOLO and write an auditable training dataset.

    Relative source folders are preserved, so duplicate camera filenames do not
    overwrite one another. Failed/no-detection images are recorded in a CSV and
    never silently copied into the normal dataset.
    """
    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Dataset source directory does not exist: {source}")
    in_place = source == output
    if in_place and not replace:
        raise ValueError("In-place dataset cropping requires --replace; normal training autocrops without modifying images")
    if not in_place and (source in output.parents or output in source.parents):
        raise ValueError("Source and output directories must not contain one another")
    if detector is None:
        if yolo_model is None:
            raise ValueError("A YOLO model path is required")
        detector = YoloByteTrackDetector(yolo_model, confidence)

    iterator = source.rglob("*") if recursive else source.glob("*")
    images = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str, str]] = []
    written = skipped = failed = 0
    for image_path in images:
        relative = image_path.relative_to(source)
        destination = output / relative
        if destination.exists() and not replace:
            skipped += 1
            rows.append((str(relative), "SKIPPED_EXISTS", str(destination)))
            continue
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            failed += 1
            rows.append((str(relative), "FAILED_READ", ""))
            continue
        try:
            crop = detector.exact_crop(frame)
        except Exception as exc:
            failed += 1
            rows.append((str(relative), f"FAILED_DETECTION: {exc}", ""))
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_path = destination.with_name(f".{destination.stem}.crop{destination.suffix}") if in_place else destination
        if not cv2.imwrite(str(write_path), crop):
            failed += 1
            rows.append((str(relative), "FAILED_WRITE", str(destination)))
            continue
        if in_place:
            write_path.replace(destination)
        written += 1
        rows.append((str(relative), "WRITTEN", str(destination)))

    manifest = output / "crop_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("source_relative_path", "status", "output_path"))
        writer.writerows(rows)
    return DatasetPreparationResult(len(images), written, skipped, failed, manifest)
