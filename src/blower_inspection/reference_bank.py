"""Reflection-resistant phase reference bank for rotating blower wheels."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


REFERENCE_BANK_VERSION = 1


def structural_descriptor(image: np.ndarray, top: float = 0.18, bottom: float = 0.82) -> np.ndarray:
    """Describe edges and their projections, largely independent of brightness."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h = gray.shape[0]
    y0, y1 = int(h * top), int(h * bottom)
    gray = gray[max(0, y0):max(y0 + 1, y1)]
    gray = cv2.GaussianBlur(cv2.resize(gray, (96, 48)), (3, 3), 0).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    magnitude /= float(np.percentile(magnitude, 95)) + 1e-6
    low = cv2.resize(np.clip(magnitude, 0, 2), (24, 12)).ravel()
    descriptor = np.concatenate((low, np.mean(np.abs(gx), axis=0), np.mean(np.abs(gy), axis=1)))
    descriptor -= descriptor.mean()
    descriptor /= np.linalg.norm(descriptor) + 1e-8
    return descriptor.astype(np.float32)


@dataclass(frozen=True)
class ReferenceMatch:
    index: int
    distance: float
    image: np.ndarray


class ReferenceBank:
    """In-memory bank; disk is touched only by :meth:`load`."""

    def __init__(self, directory: Path, images: list[np.ndarray], descriptors: np.ndarray, sources: list[str]) -> None:
        self.directory, self.images = directory, images
        self.descriptors, self.sources = descriptors.astype(np.float32), sources

    @property
    def manifest_path(self) -> Path:
        return self.directory / "manifest.json"

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()

    def candidates(self, image: np.ndarray, count: int = 3, exclude_source: str | Path | None = None) -> list[ReferenceMatch]:
        query = structural_descriptor(image)
        distances = np.linalg.norm(self.descriptors - query[None, :], axis=1)
        excluded = str(Path(exclude_source).resolve()) if exclude_source is not None else None
        order = np.argsort(distances)
        result = []
        for index in order:
            if excluded and str(Path(self.sources[int(index)]).resolve()) == excluded:
                continue
            result.append(ReferenceMatch(int(index), float(distances[index]), self.images[int(index)]))
            if len(result) >= count:
                break
        return result

    @classmethod
    def build(cls, images: list[np.ndarray], sources: list[str | Path], directory: str | Path, size: int) -> "ReferenceBank":
        if len(images) < 2 or len(images) != len(sources):
            raise ValueError("Reference bank requires at least two images with matching sources")
        descriptors = np.stack([structural_descriptor(image) for image in images])
        # Farthest-first traversal avoids storing adjacent, near-identical phases.
        chosen = [int(np.argmin(np.linalg.norm(descriptors - np.median(descriptors, axis=0), axis=1)))]
        while len(chosen) < min(size, len(images)):
            nearest = np.min(np.linalg.norm(descriptors[:, None] - descriptors[chosen][None], axis=2), axis=1)
            nearest[chosen] = -1
            chosen.append(int(np.argmax(nearest)))
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        selected_images, selected_sources = [], []
        for output_index, source_index in enumerate(chosen):
            filename = f"ref_{output_index:03d}.png"
            if not cv2.imwrite(str(directory / filename), images[source_index]):
                raise RuntimeError(f"Unable to write reference {filename}")
            selected_images.append(images[source_index].copy())
            selected_sources.append(str(Path(sources[source_index]).resolve()))
        selected_descriptors = descriptors[chosen]
        np.savez_compressed(directory / "descriptors.npz", descriptors=selected_descriptors)
        manifest = {"version": REFERENCE_BANK_VERSION, "count": len(chosen), "references": [
            {"file": f"ref_{i:03d}.png", "source": selected_sources[i]} for i in range(len(chosen))
        ]}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return cls(directory, selected_images, selected_descriptors, selected_sources)

    @classmethod
    def load(cls, directory: str | Path) -> "ReferenceBank":
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("version") != REFERENCE_BANK_VERSION:
            raise RuntimeError("Unsupported reference bank version")
        items = manifest["references"]
        images = [cv2.imread(str(directory / item["file"])) for item in items]
        if any(image is None for image in images):
            raise RuntimeError("Reference bank contains an unreadable image")
        descriptors = np.load(directory / "descriptors.npz")["descriptors"]
        if len(images) != len(descriptors) or len(images) != manifest.get("count"):
            raise RuntimeError("Reference bank manifest/descriptors are inconsistent")
        return cls(directory, images, descriptors, [item["source"] for item in items])
