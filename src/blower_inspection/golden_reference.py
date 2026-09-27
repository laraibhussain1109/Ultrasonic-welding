"""Versioned phase-matched robust golden reference database."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .reference_bank import structural_descriptor
from .registration import RegistrationResult, register_to_reference
from .trainer import reflection_invariant_gray

GOLDEN_REFERENCE_VERSION = 2


@dataclass(frozen=True)
class GoldenEvidence:
    phase_bin: int
    reference_index: int
    registration: RegistrationResult
    intensity_map: np.ndarray
    edge_map: np.ndarray
    intensity_score: float
    edge_score: float


class GoldenReferenceBank:
    def __init__(self, directory: Path, manifest: dict, descriptors: np.ndarray) -> None:
        self.directory, self.manifest = directory, manifest
        self.descriptors = descriptors.astype(np.float32)

    @property
    def digest(self) -> str:
        return hashlib.sha256((self.directory / "manifest.json").read_bytes()).hexdigest()

    @staticmethod
    def _canonical(image: np.ndarray, width: int, height: int) -> np.ndarray:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    @classmethod
    def build(cls, records: list[tuple[np.ndarray, str, str]], directory: str | Path,
              *, phase_bins: int = 12, max_width: int = 1536, noise_floor: float = .08) -> "GoldenReferenceBank":
        if len(records) < 3:
            raise ValueError("Golden references require at least three known-good views")
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        descriptors = np.stack([structural_descriptor(image, .05, .95) for image, _source, _group in records])
        bin_count = min(max(2, phase_bins), len(records))
        centres = [int(np.argmin(np.linalg.norm(descriptors - np.median(descriptors, axis=0), axis=1)))]
        while len(centres) < bin_count:
            nearest = np.min(np.linalg.norm(descriptors[:, None] - descriptors[centres][None], axis=2), axis=1)
            centres.append(int(np.argmax(nearest)))
        centroid_values = descriptors[centres]
        assignments = np.argmin(np.linalg.norm(descriptors[:, None] - centroid_values[None], axis=2), axis=1)
        median_aspect = float(np.median([image.shape[1] / image.shape[0] for image, _, _ in records]))
        width = min(max_width, int(np.median([image.shape[1] for image, _, _ in records])))
        height = max(64, int(round(width / median_aspect)))
        phases, saved_centroids = [], []
        for phase in range(bin_count):
            indices = np.flatnonzero(assignments == phase)
            if not len(indices):
                continue
            images = [cls._canonical(records[index][0], width, height) for index in indices]
            normalized = np.stack([reflection_invariant_gray(image, (height, width)) for image in images])
            median = np.median(normalized, axis=0).astype(np.float32)
            scale = np.maximum(1.4826 * np.median(np.abs(normalized - median), axis=0), noise_floor).astype(np.float32)
            edges = np.stack([cv2.magnitude(cv2.Sobel(item, cv2.CV_32F, 1, 0),
                                             cv2.Sobel(item, cv2.CV_32F, 0, 1)) for item in normalized])
            edge_median = np.median(edges, axis=0).astype(np.float32)
            edge_scale = np.maximum(1.4826 * np.median(np.abs(edges - edge_median), axis=0), noise_floor).astype(np.float32)
            representative = np.median(np.stack(images).astype(np.float32), axis=0).astype(np.uint8)
            np.savez_compressed(directory / f"phase_{phase:03d}.npz", median=median, scale=scale,
                                edge_median=edge_median, edge_scale=edge_scale)
            cv2.imwrite(str(directory / f"phase_{phase:03d}.png"), representative)
            groups = sorted({records[index][2] for index in indices})
            phases.append({"phase_bin": phase, "count": len(indices), "groups": groups,
                           "sources": [records[index][1] for index in indices]})
            saved_centroids.append(np.mean(descriptors[indices], axis=0))
        centroids = np.stack(saved_centroids).astype(np.float32)
        np.savez_compressed(directory / "descriptors.npz", descriptors=centroids)
        manifest = {"version": GOLDEN_REFERENCE_VERSION, "phase_bins": len(phases),
                    "canonical_size": [width, height], "noise_floor": noise_floor, "phases": phases}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return cls(directory, manifest, centroids)

    @classmethod
    def load(cls, directory: str | Path) -> "GoldenReferenceBank":
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("version") != GOLDEN_REFERENCE_VERSION:
            raise RuntimeError("MODEL CONFIGURATION MISMATCH — RETRAIN REQUIRED: golden reference version")
        return cls(directory, manifest, np.load(directory / "descriptors.npz")["descriptors"])

    def candidates(self, image: np.ndarray, count: int) -> list[int]:
        query = structural_descriptor(image, .05, .95)
        return [int(value) for value in np.argsort(np.linalg.norm(self.descriptors - query, axis=1))[:count]]

    def compare(self, image: np.ndarray, *, candidates: int, min_correlation: float,
                max_translation_ratio: float, max_rotation_deg: float) -> GoldenEvidence:
        width, height = self.manifest["canonical_size"]
        current = self._canonical(image, width, height)
        registrations = []
        for phase in self.candidates(image, candidates):
            reference = cv2.imread(str(self.directory / f"phase_{phase:03d}.png"))
            registrations.append(register_to_reference(current, reference, min_correlation=min_correlation,
                max_translation_ratio=max_translation_ratio, max_rotation_deg=max_rotation_deg,
                reference_index=phase, max_working_dimension=768))
        valid = [item for item in registrations if item.success]
        best = max(valid or registrations, key=lambda item: item.correlation)
        phase = int(best.reference_index or 0)
        data = np.load(self.directory / f"phase_{phase:03d}.npz")
        if not best.success:
            zeros = np.zeros((height, width), np.float32)
            return GoldenEvidence(phase, phase, best, zeros, zeros, 0., 0.)
        normalized = reflection_invariant_gray(best.aligned_image, (height, width))
        intensity = np.abs(normalized - data["median"]) / data["scale"]
        edge = cv2.magnitude(cv2.Sobel(normalized, cv2.CV_32F, 1, 0), cv2.Sobel(normalized, cv2.CV_32F, 0, 1))
        edge_residual = np.abs(edge - data["edge_median"]) / data["edge_scale"]
        return GoldenEvidence(phase, phase, best, intensity.astype(np.float32), edge_residual.astype(np.float32),
                              float(np.quantile(intensity, .9995)), float(np.quantile(edge_residual, .9995)))
