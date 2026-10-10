"""Complete intermediate evidence and append-only part history."""
from __future__ import annotations

import csv
import json
import re
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .anomaly_tolerance import heatmap_image, marked_image
from .fixed_settings import InspectionSettings, ToleranceSettings
from .fixed_views import PartInspectionResult, ViewInspectionResult
from .stationary_roi import QualityEvidence


def write_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


class ResultStorage:
    def __init__(self, settings: InspectionSettings) -> None:
        self.settings = settings
        self.directory: Path | None = None
        self._saved: set[int] = set()
        self._logged_verdict: str | None = None

    def begin(self, part: PartInspectionResult) -> Path:
        root = Path(self.settings.storage.results_dir)
        day = datetime.now().strftime("%Y-%m-%d")
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", part.part_id)[:100].strip(".") or "PART"
        self.directory = root / day / (safe + "_" + uuid.uuid4().hex[:10])
        self.directory.mkdir(parents=True, exist_ok=False)
        self._saved.clear()
        self._logged_verdict = None
        write_json(self.directory / "settings.json", self.settings.to_dict())
        write_json(self.directory / "result.json", part.metadata())
        return self.directory

    @staticmethod
    def _image(path: Path, image: np.ndarray) -> None:
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Failed to save inspection image: {path}")

    def save_view(self, view: ViewInspectionResult) -> None:
        if self.directory is None:
            raise RuntimeError("Storage has no active part")
        if view.angle in self._saved:
            return
        directory = self.directory / f"view_{view.angle}"
        directory.mkdir()
        write_json(directory / "result.json", view.metadata())
        storage = self.settings.storage
        save = storage.save_fail_images if view.verdict == "FAIL" else storage.save_pass_images
        if save:
            self._image(directory / "original.png", view.original_image)
            self._image(directory / "roi.png", view.roi_image)
            self._image(directory / "content_mask.png", view.content_mask.astype(np.uint8) * 255)
            self._image(directory / "anomaly_mask.png", view.decision.candidate_mask.astype(np.uint8) * 255)
            self._image(directory / "filtered_mask.png", view.decision.filtered_mask.astype(np.uint8) * 255)
            self._image(directory / "result.png", marked_image(view.roi_image, view.decision))
            if storage.save_heatmaps:
                # NPY retains float distances exactly; PNG is only the visualization.
                np.save(directory / "heatmap.npy", view.heatmap, allow_pickle=False)
                heatmap = heatmap_image(view.heatmap, self.settings.patchcore.heatmap_display_max)
                self._image(directory / "heatmap.png", heatmap)
                self._image(directory / "overlay.png", cv2.addWeighted(view.roi_image, .6, heatmap, .4, 0))
        self._saved.add(view.angle)

    def update_part(self, part: PartInspectionResult) -> None:
        if self.directory is None:
            raise RuntimeError("Storage has no active part")
        write_json(self.directory / "result.json", part.metadata())
        if part.final_verdict in {"PASS", "FAIL", "FAULT"} and part.final_verdict != self._logged_verdict:
            root = Path(self.settings.storage.results_dir)
            with (root / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({**part.metadata(), "directory": str(self.directory)}, allow_nan=False) + "\n")
            self._logged_verdict = part.final_verdict


def load_saved_view(directory: str | Path) -> ViewInspectionResult:
    from .anomaly_tolerance import process_heatmap
    directory = Path(directory)
    metadata = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    original = cv2.imread(str(directory / "original.png"))
    roi = cv2.imread(str(directory / "roi.png"))
    content = cv2.imread(str(directory / "content_mask.png"), cv2.IMREAD_GRAYSCALE)
    if original is None or roi is None or content is None:
        raise ValueError("Saved view requires original, ROI, and content mask; enable image saving when capturing")
    heatmap = np.load(directory / "heatmap.npy", allow_pickle=False)
    if heatmap.shape != roi.shape[:2] or content.shape != heatmap.shape:
        raise ValueError("Saved heatmap/ROI/content dimensions differ")
    tolerance = ToleranceSettings(**metadata["tolerance"])
    quality = QualityEvidence(**{**metadata["quality"], "reasons": tuple(metadata["quality"]["reasons"])})
    return ViewInspectionResult(metadata["view_number"], metadata["angle"], metadata["timestamp"], original, roi,
                   content.astype(bool), tuple(metadata["roi_bounds"]), quality, metadata["yolo_confidence"],
                   metadata["patchcore_score"], heatmap, process_heatmap(heatmap, tolerance, content.astype(bool)),
                   tolerance, metadata["model_path"], metadata["inference_ms"])


def export_history(root: str | Path, destination: str | Path) -> int:
    root, destination = Path(root), Path(destination)
    fields = ["part_id", "timestamp", "view_number", "angle", "quality_score", "blur_score", "yolo_confidence",
              "patchcore_score", "heatmap_threshold", "filtered_anomaly_pixels", "valid_roi_pixels",
              "anomaly_percentage", "largest_component_pixels", "verdict", "final_verdict", "cycle_time_s", "fault"]
    rows = 0
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        history = root / "history.jsonl"
        if not history.exists():
            return 0
        latest = {}
        for line in history.read_text(encoding="utf-8").splitlines():
            part = json.loads(line)
            latest[part.get("directory", part["part_id"])] = part
        for part in latest.values():
            for view in part["views"] or [{}]:
                writer.writerow({key: view.get(key, part.get(key, part["start_time"] if key == "timestamp" else "")) for key in fields})
                rows += 1
    return rows
