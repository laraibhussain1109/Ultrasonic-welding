"""Frozen-model production qualification; never changes model thresholds."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path

import cv2

from .camera import crop_component_roi
from .config import PartModelConfig
from .trainer import IMAGE_EXTENSIONS
from .yolo_tracking import YoloByteTrackDetector


def _images(root: Path):
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def run_qualification(inspector, config: PartModelConfig, good_dir: str | Path,
                      ng_dir: str | Path, output: str | Path | None = None) -> Path:
    """Measure false rejects and misses without fitting or calibrating anything."""
    good_dir, ng_dir = Path(good_dir), Path(ng_dir)
    records = []
    counts = Counter({"true_good": 0, "false_reject": 0, "detected_ng": 0,
                      "missed_ng": 0, "good_invalid": 0, "ng_invalid": 0,
                      "unreadable": 0})
    by_type = defaultdict(Counter)
    detector = (YoloByteTrackDetector(config.yolo_model_path, config.yolo_confidence)
                if config.yolo_model_path else None)
    items = [(path, "good", "good") for path in _images(good_dir)]
    for path in _images(ng_dir):
        category = path.relative_to(ng_dir).parts[0] if len(path.relative_to(ng_dir).parts) > 1 else "ng"
        expected = "good" if category in {"scratch_ok", "glare_ok"} else "ng"
        items.append((path, expected, category))
    if not items:
        raise ValueError("Qualification folders contain no images")
    for path, expected, category in items:
        image = cv2.imread(str(path))
        if image is None:
            records.append({"path": str(path), "expected": expected, "category": category,
                            "status": "VIEW INVALID", "reasons": ["UNREADABLE"]})
            counts["unreadable"] += 1
            counts[f"{expected}_invalid"] += 1
            by_type[category]["total"] += 1
            by_type[category]["invalid"] += 1
            continue
        # Qualification commonly receives either original 3840x2160 frames or
        # already prepared elongated blower ROIs. Cropping an ROI a second time
        # changes its geometry and guarantees registration failure. Mirror the
        # training path for full frames and preserve native ROI inputs exactly.
        elongated_roi = image.shape[1] / max(image.shape[0], 1) >= 3.0
        if elongated_roi:
            roi, input_mode = image, "native_roi"
        elif detector is not None:
            try:
                roi, input_mode = detector.exact_crop(image), "yolo_full_frame"
            except ValueError:
                roi, input_mode = crop_component_roi(image, roi_ratios=config.roi_ratios), "configured_full_frame"
        else:
            roi, input_mode = crop_component_roi(image, roi_ratios=config.roi_ratios), "configured_full_frame"
        result = inspector.inspect(config, roi, save_outputs=False, crop_to_component=False)
        predicted = "good" if result.status == "PASS" else ("invalid" if result.status == "VIEW INVALID" else "ng")
        if predicted == "invalid": counts[f"{expected}_invalid"] += 1
        elif expected == "good" and predicted == "ng": counts["false_reject"] += 1
        elif expected == "ng" and predicted == "good": counts["missed_ng"] += 1
        elif expected == predicted == "good": counts["true_good"] += 1
        elif expected == predicted == "ng": counts["detected_ng"] += 1
        by_type[category]["total"] += 1
        if predicted == "invalid": by_type[category]["invalid"] += 1
        elif expected == "ng": by_type[category]["detected" if predicted == "ng" else "missed"] += 1
        else: by_type[category]["passed" if predicted == "good" else "false_reject"] += 1
        records.append({"path": str(path), "expected": expected, "category": category,
                        "status": result.status, "score": result.anomaly_score,
                        "input_mode": input_mode, "roi_dimensions": list(roi.shape[:2][::-1]),
                        "registration_score": result.registration_score,
                        "reasons": list(result.reason_codes)})
    good_total = sum(item[1] == "good" for item in items)
    ng_total = sum(item[1] == "ng" for item in items)
    valid_total = good_total + ng_total - counts["good_invalid"] - counts["ng_invalid"]
    coverage = valid_total / max(good_total + ng_total, 1)
    report = {"created_at": datetime.now().isoformat(), "model_id": config.id,
              "qualification_status": ("COMPLETE" if coverage >= .90
                                         else "INSUFFICIENT_VALID_COVERAGE"),
              "frozen_model": str(config.surface_model_file or config.model_file),
              "thresholds_optimized_on_qualification": False,
              "confusion": dict(counts),
              "good_false_reject_rate": counts["false_reject"] / max(good_total - counts["good_invalid"], 1),
              "good_invalid_rate": counts["good_invalid"] / max(good_total, 1),
              "ng_miss_rate": counts["missed_ng"] / max(ng_total - counts["ng_invalid"], 1),
              "ng_invalid_rate": counts["ng_invalid"] / max(ng_total, 1),
              "valid_coverage_rate": coverage,
              "by_category": {key: dict(value) for key, value in by_type.items()},
              "worst_false_positives": sorted((row for row in records if row["expected"] == "good"
                                                and row["status"] in {"CANDIDATE", "FAIL"}),
                                               key=lambda row: row.get("score", 0), reverse=True)[:20],
              "missed_ng": [row for row in records if row["expected"] == "ng" and row["status"] == "PASS"],
              "invalid_views": [row for row in records if row["status"] == "VIEW INVALID"][:100],
              "images": records}
    output = Path(output or config.result_dir / "qualification_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output
