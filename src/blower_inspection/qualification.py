"""Frozen-model production qualification; never changes model thresholds."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path

import cv2

from .config import PartModelConfig
from .trainer import IMAGE_EXTENSIONS


def _images(root: Path):
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def run_qualification(inspector, config: PartModelConfig, good_dir: str | Path,
                      ng_dir: str | Path, output: str | Path | None = None) -> Path:
    """Measure false rejects and misses without fitting or calibrating anything."""
    good_dir, ng_dir = Path(good_dir), Path(ng_dir)
    records, counts, by_type = [], Counter(), defaultdict(Counter)
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
            counts["invalid"] += 1; continue
        result = inspector.inspect(config, image, save_outputs=False, crop_to_component=True)
        predicted = "good" if result.status == "PASS" else ("invalid" if result.status == "VIEW INVALID" else "ng")
        if expected == "good" and predicted == "ng": counts["false_reject"] += 1
        elif expected == "ng" and predicted == "good": counts["missed_ng"] += 1
        elif expected == predicted == "good": counts["true_good"] += 1
        elif expected == predicted == "ng": counts["detected_ng"] += 1
        by_type[category]["total"] += 1
        by_type[category]["detected" if predicted == "ng" else "missed"] += expected == "ng"
        records.append({"path": str(path), "expected": expected, "category": category,
                        "status": result.status, "score": result.anomaly_score,
                        "reasons": list(result.reason_codes)})
    good_total = sum(item[1] == "good" for item in items)
    ng_total = sum(item[1] == "ng" for item in items)
    report = {"created_at": datetime.now().isoformat(), "model_id": config.id,
              "frozen_model": str(config.surface_model_file or config.model_file),
              "thresholds_optimized_on_qualification": False,
              "confusion": dict(counts),
              "good_false_reject_rate": counts["false_reject"] / max(good_total, 1),
              "ng_miss_rate": counts["missed_ng"] / max(ng_total, 1),
              "by_category": {key: dict(value) for key, value in by_type.items()},
              "worst_false_positives": sorted((row for row in records if row["expected"] == "good" and row["status"] != "PASS"),
                                               key=lambda row: row.get("score", 0), reverse=True)[:20],
              "missed_ng": [row for row in records if row["expected"] == "ng" and row["status"] == "PASS"],
              "images": records}
    output = Path(output or config.result_dir / "qualification_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output
