"""Labeled GOOD/NG evaluation using cached raw PatchCore maps."""
from __future__ import annotations

from dataclasses import dataclass

from .fixed_settings import ToleranceSettings
from .fixed_views import ViewInspectionResult


@dataclass(frozen=True)
class CalibrationSample:
    name: str
    actual_class: str
    view: ViewInspectionResult


def evaluate_samples(samples: list[CalibrationSample], tolerance: ToleranceSettings) -> tuple[list[dict], dict]:
    rows = []
    counts = {"GOOD_correctly_passed": 0, "GOOD_incorrectly_failed": 0,
              "NG_correctly_failed": 0, "NG_incorrectly_passed": 0}
    for sample in samples:
        if sample.actual_class not in {"GOOD", "NG"}:
            raise ValueError("Calibration labels must be GOOD or NG")
        tuned = sample.view.retune(tolerance)
        verdict = tuned.verdict
        key = ("GOOD_correctly_passed" if verdict == "PASS" else "GOOD_incorrectly_failed") if sample.actual_class == "GOOD" else ("NG_correctly_failed" if verdict == "FAIL" else "NG_incorrectly_passed")
        counts[key] += 1
        rows.append({"image": sample.name, "actual_class": sample.actual_class, **tuned.decision.statistics()})
    good = counts["GOOD_correctly_passed"] + counts["GOOD_incorrectly_failed"]
    ng = counts["NG_correctly_failed"] + counts["NG_incorrectly_passed"]
    return rows, {**counts, "GOOD_count": good, "NG_count": ng,
                  "false_positive_rate": counts["GOOD_incorrectly_failed"] / good if good else None,
                  "false_negative_rate": counts["NG_incorrectly_passed"] / ng if ng else None}
