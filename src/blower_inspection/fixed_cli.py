"""Commands for training and reviewing the fixed-position workflow."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import cv2

from .fixed_settings import InspectionSettings
from .fixed_views import FixedViewPipeline, PartInspectionResult
from .inspection_storage import ResultStorage, export_history, load_saved_view
from .patchcore_spatial import SpatialPatchCore
from .tolerance_calibration import CalibrationSample, evaluate_samples

COMMANDS = {"train-fixed", "inspect-fixed", "calibrate-fixed", "export-fixed-history"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="blower-inspection")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train-fixed", help="Train original spatial PatchCore on GOOD full-camera images")
    train.add_argument("--angle", type=int, choices=range(60, 361, 60))
    inspect = sub.add_parser("inspect-fixed", help="Inspect a saved image and retain intermediate evidence")
    inspect.add_argument("image")
    inspect.add_argument("--angle", type=int, choices=range(60, 361, 60), default=60)
    calibrate = sub.add_parser("calibrate-fixed", help="Evaluate labeled GOOD/NG images or stored heatmaps")
    calibrate.add_argument("manifest", help="JSON array: image/saved_view, actual_class, angle")
    calibrate.add_argument("--output", default="data/results/fixed_calibration")
    export = sub.add_parser("export-fixed-history", help="Export per-part/per-view records as CSV")
    export.add_argument("destination")
    for command in (train, inspect, calibrate, export):
        command.add_argument("--settings", default="config/fixed_inspection.json")
    args = parser.parse_args(argv)
    settings = InspectionSettings.load(args.settings)
    if args.command == "train-fixed":
        path = SpatialPatchCore(settings).train_fixed(args.angle, lambda progress: print(progress.format(), flush=True))
        print(f"PatchCore memory bank and frozen backbone: {path}")
        return 0
    if args.command == "export-fixed-history":
        print(f"Exported {export_history(settings.storage.results_dir, args.destination)} rows")
        return 0
    if args.command == "inspect-fixed":
        pipeline = FixedViewPipeline(settings)
        view = pipeline.inspect_frames([cv2.imread(args.image)], args.angle)
        part = PartInspectionResult("OFFLINE_SINGLE_VIEW", view.timestamp, time.monotonic(), {view.angle: view})
        storage = ResultStorage(replace(settings, storage=replace(settings.storage, save_pass_images=True,
                                                                  save_fail_images=True, save_heatmaps=True)))
        directory = storage.begin(part)
        storage.save_view(view)
        storage.update_part(part)
        print(json.dumps({**view.metadata(), "evidence_directory": str(directory),
                          "part_verdict": "INCOMPLETE (single-view diagnostic)"}, indent=2))
        return 1 if view.verdict == "FAIL" else 0
    entries = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pipeline = FixedViewPipeline(settings)
    storage_settings = replace(settings, storage=replace(settings.storage, results_dir=str(output),
                                                         save_pass_images=True, save_fail_images=True, save_heatmaps=True))
    samples, errors = [], []
    base = Path(args.manifest).resolve().parent
    for index, entry in enumerate(entries, 1):
        label = entry["actual_class"]
        if label not in {"GOOD", "NG"}:
            raise ValueError("Calibration classes must be GOOD or NG")
        key = "saved_view" if "saved_view" in entry else "image"
        path = Path(entry[key])
        if not path.is_absolute():
            path = base / path
        try:
            view = load_saved_view(path) if key == "saved_view" else pipeline.inspect_frames([cv2.imread(str(path))], int(entry.get("angle", 60)))
            samples.append(CalibrationSample(str(path), label, view))
            part = PartInspectionResult(f"CALIBRATION_{label}_{index:06d}", view.timestamp, time.monotonic(), {view.angle: view})
            storage = ResultStorage(storage_settings)
            storage.begin(part)
            storage.save_view(view)
            storage.update_part(part)
        except Exception as exc:
            errors.append({"image": str(path), "actual_class": label, "status": "INVALID / UNRUN", "error": str(exc)})
    rows, metrics = evaluate_samples(samples, settings.tolerance)
    report = {"samples": rows, "metrics": metrics, "invalid_or_unrun": errors,
              "settings": settings.to_dict(), "complete_labeled_validation": bool(samples) and not errors and metrics["GOOD_count"] > 0 and metrics["NG_count"] > 0}
    (output / "calibration.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["complete_labeled_validation"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
