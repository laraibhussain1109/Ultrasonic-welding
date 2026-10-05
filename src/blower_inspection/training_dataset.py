"""Physical-part-aware normal dataset inventory and deterministic splitting."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetSplit:
    training: tuple[Path, ...]
    calibration: tuple[Path, ...]
    validation: tuple[Path, ...]
    groups: dict[str, tuple[Path, ...]]
    assignments: dict[str, str]
    warnings: tuple[str, ...]

    def report(self) -> dict:
        return {
            "physical_groups": {key: [str(path) for path in paths] for key, paths in self.groups.items()},
            "group_assignments": self.assignments,
            "training_images": [str(path) for path in self.training],
            "calibration_images": [str(path) for path in self.calibration],
            "validation_images": [str(path) for path in self.validation],
            "warnings": list(self.warnings),
            "physical_independence": not self.warnings,
        }


def _flat_session(name: str) -> str | None:
    stem = Path(name).stem
    match = re.match(r"(.+?)(?:[_-](?:frame|view|img)?\d+)$", stem, re.IGNORECASE)
    return match.group(1) if match else None


def inventory_groups(root: str | Path) -> tuple[dict[str, tuple[Path, ...]], list[str]]:
    """Group images by parts/session folders or conservative filename sessions."""
    root = Path(root)
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    grouped: dict[str, list[Path]] = {}
    flat: list[Path] = []
    for path in paths:
        relative = path.relative_to(root)
        if len(relative.parts) > 1:
            group = "/".join(relative.parts[:-1])
            grouped.setdefault(group, []).append(path)
        else:
            inferred = _flat_session(path.name)
            if inferred:
                grouped.setdefault(f"filename:{inferred}", []).append(path)
            else:
                flat.append(path)
    warnings: list[str] = []
    if flat:
        # Unknown flat files are kept together; never pretend adjacent frames
        # are independent physical parts.
        grouped["legacy_flat_unknown_part"] = flat
        warnings.append(
            "Flat normal images have no physical_part_id/session metadata. "
            "Production qualification requires data/training/<model>/parts/<part_id>/..."
        )
    return {key: tuple(value) for key, value in grouped.items()}, warnings


def split_physical_groups(root: str | Path, *, seed: int = 42) -> DatasetSplit:
    groups, warnings = inventory_groups(root)
    if not groups:
        raise ValueError(f"No training images found under {root}")
    ordered = sorted(groups, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    if len(ordered) < 3:
        warnings.append("Fewer than three physical groups: held-out independence cannot be guaranteed")
    if len(ordered) == 1:
        source = groups[ordered[0]]
        if len(source) < 6:
            raise ValueError("At least six normal views are required for legacy flat splitting")
        first, second = max(1, int(len(source) * .70)), max(2, int(len(source) * .85))
        fallback_groups = {"legacy_fit_views": source[:first],
                           "legacy_calibration_views": source[first:second],
                           "legacy_validation_views": source[second:]}
        warnings.append("Legacy flat split is view-disjoint but not physically independent")
        return DatasetSplit(fallback_groups["legacy_fit_views"], fallback_groups["legacy_calibration_views"],
                            fallback_groups["legacy_validation_views"], fallback_groups,
                            {"legacy_fit_views": "training", "legacy_calibration_views": "calibration",
                             "legacy_validation_views": "validation"}, tuple(dict.fromkeys(warnings)))
    n = len(ordered)
    train_n = max(1, int(n * .70))
    calibration_n = max(1, int(n * .15)) if n >= 2 else 0
    if train_n + calibration_n >= n and n >= 3:
        train_n = n - 2
        calibration_n = 1
    train_groups = ordered[:train_n]
    calibration_groups = ordered[train_n:train_n + calibration_n]
    validation_groups = ordered[train_n + calibration_n:]
    # Backward-compatible fallback keeps image sets disjoint but explicitly
    # reports that they may belong to one physical blower.
    if not validation_groups:
        source = groups[ordered[-1]]
        cut = max(1, len(source) // 5)
        groups = dict(groups)
        groups[ordered[-1]] = source[:-cut] or source
        groups["legacy_heldout_views"] = source[-cut:]
        validation_groups = ["legacy_heldout_views"]
        warnings.append("Validation uses held-out views, not an independent physical part")
    assignments = {key: "training" for key in train_groups}
    assignments.update({key: "calibration" for key in calibration_groups})
    assignments.update({key: "validation" for key in validation_groups})
    flatten = lambda keys: tuple(path for key in keys for path in groups[key])
    return DatasetSplit(flatten(train_groups), flatten(calibration_groups), flatten(validation_groups),
                        groups, assignments, tuple(dict.fromkeys(warnings)))
