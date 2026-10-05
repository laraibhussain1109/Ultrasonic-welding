from pathlib import Path

from blower_inspection.training_dataset import split_physical_groups


def touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")


def test_physical_parts_never_cross_training_calibration_validation(tmp_path):
    for part in range(10):
        for view in range(3):
            touch(tmp_path / "parts" / f"blower_{part:02d}" / f"view_{view:03d}.png")

    split = split_physical_groups(tmp_path, seed=42)

    path_sets = [set(split.training), set(split.calibration), set(split.validation)]
    assert not path_sets[0] & path_sets[1]
    assert not path_sets[0] & path_sets[2]
    assert not path_sets[1] & path_sets[2]
    for group, paths in split.groups.items():
        assert set(paths) <= set(getattr(split, split.assignments[group]))


def test_flat_legacy_dataset_is_disjoint_but_warns_not_physically_independent(tmp_path):
    for view in range(20):
        touch(tmp_path / f"image{view:03d}.png")

    split = split_physical_groups(tmp_path)

    assert split.training and split.calibration and split.validation
    assert set(split.training).isdisjoint(split.calibration)
    assert any("not physically independent" in warning for warning in split.warnings)
