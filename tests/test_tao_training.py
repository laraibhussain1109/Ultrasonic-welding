from pathlib import Path
from unittest.mock import patch

import pytest

from blower_inspection.tao_training import VISUAL_CHANGENET_MODULE, run_visual_changenet_task


def test_visual_changenet_training_runs_official_module_in_docker(tmp_path: Path):
    spec = tmp_path / "specs" / "train.yaml"
    spec.parent.mkdir()
    spec.write_text("train: {}\n", encoding="utf-8")

    with patch("blower_inspection.tao_training.subprocess.run") as run:
        run_visual_changenet_task("train", spec, project_dir=tmp_path)

    command = run.call_args.args[0]
    assert command[:3] == ["docker", "run", "--rm"]
    assert VISUAL_CHANGENET_MODULE in command
    assert command[-3:] == ["train", "-e", "/workspace/project/specs/train.yaml"]
    assert run.call_args.kwargs == {"check": True}


def test_visual_changenet_spec_must_exist(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="experiment spec not found"):
        run_visual_changenet_task("train", tmp_path / "missing.yaml", project_dir=tmp_path)


def test_visual_changenet_spec_must_be_in_project(tmp_path: Path):
    outside = tmp_path.parent / "outside_visual_changenet.yaml"
    outside.write_text("export: {}\n", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="inside the mounted project"):
            run_visual_changenet_task("export", outside, project_dir=tmp_path)
    finally:
        outside.unlink()
