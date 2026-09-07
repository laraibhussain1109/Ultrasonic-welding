from pathlib import Path
from unittest.mock import patch

import pytest

from blower_inspection.tao_training import (
    VISUAL_CHANGENET_MODULE,
    _clean_spec_output,
    copy_default_visual_changenet_spec,
    download_visual_changenet_pretrained,
    find_visual_changenet_pretrained,
    run_visual_changenet_task,
    validate_visual_changenet_dataset,
    validate_visual_changenet_spec,
)


def test_visual_changenet_training_runs_official_module_in_docker(tmp_path: Path):
    spec = tmp_path / "specs" / "train.yaml"
    spec.parent.mkdir()
    spec.write_text(
        "task: segment\ntrain:\n  pretrained_model_path: null\n"
        "results_dir: /results\ndataset:\n  segment:\n    root_dir: /data/example\n",
        encoding="utf-8",
    )
    dataset = _dataset(tmp_path)

    with patch("blower_inspection.tao_training.subprocess.run") as run:
        run_visual_changenet_task("train", spec, project_dir=tmp_path, dataset_dir=dataset, from_scratch=True)

    command = run.call_args.args[0]
    assert command[:3] == ["docker", "run", "--rm"]
    assert VISUAL_CHANGENET_MODULE in command
    assert command[-3:] == ["train", "-e", "/workspace/project/specs/.train.runtime.yaml"]
    assert f"{dataset.resolve()}:/data/TAO_VCN_DATASET:ro" in command
    assert run.call_args.kwargs == {"check": True}


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "TAO_VCN_DATASET"
    for folder in ("A", "B", "label", "list"):
        (root / folder).mkdir(parents=True)
    for folder in ("A", "B", "label"):
        (root / folder / "part.png").write_bytes(b"png")
    for split in ("train.txt", "val.txt", "test.txt", "predict.txt"):
        (root / "list" / split).write_text("part.png\n", encoding="utf-8")
    return root


def test_existing_visual_changenet_dataset_is_validated_without_changes(tmp_path: Path):
    root = _dataset(tmp_path)
    before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
    assert validate_visual_changenet_dataset(root) == root.resolve()
    assert sorted(str(path.relative_to(root)) for path in root.rglob("*")) == before


def test_pretrained_checkpoint_is_found_below_ngc_download_folder(tmp_path: Path):
    checkpoint = tmp_path / "visual_changenet_levircd_vtrainable_v1.0" / "changenet_segment_levir_cd.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"weights")
    assert find_visual_changenet_pretrained([tmp_path]) == checkpoint.resolve()


def test_ngc_download_returns_discovered_checkpoint(tmp_path: Path):
    def download(command, check):
        checkpoint = tmp_path / "download" / "version" / "changenet_segment_levir_cd.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"weights")

    with patch("blower_inspection.tao_training.subprocess.run", side_effect=download) as run:
        checkpoint = download_visual_changenet_pretrained(tmp_path / "download")

    assert checkpoint.name == "changenet_segment_levir_cd.pth"
    assert run.call_args.args[0][:4] == ["ngc", "registry", "model", "download-version"]


def test_visual_changenet_spec_must_exist(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="experiment spec not found"):
        run_visual_changenet_task("train", tmp_path / "missing.yaml", project_dir=tmp_path)


def test_visual_changenet_spec_must_be_in_project(tmp_path: Path):
    outside = tmp_path.parent / "outside_visual_changenet.yaml"
    outside.write_text("task: segment\nresults_dir: /results\nexport: {}\n", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="inside the mounted project"):
            run_visual_changenet_task("export", outside, project_dir=tmp_path)
    finally:
        outside.unlink()


def test_copy_default_segmentation_spec_from_installed_container(tmp_path: Path):
    completed = type("Completed", (), {"stdout": "encryption_key: key\ntask: segment\n"})()
    with patch("blower_inspection.tao_training.subprocess.run", return_value=completed) as run:
        output = copy_default_visual_changenet_spec(tmp_path / "spec.yaml")

    assert output.read_text(encoding="utf-8") == "encryption_key: key\ntask: segment\n"
    command = run.call_args.args[0]
    assert command[:3] == ["docker", "run", "--rm"]
    assert command[3:5] == ["--entrypoint", "cat"]
    assert command[-1] == "/usr/local/lib/python3.12/dist-packages/nvidia_tao_pytorch/cv/visual_changenet/experiment_specs/experiment_spec.yaml"
    assert run.call_args.kwargs == {"check": True, "capture_output": True, "text": True}


def test_tao_banner_is_removed_from_copied_yaml():
    output = _clean_spec_output(
        "=== TAO Toolkit PyTorch ===\nNVIDIA Release 7.1.0\n\n"
        "encryption_key: key\ntask: segment\n"
    )
    assert output == "encryption_key: key\ntask: segment\n"


def test_banner_corrupted_existing_spec_is_rejected(tmp_path: Path):
    spec = tmp_path / "bad.yaml"
    spec.write_text(
        "=== TAO Toolkit PyTorch ===\nNVIDIA Release 7.1.0\n"
        "encryption_key: key\ntask: segment\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="banner before the YAML"):
        validate_visual_changenet_spec(spec)
