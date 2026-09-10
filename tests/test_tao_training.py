from pathlib import Path
from unittest.mock import patch

import pytest

from blower_inspection.tao_training import (
    PRETRAINED_NGC_MODEL,
    VISUAL_CHANGENET_MODULE,
    _clean_spec_output,
    copy_default_visual_changenet_spec,
    download_visual_changenet_pretrained,
    find_visual_changenet_pretrained,
    find_training_checkpoint,
    ensure_visual_changenet_pretrained,
    run_visual_changenet_task,
    resolve_visual_changenet_pretrained,
    validate_visual_changenet_dataset,
    validate_visual_changenet_spec,
)


def test_visual_changenet_training_runs_official_module_in_docker(tmp_path: Path):
    spec = tmp_path / "specs" / "train.yaml"
    spec.parent.mkdir()
    spec.write_text(
        "task: segment\ntrain:\n  pretrained_model_path: null\n"
        "  num_epochs: 1\n"
        "results_dir: /results\ndataset:\n  segment:\n    root_dir: /data/example\n",
        encoding="utf-8",
    )
    dataset = _dataset(tmp_path)

    captured = {}

    def execute(command, check):
        runtime_relative = command[-1].removeprefix("/workspace/project/")
        captured["spec"] = (tmp_path / runtime_relative).read_text(encoding="utf-8")

    with patch("blower_inspection.tao_training.subprocess.run", side_effect=execute) as run:
        run_visual_changenet_task("train", spec, project_dir=tmp_path, dataset_dir=dataset, from_scratch=True)

    command = run.call_args.args[0]
    assert command[:3] == ["docker", "run", "--rm"]
    assert VISUAL_CHANGENET_MODULE in command
    assert command[-3:] == ["train", "-e", "/workspace/project/specs/.train.runtime.yaml"]
    assert f"{dataset.resolve()}:/data/TAO_VCN_DATASET:ro" in command
    assert "num_epochs: 50" in captured["spec"]
    assert "num_epochs: 1" in spec.read_text(encoding="utf-8")
    assert run.call_args.kwargs == {"check": True}


def test_visual_changenet_rejects_one_epoch_before_docker(tmp_path: Path):
    spec = tmp_path / "spec.yaml"
    spec.write_text("task: segment\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least 2 epochs"):
        run_visual_changenet_task("train", spec, project_dir=tmp_path, epochs=1)


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


def test_pretrained_download_directory_can_be_passed_directly(tmp_path: Path):
    checkpoint = tmp_path / "ngc-version" / "changenet_segment_levir_cd.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"weights")
    assert resolve_visual_changenet_pretrained(tmp_path) == checkpoint.resolve()


def test_identical_duplicate_checkpoints_choose_shortest_path(tmp_path: Path):
    direct = tmp_path / "changenet_segment_levir_cd.pth"
    nested = tmp_path / "download" / "version" / direct.name
    nested.parent.mkdir(parents=True)
    direct.write_bytes(b"same weights")
    nested.write_bytes(b"same weights")
    assert find_visual_changenet_pretrained([tmp_path]) == direct.resolve()


def test_ngc_download_returns_discovered_checkpoint(tmp_path: Path):
    def download(command, check):
        checkpoint = tmp_path / "download" / "version" / "changenet_segment_levir_cd.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"weights")

    with patch("blower_inspection.tao_training.subprocess.run", side_effect=download) as run:
        checkpoint = download_visual_changenet_pretrained(tmp_path / "download")

    assert checkpoint.name == "changenet_segment_levir_cd.pth"
    assert run.call_args.args[0][:4] == ["ngc", "registry", "model", "download-version"]
    assert PRETRAINED_NGC_MODEL in run.call_args.args[0]


def test_ngc_download_reuses_checkpoint_in_canonical_directory(tmp_path: Path):
    checkpoint = tmp_path / "version" / "changenet_segment_levir_cd.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"weights")
    with patch("blower_inspection.tao_training.subprocess.run") as run:
        result = download_visual_changenet_pretrained(tmp_path)
    assert result == tmp_path / "changenet_segment_levir_cd.pth"
    assert result.read_bytes() == b"weights"
    run.assert_not_called()


def test_tao_init_installs_manual_root_download_canonically(tmp_path: Path):
    manual = tmp_path / "ngc-download" / "changenet_segment_levir_cd.pth"
    manual.parent.mkdir()
    manual.write_bytes(b"weights")
    canonical = tmp_path / "data" / "models" / "pretrained"
    with patch("blower_inspection.tao_training.subprocess.run") as run:
        installed = ensure_visual_changenet_pretrained(canonical, search_roots=[tmp_path])
    assert installed == canonical / "changenet_segment_levir_cd.pth"
    assert installed.read_bytes() == b"weights"
    run.assert_not_called()


def test_ngc_access_denial_has_actionable_message(tmp_path: Path):
    from subprocess import CalledProcessError

    with patch(
        "blower_inspection.tao_training.subprocess.run",
        side_effect=CalledProcessError(1, ["ngc"]),
    ):
        with pytest.raises(RuntimeError, match="not proof that the API key is wrong"):
            download_visual_changenet_pretrained(tmp_path)


def test_visual_changenet_spec_must_exist(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="experiment spec not found"):
        run_visual_changenet_task("train", tmp_path / "missing.yaml", project_dir=tmp_path)


def test_export_finds_latest_checkpoint_and_writes_configured_onnx(tmp_path: Path):
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        "task: segment\nresults_dir: /results\n"
        "export:\n  results_dir: ${results_dir}/export\n"
        "  checkpoint: ${results_dir}/train/changenet.pth\n"
        "  onnx_file: ${export.results_dir}/changenet.onnx\n"
        "  batch_size: 2\n",
        encoding="utf-8",
    )
    results = tmp_path / "results"
    checkpoint = results / "train" / "epoch=49.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")
    output = tmp_path / "data" / "models" / "BF-001" / "visual_changenet.onnx"
    captured = {}

    def export(command, check):
        runtime = tmp_path / command[-1].removeprefix("/workspace/project/")
        captured["spec"] = runtime.read_text(encoding="utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"onnx")

    with patch("blower_inspection.tao_training.subprocess.run", side_effect=export):
        run_visual_changenet_task(
            "export", spec, project_dir=tmp_path, results_dir=results, export_file=output
        )

    assert find_training_checkpoint(results) == checkpoint
    assert "/workspace/project/results/train/epoch=49.pth" in captured["spec"]
    assert "/workspace/project/data/models/BF-001/visual_changenet.onnx" in captured["spec"]
    assert "  batch_size: 1" in captured["spec"]


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
    assert command[3:5] == ["--entrypoint", "sh"]
    assert command[-2] == "-c"
    assert "find /usr/local/lib /opt" in command[-1]
    assert "*/nvidia_tao_pytorch/cv/visual_changenet/experiment_specs/experiment_spec.yaml" in command[-1]
    assert "cat \"$spec\"" in command[-1]
    assert run.call_args.kwargs == {"check": True, "capture_output": True, "text": True}


def test_copy_default_spec_reports_container_lookup_failure(tmp_path: Path):
    from subprocess import CalledProcessError

    error = CalledProcessError(2, ["docker"], stderr="VisualChangeNet default spec was not found")
    with patch("blower_inspection.tao_training.subprocess.run", side_effect=error):
        with pytest.raises(RuntimeError, match="default spec was not found"):
            copy_default_visual_changenet_spec(tmp_path / "spec.yaml")


def test_copy_default_spec_rejects_empty_container_output(tmp_path: Path):
    completed = type("Completed", (), {"stdout": ""})()
    with patch("blower_inspection.tao_training.subprocess.run", return_value=completed):
        with pytest.raises(RuntimeError, match="empty VisualChangeNet spec: experiment_spec.yaml"):
            copy_default_visual_changenet_spec(tmp_path / "spec.yaml")


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
