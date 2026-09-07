"""Launch NVIDIA TAO VisualChangeNet tasks in the official Docker image."""

from __future__ import annotations

import subprocess
import re
from pathlib import Path

VISUAL_CHANGENET_MODULE = (
    "nvidia_tao_pytorch.cv.visual_changenet.entrypoint.visual_changenet"
)
VISUAL_CHANGENET_SPECS = (
    "/usr/local/lib/python3.12/dist-packages/nvidia_tao_pytorch/cv/"
    "visual_changenet/experiment_specs"
)


def _clean_spec_output(output: str) -> str:
    """Strip a TAO entrypoint banner accidentally emitted before YAML."""
    lines = output.replace("\ufeff", "").splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line.startswith(("encryption_key:", "task:"))),
        None,
    )
    if start is None:
        raise RuntimeError("TAO output did not contain a VisualChangeNet YAML document")
    cleaned = "\n".join(lines[start:]).strip() + "\n"
    if "TAO Toolkit PyTorch" in cleaned:
        raise RuntimeError("TAO container banner was embedded in the VisualChangeNet YAML")
    return cleaned


def validate_visual_changenet_spec(path: str | Path) -> None:
    """Reject the known banner-corrupted spec before starting a long container."""
    spec = Path(path)
    text = spec.read_text(encoding="utf-8-sig")
    first_yaml = min(
        (position for marker in ("encryption_key:", "task:") if (position := text.find(marker)) >= 0),
        default=-1,
    )
    if first_yaml < 0:
        raise ValueError(f"VisualChangeNet spec contains no `task:` YAML key: {spec}")
    prefix = text[:first_yaml]
    if "TAO Toolkit" in prefix or "NVIDIA Release" in prefix or "Copyright" in prefix:
        raise ValueError(
            f"VisualChangeNet spec contains a Docker/TAO banner before the YAML: {spec}. "
            "Regenerate it with `tao-init` using the updated application."
        )


def validate_visual_changenet_dataset(path: str | Path) -> Path:
    """Validate an existing CNDataset without changing it."""
    root = Path(path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"TAO VisualChangeNet dataset does not exist: {root}")
    for name in ("A", "B", "label", "list"):
        if not (root / name).is_dir():
            raise ValueError(f"TAO dataset validation failed: missing directory {root / name}")
    for name in ("train.txt", "val.txt", "test.txt"):
        split = root / "list" / name
        if not split.is_file() or not split.read_text(encoding="utf-8-sig").strip():
            raise ValueError(f"TAO dataset validation failed: missing or empty split {split}")
    image_names = {
        folder: {item.name for item in (root / folder).iterdir() if item.is_file()}
        for folder in ("A", "B", "label")
    }
    if not image_names["A"]:
        raise ValueError(f"TAO dataset validation failed: no reference images in {root / 'A'}")
    if image_names["A"] != image_names["B"] or image_names["A"] != image_names["label"]:
        raise ValueError("TAO dataset validation failed: A, B, and label filenames do not match exactly")
    return root


def _replace_yaml_scalar(text: str, key: str, value: str, *, top_level: bool = False) -> str:
    indent = "" if top_level else r"[ \t]+"
    pattern = rf"(?m)^{indent}{re.escape(key)}\s*:\s*.*$"
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1:
        raise ValueError(f"VisualChangeNet spec must contain exactly one `{key}:` field; found {len(matches)}")
    original = matches[0].group(0)
    leading = original[: len(original) - len(original.lstrip())]
    return text[:matches[0].start()] + f"{leading}{key}: {value}" + text[matches[0].end():]


def copy_default_visual_changenet_spec(
    destination: str | Path,
    *,
    variant: str = "segmentation",
    image: str = "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt",
) -> Path:
    """Copy the exact TAO-container default spec into the project."""
    filenames = {
        "segmentation": "experiment_spec.yaml",
        "classification": "experiment_spec_classify.yaml",
    }
    if variant not in filenames:
        raise ValueError(f"Unsupported VisualChangeNet variant: {variant}")
    source = f"{VISUAL_CHANGENET_SPECS}/{filenames[variant]}"
    # Override the TAO entrypoint. Otherwise its release/license banner is sent
    # to stdout before `cat`, producing an invalid YAML file.
    command = ["docker", "run", "--rm", "--entrypoint", "cat", image, source]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    if not completed.stdout.strip():
        raise RuntimeError(f"TAO returned an empty VisualChangeNet spec: {source}")
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(_clean_spec_output(completed.stdout), encoding="utf-8")
    temporary.replace(output)
    return output


def run_visual_changenet_task(
    task: str,
    spec: str | Path,
    *,
    image: str = "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt",
    project_dir: str | Path = ".",
    dataset_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    pretrained_model: str | Path | None = None,
    from_scratch: bool = False,
) -> None:
    """Run a VisualChangeNet train/export task with the project mounted.

    The experiment spec must be inside ``project_dir`` because dataset and
    result paths in the spec must be meaningful inside the Linux container.
    """
    if task not in {"train", "export"}:
        raise ValueError(f"Unsupported VisualChangeNet task: {task}")
    project = Path(project_dir).resolve()
    spec_path = Path(spec).resolve()
    if not spec_path.is_file():
        raise FileNotFoundError(f"VisualChangeNet experiment spec not found: {spec_path}")
    validate_visual_changenet_spec(spec_path)
    try:
        relative_spec = spec_path.relative_to(project)
    except ValueError as exc:
        raise ValueError(
            f"Experiment spec must be inside the mounted project {project}: {spec_path}"
        ) from exc
    container_spec = Path("/workspace/project") / relative_spec
    mounts = ["-v", f"{project}:/workspace/project"]
    runtime_text = spec_path.read_text(encoding="utf-8-sig")
    result_host = Path(results_dir or project / "data/results/tao_visual_changenet").resolve()
    result_host.mkdir(parents=True, exist_ok=True)
    try:
        result_relative = result_host.relative_to(project)
    except ValueError as exc:
        raise ValueError(f"TAO results directory must be inside the project: {result_host}") from exc
    runtime_text = _replace_yaml_scalar(
        runtime_text,
        "results_dir",
        f'"{(Path("/workspace/project") / result_relative).as_posix()}"',
        top_level=True,
    )
    if task == "train":
        if dataset_dir is None:
            raise ValueError("TAO training requires --dataset pointing to the existing TAO_VCN_DATASET")
        dataset = validate_visual_changenet_dataset(dataset_dir)
        if pretrained_model is not None and from_scratch:
            raise ValueError("Use either --pretrained-model or --from-scratch, not both")
        runtime_text = _replace_yaml_scalar(runtime_text, "root_dir", "/data/TAO_VCN_DATASET")
        if pretrained_model is not None:
            pretrained = Path(pretrained_model).resolve()
            if not pretrained.is_file():
                raise FileNotFoundError(f"TAO pretrained model not found: {pretrained}")
            mounts.extend(["-v", f"{pretrained}:/pretrained/model.pth:ro"])
            runtime_text = _replace_yaml_scalar(runtime_text, "pretrained_model_path", "/pretrained/model.pth")
        elif from_scratch:
            runtime_text = _replace_yaml_scalar(runtime_text, "pretrained_model_path", "null")
        elif re.search(r"(?m)^\s+pretrained_model_path\s*:\s*/results/", runtime_text):
            raise ValueError(
                "The copied TAO spec still points to NVIDIA's unavailable /results/pretrained example. "
                "Pass --pretrained-model <file.pth> or explicitly choose --from-scratch."
            )
        mounts.extend(["-v", f"{dataset}:/data/TAO_VCN_DATASET:ro"])
    runtime_spec = spec_path.with_name(f".{spec_path.stem}.runtime.yaml")
    runtime_spec.write_text(runtime_text, encoding="utf-8")
    container_spec = Path("/workspace/project") / runtime_spec.relative_to(project)
    command = [
        "docker", "run", "--rm", "--gpus", "all", "--shm-size=16g",
        "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
        *mounts, "-w", "/workspace/project",
        image, "python", "-m", VISUAL_CHANGENET_MODULE,
        task, "-e", container_spec.as_posix(),
    ]
    try:
        subprocess.run(command, check=True)
    finally:
        if runtime_spec is not None:
            runtime_spec.unlink(missing_ok=True)
