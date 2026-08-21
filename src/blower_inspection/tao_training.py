"""Launch NVIDIA TAO VisualChangeNet tasks in the official Docker image."""

from __future__ import annotations

import subprocess
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
    command = [
        "docker", "run", "--rm", "--gpus", "all", "--shm-size=16g",
        "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
        "-v", f"{project}:/workspace/project", "-w", "/workspace/project",
        image, "python", "-m", VISUAL_CHANGENET_MODULE,
        task, "-e", container_spec.as_posix(),
    ]
    subprocess.run(command, check=True)
