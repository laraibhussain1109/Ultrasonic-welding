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
    command = ["docker", "run", "--rm", image, "cat", source]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    if not completed.stdout.strip():
        raise RuntimeError(f"TAO returned an empty VisualChangeNet spec: {source}")
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(completed.stdout, encoding="utf-8")
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
