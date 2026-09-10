"""Launch NVIDIA TAO VisualChangeNet tasks in the official Docker image."""

from __future__ import annotations

import subprocess
import re
import hashlib
import shutil
from pathlib import Path

VISUAL_CHANGENET_MODULE = (
    "nvidia_tao_pytorch.cv.visual_changenet.entrypoint.visual_changenet"
)
VISUAL_CHANGENET_SPECS = "nvidia_tao_pytorch/cv/visual_changenet/experiment_specs"
PRETRAINED_FILENAME = "changenet_segment_levir_cd.pth"
PRETRAINED_NGC_MODEL = (
    "nvidia/tao/visual_changenet_segmentation_levircd:"
    "visual_changenet_levircd_trainable_v1.0"
)


def find_visual_changenet_pretrained(search_roots: list[str | Path]) -> Path | None:
    """Find the NVIDIA LEVIR-CD checkpoint without assuming its download folder."""
    matches: set[Path] = set()
    for value in search_roots:
        root = Path(value).expanduser().resolve()
        if root.is_file() and root.name == PRETRAINED_FILENAME:
            matches.add(root)
        elif root.is_dir():
            matches.update(path.resolve() for path in root.rglob(PRETRAINED_FILENAME))
    if len(matches) > 1:
        digests = {_file_sha256(path) for path in matches}
        if len(digests) == 1:
            # NGC/manual downloads often leave the same checkpoint both at the
            # project root and in a versioned directory. Identical bytes are
            # interchangeable; choose the shortest deterministic path.
            return min(matches, key=lambda path: (len(path.parts), str(path)))
        choices = "\n  ".join(str(path) for path in sorted(matches))
        raise ValueError(f"Multiple VisualChangeNet pretrained checkpoints found; select one with --pretrained-model:\n  {choices}")
    return next(iter(matches), None)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_visual_changenet_pretrained(value: str | Path) -> Path:
    """Accept either the checkpoint itself or its NGC download directory."""
    candidate = Path(value).expanduser().resolve()
    found = find_visual_changenet_pretrained([candidate])
    if found is None:
        raise FileNotFoundError(
            f"{PRETRAINED_FILENAME} was not found at or below: {candidate}"
        )
    return found


def _install_checkpoint_canonically(checkpoint: Path, output_dir: Path) -> Path:
    """Ensure the checkpoint also exists at the stable, non-versioned path."""
    destination = output_dir / PRETRAINED_FILENAME
    if checkpoint.resolve() == destination.resolve():
        return destination
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(checkpoint, temporary)
    if _file_sha256(temporary) != _file_sha256(checkpoint):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("VisualChangeNet checkpoint copy verification failed")
    temporary.replace(destination)
    return destination


def download_visual_changenet_pretrained(
    output_dir: str | Path = "data/models/pretrained",
) -> Path:
    """Download the TAO trainable checkpoint with NVIDIA's NGC CLI."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = find_visual_changenet_pretrained([output])
    if existing is not None:
        return _install_checkpoint_canonically(existing, output)
    command = [
        "ngc", "registry", "model", "download-version", PRETRAINED_NGC_MODEL,
        "--dest", str(output),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "NVIDIA NGC CLI (`ngc`) is not installed or not on PATH. Install the official NGC CLI, "
            "authenticate it, then rerun `tao-download-weights`."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "NGC rejected the VisualChangeNet model download. This is not proof that the API key is "
            "wrong: verify the fully-qualified public model identifier, accept any model terms in NGC, "
            "and confirm access with `ngc registry model info "
            f"{PRETRAINED_NGC_MODEL}`. If a key was shown in a screenshot or log, revoke it first."
        ) from exc
    checkpoint = find_visual_changenet_pretrained([output])
    if checkpoint is None:
        raise FileNotFoundError(
            f"NGC download completed but {PRETRAINED_FILENAME} was not found under {output}"
        )
    return _install_checkpoint_canonically(checkpoint, output)


def ensure_visual_changenet_pretrained(
    output_dir: str | Path = "data/models/pretrained",
    *,
    search_roots: list[str | Path] | None = None,
) -> Path:
    """Install an existing manual download canonically, or download from NGC."""
    output = Path(output_dir).resolve()
    existing = find_visual_changenet_pretrained([output])
    if existing is not None:
        return _install_checkpoint_canonically(existing, output)
    found = find_visual_changenet_pretrained(search_roots or [Path.cwd()])
    if found is None:
        return download_visual_changenet_pretrained(output)
    return _install_checkpoint_canonically(found, output)


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


def _replace_yaml_section_scalar(text: str, section: str, key: str, value: str) -> str:
    """Replace a direct scalar child in one top-level YAML section."""
    section_match = re.search(rf"(?m)^{re.escape(section)}:\s*$", text)
    if section_match is None:
        raise ValueError(f"VisualChangeNet spec is missing `{section}:`")
    next_section = re.search(r"(?m)^\S[^\n]*:\s*(?:.*)?$", text[section_match.end():])
    end = section_match.end() + (next_section.start() if next_section else len(text[section_match.end():]))
    block = text[section_match.end():end]
    updated = _replace_yaml_scalar(block, key, value)
    return text[:section_match.end()] + updated + text[end:]


def find_training_checkpoint(results_dir: str | Path) -> Path:
    """Select the stable checkpoint, or the newest completed TAO checkpoint."""
    train_dir = Path(results_dir).resolve() / "train"
    preferred = train_dir / "changenet.pth"
    if preferred.is_file():
        return preferred
    candidates = [path for path in train_dir.rglob("*.pth") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No trained VisualChangeNet .pth checkpoint found under {train_dir}")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))


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
    filename = filenames[variant]
    # TAO images do not all use the same Python minor version. Locate the spec
    # rather than embedding (for example) ``python3.12`` in its container path.
    # The fixed suffix and filename are application constants, not shell input.
    script = (
        "spec=$(find /usr/local/lib /opt -type f "
        f"-path '*/{VISUAL_CHANGENET_SPECS}/{filename}' -print -quit 2>/dev/null); "
        "if [ -z \"$spec\" ]; then "
        f"echo 'VisualChangeNet default spec {filename} was not found in the TAO image' >&2; "
        "exit 2; fi; cat \"$spec\""
    )
    # Override the TAO entrypoint. Otherwise its release/license banner is sent
    # to stdout before the YAML, corrupting the generated spec.
    command = ["docker", "run", "--rm", "--entrypoint", "sh", image, "-c", script]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(
            f"Unable to read the VisualChangeNet {variant} default spec from {image}{suffix}"
        ) from exc
    if not completed.stdout.strip():
        raise RuntimeError(f"TAO returned an empty VisualChangeNet spec: {filename}")
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
    epochs: int = 50,
    export_file: str | Path | None = None,
) -> None:
    """Run a VisualChangeNet train/export task with the project mounted.

    The experiment spec must be inside ``project_dir`` because dataset and
    result paths in the spec must be meaningful inside the Linux container.
    """
    if task not in {"train", "export"}:
        raise ValueError(f"Unsupported VisualChangeNet task: {task}")
    if task == "train" and epochs < 2:
        raise ValueError("VisualChangeNet training requires at least 2 epochs; use --epochs 50 or more")
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
    expected_export: Path | None = None
    if task == "export":
        checkpoint = find_training_checkpoint(result_host)
        checkpoint_relative = checkpoint.relative_to(project)
        expected_export = Path(export_file or project / "data/models/visual_changenet.onnx").resolve()
        try:
            export_relative = expected_export.relative_to(project)
        except ValueError as exc:
            raise ValueError(f"TAO ONNX export path must be inside the project: {expected_export}") from exc
        expected_export.parent.mkdir(parents=True, exist_ok=True)
        runtime_text = _replace_yaml_section_scalar(
            runtime_text, "export", "checkpoint",
            f'"{(Path("/workspace/project") / checkpoint_relative).as_posix()}"',
        )
        runtime_text = _replace_yaml_section_scalar(
            runtime_text, "export", "onnx_file",
            f'"{(Path("/workspace/project") / export_relative).as_posix()}"',
        )
        runtime_text = _replace_yaml_section_scalar(runtime_text, "export", "batch_size", "1")
    if task == "train":
        if dataset_dir is None:
            raise ValueError("TAO training requires --dataset pointing to the existing TAO_VCN_DATASET")
        dataset = validate_visual_changenet_dataset(dataset_dir)
        if pretrained_model is not None and from_scratch:
            raise ValueError("Use either --pretrained-model or --from-scratch, not both")
        if pretrained_model is None and not from_scratch:
            pretrained_model = find_visual_changenet_pretrained(
                [project, project / "data/models/pretrained", dataset.parent, Path.home() / "Downloads"]
            )
            if pretrained_model is None:
                raise FileNotFoundError(
                    f"Unable to find {PRETRAINED_FILENAME}. Run `python -m src.blower_inspection.cli "
                    "tao-download-weights`, pass its real path with --pretrained-model, or explicitly "
                    "choose --from-scratch."
                )
        runtime_text = _replace_yaml_scalar(runtime_text, "root_dir", "/data/TAO_VCN_DATASET")
        runtime_text = _replace_yaml_scalar(runtime_text, "num_epochs", str(epochs))
        if pretrained_model is not None:
            pretrained = resolve_visual_changenet_pretrained(pretrained_model)
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
        if expected_export is not None and not expected_export.is_file():
            raise FileNotFoundError(
                f"TAO export command completed but ONNX was not created at {expected_export}"
            )
    finally:
        if runtime_spec is not None:
            runtime_spec.unlink(missing_ok=True)
