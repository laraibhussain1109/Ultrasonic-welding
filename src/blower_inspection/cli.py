"""Command-line tools for training, inspection, and user management."""

from __future__ import annotations

import argparse
import getpass
from pathlib import Path

import cv2

from .auth import AuthStore
from .config import ModelRegistry, ensure_model_folders
from .tao_inspector import inspector_for_model
from .esp32_output import ESP32FailOutput
from .dataset import prepare_yolo_dataset
from .tao_training import (
    copy_default_visual_changenet_spec,
    download_visual_changenet_pretrained,
    find_visual_changenet_pretrained,
    run_visual_changenet_task,
)


def resolve_onnx_model(value: str | Path) -> Path:
    """Resolve an ONNX file, accepting a directory only when unambiguous."""
    candidate = Path(value).expanduser().resolve()
    if candidate.is_dir():
        exports = sorted(candidate.rglob("*.onnx"))
        if len(exports) == 1:
            return exports[0]
        if not exports:
            raise ValueError(
                f"No .onnx export exists in {candidate}. You selected a folder, not a TAO model. "
                "NVIDIA TAO is a separate training toolkit: train and export a visual-anomaly model "
                "first, then pass the exact .onnx file to --model-file. See docs/tao_deployment.md."
            )
        names = ", ".join(str(path) for path in exports[:5])
        raise ValueError(
            f"Multiple ONNX exports were found in {candidate}; pass one exact file to --model-file: {names}"
        )
    if not candidate.is_file():
        raise ValueError(
            f"TAO ONNX export does not exist: {candidate}. --model-file must name the exported .onnx "
            "file, not its intended output folder. See docs/tao_deployment.md."
        )
    if candidate.suffix.lower() != ".onnx":
        raise ValueError(f"TAO model must be an .onnx export, not: {candidate}")
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="blower-inspection")
    parser.add_argument("--models", default="config/models.json", help="Path to model registry JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-models", help="Show configured part models")

    train = sub.add_parser(
        "train",
        help="Train legacy models or calibrate a NVIDIA TAO ONNX export from normal images",
    )
    train.add_argument("model_id")
    train.add_argument("--model-file", help="TAO Deploy ONNX export to import before calibration")

    tao_train = sub.add_parser("tao-train", help="Train VisualChangeNet in the TAO Docker image")
    tao_train.add_argument("model_id", nargs="?", help="Part model; defaults to active_model")
    tao_train.add_argument("--spec", required=True, help="VisualChangeNet training experiment YAML inside this project")
    tao_train.add_argument("--image", default="nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt")
    tao_train.add_argument("--dataset", required=True, help="Existing TAO CNDataset root containing A/B/label/list")
    tao_train.add_argument("--results-dir", help="Host results directory inside this project")
    weights = tao_train.add_mutually_exclusive_group()
    weights.add_argument("--pretrained-model", help="Host VisualChangeNet .pth checkpoint")
    weights.add_argument("--from-scratch", action="store_true", help="Explicitly train without pretrained weights")

    tao_export = sub.add_parser("tao-export", help="Export VisualChangeNet in the TAO Docker image")
    tao_export.add_argument("model_id", nargs="?", help="Part model; defaults to active_model")
    tao_export.add_argument("--spec", required=True, help="VisualChangeNet export experiment YAML inside this project")
    tao_export.add_argument("--image", default="nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt")
    tao_export.add_argument("--results-dir", help="Same host results directory used for training")

    tao_init = sub.add_parser("tao-init", help="Copy the installed TAO VisualChangeNet default spec")
    tao_init.add_argument("model_id", nargs="?", help="Part model; defaults to active_model")
    tao_init.add_argument("--variant", choices=["segmentation", "classification"], default="segmentation")
    tao_init.add_argument("--output", help="Destination YAML; defaults under specs/visual_changenet")
    tao_init.add_argument("--image", default="nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt")

    tao_weights = sub.add_parser("tao-download-weights", help="Download NVIDIA's VisualChangeNet LEVIR-CD checkpoint")
    tao_weights.add_argument("--output", default="data/models/pretrained")

    tao_find_weights = sub.add_parser("tao-find-weights", help="Print an existing VisualChangeNet checkpoint path")
    tao_find_weights.add_argument("--search", default=".", help="File or directory to search recursively")

    inspect = sub.add_parser("inspect", help="Inspect one image with a trained model")
    inspect.add_argument("model_id")
    inspect.add_argument("image")
    inspect.add_argument("--esp32-output", action="store_true", help="Send FAIL/PASS output to ESP32 after inspecting the image")

    prepare = sub.add_parser("prepare-dataset", help="YOLO-crop known-good full-FOV images for training")
    prepare.add_argument("model_id", help="Configured part model whose YOLO checkpoint should be used")
    prepare.add_argument("source", help="Directory containing known-good full-FOV images")
    prepare.add_argument("--output", help="Crop destination; defaults to the model normal_image_dir")
    prepare.add_argument("--confidence", type=float, help="Override configured YOLO confidence")
    prepare.add_argument("--no-recursive", action="store_true", help="Read only the source directory")
    prepare.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing crops; in-place preparation always creates a sibling backup",
    )

    add_user = sub.add_parser("add-user", help="Create or update a UI login")
    add_user.add_argument("username")
    add_user.add_argument("--role", choices=["admin", "user"], default="user")
    add_user.add_argument("--password", help="Password. If omitted, you will be prompted.")
    add_user.add_argument("--users", default="config/users.json", help="Path to users JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = ModelRegistry(args.models)
    ensure_model_folders(registry)
    if args.command == "list-models":
        for model in registry.all():
            trained = "trained" if model.model_file.exists() else "not trained"
            print(f"{model.id}\t{model.name}\t{trained}\ttraining={model.normal_image_dir}")
        return 0

    if args.command == "train":
        model = registry.get(args.model_id)
        if args.model_file:
            try:
                candidate = resolve_onnx_model(args.model_file)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            model = registry.update_model_settings(model.id, model_file=candidate)
        inspector = inspector_for_model(model)
        if model.algorithm == "hybrid_patchcore_padim" and model.yolo_model_path is None:
            raise SystemExit(f"No yolo_model_path is configured for {model.id}")
        if model.algorithm == "nvidia_tao":
            if not model.model_file.is_file():
                raise SystemExit(
                    f"VisualChangeNet export not found: {model.model_file}. The `train` command calibrates an "
                    "exported model; it does not run TAO weight training. Run `tao-train --spec <train.yaml>`, "
                    "then `tao-export --spec <export.yaml>`, or pass the resulting ONNX with --model-file."
                )
            print(f"Calibrating TAO export {model.model_file} with normals in {model.normal_image_dir}...")
        else:
            print(
                f"Auto-cropping {model.normal_image_dir} in memory with {model.yolo_model_path} "
                "before training (source images will not be modified)..."
            )
        output = inspector.train(model)
        action = "Calibrated" if model.algorithm == "nvidia_tao" else "Trained"
        print(f"{action} {model.id}: {output}")
        return 0

    if args.command == "tao-init":
        model = registry.get(args.model_id) if args.model_id else registry.active()
        output = Path(args.output or f"specs/visual_changenet/{model.id.lower()}_{args.variant}.yaml")
        copied = copy_default_visual_changenet_spec(output, variant=args.variant, image=args.image)
        print(f"Copied TAO 7.1 VisualChangeNet {args.variant} spec to {copied}")
        print("The TAO Docker entrypoint was bypassed so its banner is not embedded in the YAML.")
        print("Edit dataset, results, pretrained-model, and training fields before running tao-train.")
        return 0

    if args.command == "tao-download-weights":
        checkpoint = download_visual_changenet_pretrained(args.output)
        print(f"VisualChangeNet pretrained checkpoint: {checkpoint}")
        return 0

    if args.command == "tao-find-weights":
        checkpoint = find_visual_changenet_pretrained([args.search])
        if checkpoint is None:
            raise SystemExit(f"changenet_segment_levir_cd.pth was not found under {Path(args.search).resolve()}")
        print(checkpoint)
        return 0

    if args.command in {"tao-train", "tao-export"}:
        model = registry.get(args.model_id) if args.model_id else registry.active()
        if model.algorithm != "nvidia_tao":
            raise SystemExit(f"{model.id} is not configured for NVIDIA TAO")
        task = "train" if args.command == "tao-train" else "export"
        print(f"Starting TAO VisualChangeNet {task} with {args.spec} in {args.image}...")
        kwargs = {
            "image": args.image,
            "results_dir": args.results_dir or f"data/results/{model.id}/tao",
        }
        if task == "train":
            kwargs.update(
                dataset_dir=args.dataset,
                pretrained_model=args.pretrained_model,
                from_scratch=args.from_scratch,
            )
        run_visual_changenet_task(task, args.spec, **kwargs)
        print(f"TAO VisualChangeNet {task} completed")
        return 0

    if args.command == "inspect":
        model = registry.get(args.model_id)
        inspector = inspector_for_model(model)
        image = cv2.imread(str(Path(args.image)))
        if image is None:
            raise SystemExit(f"Unable to read image: {args.image}")
        result = inspector.inspect(model, image)
        if args.esp32_output:
            esp32 = ESP32FailOutput()
            if not esp32.set_fail(not result.is_pass):
                print(f"ESP32 output warning: {esp32.last_error}")
        print(
            f"{result.status} score={result.anomaly_score:.2f} "
            f"area={result.defect_area_px} bad_sector_ratio={result.bad_sector_ratio:.3f} "
            f"overlay={result.overlay_path} report={result.report_path}"
        )
        return 0

    if args.command == "prepare-dataset":
        model = registry.get(args.model_id)
        if model.yolo_model_path is None:
            raise SystemExit(f"No yolo_model_path is configured for {model.id}")
        output = Path(args.output or model.normal_image_dir)
        source = Path(args.source)
        in_place = source.resolve() == output.resolve()
        # The normal folder is the most natural place for operators to collect
        # images.  Make that workflow safe and useful: an omitted --output means
        # "prepare the configured normal folder", with an automatic backup.
        replace = args.replace or (args.output is None and in_place)
        if args.output is None and in_place and not args.replace:
            print("Source is the configured normal folder; cropping in place after creating a full backup...")
        result = prepare_yolo_dataset(
            args.source,
            output,
            model.yolo_model_path,
            confidence=args.confidence if args.confidence is not None else model.yolo_confidence,
            recursive=not args.no_recursive,
            replace=replace,
            backup_in_place=in_place,
        )
        print(
            f"Dataset prepared: discovered={result.discovered} written={result.written} "
            f"skipped={result.skipped} failed={result.failed} manifest={result.manifest} "
            f"backup={result.backup_dir or '-'}"
        )
        return 1 if result.failed else 0

    if args.command == "add-user":
        password = args.password or getpass.getpass("Password: ")
        AuthStore(args.users).upsert_user(args.username, password, args.role)
        print(f"Saved user '{args.username}' with role '{args.role}'")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
