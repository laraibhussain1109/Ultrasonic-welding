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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="blower-inspection")
    parser.add_argument("--models", default="config/models.json", help="Path to model registry JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-models", help="Show configured part models")

    train = sub.add_parser("train", help="Train selected part model from its normal-image folder")
    train.add_argument("model_id")

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
    prepare.add_argument("--replace", action="store_true", help="Replace existing crops")

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
        inspector = inspector_for_model(model)
        if model.algorithm == "hybrid_patchcore_padim" and model.yolo_model_path is None:
            raise SystemExit(f"No yolo_model_path is configured for {model.id}")
        if model.algorithm == "nvidia_tao":
            print(f"Calibrating TAO export {model.model_file} with normals in {model.normal_image_dir}...")
        else:
            print(
                f"Auto-cropping {model.normal_image_dir} in memory with {model.yolo_model_path} "
                "before training (source images will not be modified)..."
            )
        output = inspector.train(model)
        print(f"Trained {model.id}: {output}")
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
        result = prepare_yolo_dataset(
            args.source,
            args.output or model.normal_image_dir,
            model.yolo_model_path,
            confidence=args.confidence if args.confidence is not None else model.yolo_confidence,
            recursive=not args.no_recursive,
            replace=args.replace,
        )
        print(
            f"Dataset prepared: discovered={result.discovered} written={result.written} "
            f"skipped={result.skipped} failed={result.failed} manifest={result.manifest}"
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
