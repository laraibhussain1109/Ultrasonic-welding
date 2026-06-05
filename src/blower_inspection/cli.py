"""Command-line tools for training, inspection, and user management."""

from __future__ import annotations

import argparse
import getpass
from pathlib import Path

import cv2

from .auth import AuthStore
from .config import ModelRegistry, ensure_model_folders
from .trainer import NormalTemplateTrainer


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
    trainer = NormalTemplateTrainer()

    if args.command == "list-models":
        for model in registry.all():
            trained = "trained" if model.model_file.exists() else "not trained"
            print(f"{model.id}\t{model.name}\t{trained}\ttraining={model.normal_image_dir}")
        return 0

    if args.command == "train":
        model = registry.get(args.model_id)
        output = trainer.train(model)
        print(f"Trained {model.id}: {output}")
        return 0

    if args.command == "inspect":
        model = registry.get(args.model_id)
        image = cv2.imread(str(Path(args.image)))
        if image is None:
            raise SystemExit(f"Unable to read image: {args.image}")
        result = trainer.inspect(model, image)
        print(
            f"{result.status} score={result.anomaly_score:.2f} "
            f"area={result.defect_area_px} bad_sector_ratio={result.bad_sector_ratio:.3f} "
            f"overlay={result.overlay_path} report={result.report_path}"
        )
        return 0

    if args.command == "add-user":
        password = args.password or getpass.getpass("Password: ")
        AuthStore(args.users).upsert_user(args.username, password, args.role)
        print(f"Saved user '{args.username}' with role '{args.role}'")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
