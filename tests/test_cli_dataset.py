import pytest

pytest.importorskip("cv2", exc_type=ImportError)

from blower_inspection.cli import build_parser


def test_prepare_dataset_cli_options():
    args = build_parser().parse_args(
        ["prepare-dataset", "BF-001", "camera-images", "--output", "normal-crops", "--replace"]
    )

    assert args.command == "prepare-dataset"
    assert args.model_id == "BF-001"
    assert args.source == "camera-images"
    assert args.output == "normal-crops"
    assert args.replace is True
