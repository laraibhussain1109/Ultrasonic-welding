import pytest
import numpy as np

pytest.importorskip("cv2", exc_type=ImportError)

from blower_inspection.cli import build_parser
from blower_inspection.dataset import prepare_yolo_dataset


def test_prepare_dataset_cli_options():
    args = build_parser().parse_args(
        ["prepare-dataset", "BF-001", "camera-images", "--output", "normal-crops", "--replace"]
    )

    assert args.command == "prepare-dataset"
    assert args.model_id == "BF-001"
    assert args.source == "camera-images"
    assert args.output == "normal-crops"
    assert args.replace is True


def test_train_accepts_tao_model_file():
    args = build_parser().parse_args(["train", "BF-001", "--model-file", "export.onnx"])
    assert args.model_file == "export.onnx"


def test_in_place_prepare_can_backup_originals(tmp_path):
    import cv2

    class Detector:
        def exact_crop(self, frame):
            return frame[1:3, 1:4]

    normal = tmp_path / "normal"
    normal.mkdir()
    original = np.zeros((5, 6, 3), dtype=np.uint8)
    original[:, :, 1] = 200
    cv2.imwrite(str(normal / "good.png"), original)

    result = prepare_yolo_dataset(
        normal, normal, replace=True, backup_in_place=True, detector=Detector()
    )

    assert result.written == 1
    assert result.backup_dir is not None
    assert cv2.imread(str(result.backup_dir / "good.png")).shape[:2] == (5, 6)
    assert cv2.imread(str(normal / "good.png")).shape[:2] == (2, 3)
