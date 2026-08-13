from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", exc_type=ImportError)

from blower_inspection.dataset import prepare_yolo_dataset


class CenterCropDetector:
    def exact_crop(self, frame):
        height, width = frame.shape[:2]
        return frame[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]


class NoDetection:
    def exact_crop(self, _frame):
        raise ValueError("no part")


def write_image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((80, 160, 3), 100, dtype=np.uint8))


def test_prepare_dataset_preserves_folders_and_writes_manifest(tmp_path):
    source = tmp_path / "full_fov"
    output = tmp_path / "normal_crops"
    write_image(source / "angle_1" / "part.jpg")
    write_image(source / "angle_2" / "part.jpg")

    result = prepare_yolo_dataset(source, output, detector=CenterCropDetector())

    assert result.discovered == 2
    assert result.written == 2
    assert result.failed == 0
    assert cv2.imread(str(output / "angle_1" / "part.jpg")).shape[:2] == (40, 80)
    assert (output / "angle_2" / "part.jpg").exists()
    assert "WRITTEN" in result.manifest.read_text(encoding="utf-8")


def test_prepare_dataset_records_no_detection_without_copying_image(tmp_path):
    source = tmp_path / "full_fov"
    output = tmp_path / "normal_crops"
    write_image(source / "part.png")

    result = prepare_yolo_dataset(source, output, detector=NoDetection())

    assert result.failed == 1
    assert not (output / "part.png").exists()
    assert "FAILED_DETECTION" in result.manifest.read_text(encoding="utf-8")


def test_prepare_dataset_rejects_nested_output(tmp_path):
    source = tmp_path / "images"
    source.mkdir()

    with pytest.raises(ValueError, match="must not contain"):
        prepare_yolo_dataset(source, source / "crops", detector=CenterCropDetector())
