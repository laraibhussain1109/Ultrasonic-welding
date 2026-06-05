import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from blower_inspection.config import PartModelConfig
from blower_inspection.trainer import NormalTemplateTrainer


def fan_image(size=256, defect=False):
    image = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    cv2.circle(image, center, 105, (120, 120, 120), -1)
    cv2.circle(image, center, 35, (25, 25, 25), -1)
    for sector in range(24):
        angle = 2 * np.pi * sector / 24
        p1 = (int(center[0] + 42 * np.cos(angle)), int(center[1] + 42 * np.sin(angle)))
        p2 = (int(center[0] + 100 * np.cos(angle)), int(center[1] + 100 * np.sin(angle)))
        cv2.line(image, p1, p2, (190, 190, 190), 3)
    if defect:
        cv2.line(image, (120, 80), (160, 135), (255, 255, 255), 6)
    return image


def test_train_and_detect_synthetic_defect(tmp_path):
    normal_dir = tmp_path / "normal"
    normal_dir.mkdir()
    for i in range(5):
        cv2.imwrite(str(normal_dir / f"good_{i}.png"), fan_image())
    config = PartModelConfig(
        id="TEST",
        name="Synthetic",
        normal_image_dir=normal_dir,
        model_file=tmp_path / "model.npz",
        result_dir=tmp_path / "results",
        expected_fins=24,
        anomaly_threshold=3.0,
        min_defect_area_px=20,
        max_bad_sector_ratio=0.5,
    )
    trainer = NormalTemplateTrainer(image_size=(256, 256))
    trainer.train(config)
    good = trainer.inspect(config, fan_image(), save_outputs=False)
    bad = trainer.inspect(config, fan_image(defect=True), save_outputs=False)
    assert good.status == "PASS"
    assert bad.status == "FAIL"
    assert bad.defect_area_px > good.defect_area_px
