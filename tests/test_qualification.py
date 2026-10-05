from pathlib import Path
from types import SimpleNamespace

import json
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", exc_type=ImportError)

from blower_inspection.qualification import run_qualification


class FrozenInspector:
    def __init__(self, statuses):
        self.statuses = iter(statuses)
        self.shapes = []

    def inspect(self, _config, image, **kwargs):
        self.shapes.append(image.shape)
        assert kwargs["crop_to_component"] is False
        status = next(self.statuses)
        return SimpleNamespace(status=status, anomaly_score=0.2,
                               reason_codes=("REGISTRATION_INVALID",) if status == "VIEW INVALID" else (),
                               registration_score=0.8 if status != "VIEW INVALID" else 0.1)


def _write(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((80, 400, 3), np.uint8))


def test_qualification_preserves_pre_cropped_roi_and_reports_invalid_separately(tmp_path):
    good, ng = tmp_path / "good", tmp_path / "ng"
    _write(good / "good_pass.png")
    _write(good / "good_invalid.png")
    _write(ng / "broken_fin" / "ng_detected.png")
    inspector = FrozenInspector(["PASS", "VIEW INVALID", "FAIL"])
    config = SimpleNamespace(id="BF", yolo_model_path=None, yolo_confidence=.7,
                             roi_ratios=None, result_dir=tmp_path,
                             surface_model_file=tmp_path / "surface.pt", model_file=tmp_path / "model.pt")

    output = run_qualification(inspector, config, good, ng)
    report = json.loads(output.read_text())

    assert inspector.shapes == [(80, 400, 3)] * 3
    assert report["confusion"]["true_good"] == 1
    assert report["confusion"]["good_invalid"] == 1
    assert report["confusion"]["detected_ng"] == 1
    assert report["good_false_reject_rate"] == 0
    assert len(report["invalid_views"]) == 1
    assert report["worst_false_positives"] == []
