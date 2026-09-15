from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from blower_inspection.geometry_inspector import FinGeometryInspector, GeometryEvidence, glare_evidence
from blower_inspection.hybrid_fusion import TaoEvidence, fuse_evidence
from blower_inspection.reference_bank import ReferenceBank, structural_descriptor
from blower_inspection.registration import register_to_reference
from blower_inspection.yolo_tracking import RotatingPartInspector, TrackedPart


def fins(missing=None, tilted=False, glare=False):
    image = np.zeros((160, 320, 3), np.uint8)
    for i, y in enumerate(range(25, 140, 10)):
        if i == missing:
            continue
        cv2.line(image, (5, y), (315, y + (18 if tilted and i == 5 else 0)), (150, 150, 150), 2)
    for x in (80, 240):
        cv2.line(image, (x, 5), (x, 155), (220, 220, 220), 5)
    if glare:
        cv2.ellipse(image, (170, 75), (55, 28), 0, 0, 360, (255, 255, 255), -1)
    return image


def evidence(tao=0, geometry=0, periodicity=0, broken=0, orientation=0):
    mask = np.zeros((16, 16), bool)
    t = TaoEvidence(mask.astype(np.float32), mask.astype(np.float32), tao, tao, mask, 0)
    g = GeometryEvidence(geometry, orientation, 0, 0, broken, periodicity, 1, mask)
    return t, g


def test_reference_bank_is_diverse_and_leave_one_out(tmp_path: Path):
    images = [np.roll(fins(), i * 3, axis=0) for i in range(8)]
    sources = [tmp_path / f"n{i}.png" for i in range(8)]
    bank = ReferenceBank.build(images, sources, tmp_path / "bank", 4)
    assert len(bank.images) == 4
    match = bank.candidates(bank.images[0], 1, exclude_source=bank.sources[0])[0]
    assert match.index != 0
    assert ReferenceBank.load(tmp_path / "bank").manifest_sha256 == bank.manifest_sha256


def test_descriptor_resists_broad_brightness_change():
    normal = fins()
    bright = cv2.add(normal, np.full_like(normal, 60))
    assert np.linalg.norm(structural_descriptor(normal) - structural_descriptor(bright)) < .35


@pytest.mark.parametrize("angle,shift", [(0, (4, -3)), (1.5, (2, 1))])
def test_limited_registration_accepts_small_motion(angle, shift):
    reference = fins()
    matrix = cv2.getRotationMatrix2D((160, 80), angle, 1)
    matrix[:, 2] += shift
    moved = cv2.warpAffine(reference, matrix, (320, 160))
    result = register_to_reference(moved, reference, min_correlation=.3)
    assert result.success


def test_registration_rejects_excessive_motion():
    moved = np.roll(fins(), 70, axis=1)
    result = register_to_reference(moved, fins(), max_translation_ratio=.05)
    assert not result.success
    assert result.failure_reason is not None


def test_geometry_calibration_and_periodicity_disruption():
    normal = [np.roll(fins(), dy, axis=0) for dy in (-1, 0, 1)]
    inspector = FinGeometryInspector(FinGeometryInspector.calibrate(normal))
    baseline = inspector.inspect(fins())
    missing = inspector.inspect(fins(missing=5))
    assert missing.periodicity_score >= baseline.periodicity_score


def test_support_ribs_and_broad_glare_are_not_broken_fins():
    calibration = FinGeometryInspector.calibrate([fins(), fins(), fins()])
    result = FinGeometryInspector(calibration).inspect(fins(glare=True))
    assert result.broken_fin_score < 1
    assert glare_evidence(fins(glare=True), (160, 320)).score > 0


def test_hybrid_fusion_glare_and_structural_rules():
    t, g = evidence(tao=1.3, geometry=.08, periodicity=.05)
    assert fuse_evidence(t, g, glare_score=.9, registration_valid=True).status == "PASS"
    t, g = evidence(tao=1.1, geometry=.8)
    assert fuse_evidence(t, g, glare_score=.1, registration_valid=True).status == "FAIL"
    t, g = evidence(tao=.1, geometry=1.2, broken=1.2)
    assert fuse_evidence(t, g, glare_score=.9, registration_valid=True).immediate_failure
    assert fuse_evidence(t, g, glare_score=0, registration_valid=False).status == "VIEW INVALID"


def test_temporal_candidate_requires_persistence_and_catastrophe_is_immediate():
    tracker = RotatingPartInspector(counting_line_ratio=.8, weak_candidate_required_views=2)
    tracker.observe_tracks([TrackedPart(1, (10, 0, 20, 20), .9)], 100)
    tracker.record_inspection(1, is_pass=False, anomaly_score=.8, immediate_failure=False, provisional_candidate=True)
    assert not tracker.latched_failure(1)
    tracker.record_inspection(1, is_pass=False, anomaly_score=.9, immediate_failure=False, provisional_candidate=True)
    assert tracker.latched_failure(1)

    severe = RotatingPartInspector(counting_line_ratio=.8)
    severe.observe_tracks([TrackedPart(2, (10, 0, 20, 20), .9)], 100)
    severe.record_inspection(2, is_pass=False, anomaly_score=1.2, immediate_failure=True, geometry_score=1.2)
    assert severe.latched_failure(2)
