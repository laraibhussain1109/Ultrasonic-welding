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


def test_registration_scales_bounded_working_transform_to_full_crop():
    reference = cv2.resize(fins(), (1600, 800))
    matrix = np.float32([[1, 0, 8], [0, 1, -5]])
    moved = cv2.warpAffine(reference, matrix, (1600, 800))

    result = register_to_reference(moved, reference, min_correlation=.3)

    assert result.success
    assert abs(result.translation_x) > 4


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


def test_fusion_rejects_mismatched_evidence_coordinates():
    t, g = evidence(tao=1.1, geometry=.8)
    g = GeometryEvidence(g.score, g.orientation_score, g.pitch_score, g.continuity_score,
                         g.broken_fin_score, g.periodicity_score, g.support_rib_confidence,
                         np.zeros((8, 12), bool))

    with pytest.raises(ValueError, match="same fusion coordinate system"):
        fuse_evidence(t, g, glare_score=0, registration_valid=True)


def test_tao_only_evidence_must_be_localized_before_becoming_persistent():
    mask = np.zeros((100, 100), bool)
    geometry = GeometryEvidence(.1, 0, 0, 0, 0, .1, 1, mask)

    broad = np.ones_like(mask)
    broad_tao = TaoEvidence(broad.astype(np.float32), broad.astype(np.float32), 20, 20,
                            broad, int(broad.sum()))
    assert fuse_evidence(broad_tao, geometry, glare_score=0, registration_valid=True).status == "PASS"

    localized = mask.copy()
    localized[30:40, 40:50] = True
    localized_tao = TaoEvidence(localized.astype(np.float32), localized.astype(np.float32),
                                2, 2, localized, int(localized.sum()))
    decision = fuse_evidence(localized_tao, geometry, glare_score=0, registration_valid=True)
    assert decision.status == "CANDIDATE"
    assert decision.provisional_candidate


def test_pitch_anomaly_alone_is_not_labeled_missing_fin():
    mask = np.zeros((20, 20), bool)
    tao = TaoEvidence(mask.astype(np.float32), mask.astype(np.float32), .1, .1, mask, 0)
    geometry = GeometryEvidence(
        score=.5, orientation_score=0, pitch_score=2.0, continuity_score=0,
        broken_fin_score=0, periodicity_score=0, support_rib_confidence=1,
        defect_mask=mask, missing_fin_score=0,
    )

    decision = fuse_evidence(tao, geometry, glare_score=0, registration_valid=True)

    assert decision.status == "PASS"
    assert "MISSING_FIN" not in decision.reason_codes


def test_pitch_and_periodicity_harmonic_need_continuity_corroboration(monkeypatch):
    import blower_inspection.geometry_inspector as geometry_module

    image = fins()
    features = {"orientation": 0, "pitch": 30, "periodicity": .2,
                "continuity": .5, "rib_confidence": 1}
    calibration = {
        "orientation": {"median": 0, "mad": 1, "reject_delta": 4},
        "pitch": {"median": 10, "mad": 1, "reject_delta": 2},
        "periodicity": {"median": .8, "mad": .01, "reject_delta": .15},
        "continuity": {"median": .5, "mad": .01, "reject_delta": .12},
    }
    monkeypatch.setattr(geometry_module, "_features", lambda *_args: (
        features, np.zeros(image.shape[:2], bool), np.zeros(image.shape[:2], bool)))
    monkeypatch.setattr(geometry_module, "broken_fin_mask",
                        lambda *_args: np.zeros(image.shape[:2], bool))
    monkeypatch.setattr(geometry_module, "smooth_reflection_mask",
                        lambda *_args: np.zeros(image.shape[:2], bool))

    result = FinGeometryInspector(calibration).inspect(image)

    assert result.pitch_score == 2
    assert result.periodicity_score == 2
    assert result.continuity_score == 0
    assert result.missing_fin_score == 0
    assert result.score == 0


def test_missing_fin_reason_requires_corroborated_geometry_score():
    mask = np.zeros((20, 20), bool)
    tao = TaoEvidence(mask.astype(np.float32), mask.astype(np.float32), .1, .1, mask, 0)
    geometry = GeometryEvidence(
        score=1.2, orientation_score=0, pitch_score=1.5, continuity_score=1.2,
        broken_fin_score=0, periodicity_score=1.1, support_rib_confidence=1,
        defect_mask=mask, missing_fin_score=1.2,
    )

    decision = fuse_evidence(tao, geometry, glare_score=0, registration_valid=True)

    assert decision.status == "FAIL"
    assert "MISSING_FIN" in decision.reason_codes


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
