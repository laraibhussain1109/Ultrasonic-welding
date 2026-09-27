import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", exc_type=ImportError)

from blower_inspection.golden_reference import GoldenReferenceBank


def fin_view(shift=0, brightness=0):
    image = np.full((120, 480, 3), 80 + brightness, np.uint8)
    for y in range(15 + shift, 110, 12):
        cv2.line(image, (5, y), (474, y), (180 + brightness, 180 + brightness, 180 + brightness), 2)
    return image


def test_phase_golden_uses_multiple_parts_and_normalizes_brightness(tmp_path):
    records = []
    for part in range(4):
        for phase in range(3):
            records.append((fin_view(phase, part * 2), f"p{part}_v{phase}.png", f"part_{part}"))
    bank = GoldenReferenceBank.build(records, tmp_path / "golden", phase_bins=3)

    evidence = bank.compare(fin_view(1, 8), candidates=3, min_correlation=.1,
                            max_translation_ratio=.1, max_rotation_deg=3)

    assert evidence.registration.success
    assert bank.manifest["phase_bins"] == 3
    assert all(item["count"] >= 1 for item in bank.manifest["phases"])
    assert len({group for item in bank.manifest["phases"] for group in item["groups"]}) == 4


def test_bad_registration_is_invalid_evidence_not_a_defect(tmp_path):
    records = [(fin_view(phase), f"v{phase}.png", f"part_{phase}") for phase in range(4)]
    bank = GoldenReferenceBank.build(records, tmp_path / "golden", phase_bins=2)
    noise = np.random.default_rng(4).integers(0, 255, (120, 480, 3), np.uint8)

    evidence = bank.compare(noise, candidates=2, min_correlation=.99,
                            max_translation_ratio=.01, max_rotation_deg=.1)

    assert not evidence.registration.success
    assert evidence.intensity_score == 0
