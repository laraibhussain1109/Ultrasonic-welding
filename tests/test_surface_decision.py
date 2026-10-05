from blower_inspection.surface_decision import production_status, surface_decision_state


def test_strong_learned_appearance_alone_requires_persistence():
    assert production_status(appearance_candidate=True, appearance_strong=True,
                             golden_candidate=False, golden_structural_strong=False,
                             geometry_fail=False, fine_break_strong=False) == "CANDIDATE"


def test_native_structural_break_fails_immediately():
    assert production_status(appearance_candidate=False, appearance_strong=False,
                             golden_candidate=False, golden_structural_strong=False,
                             geometry_fail=False, fine_break_strong=True) == "FAIL"


def decide(**overrides):
    values = {
        "local_score": 0.0,
        "candidate_line": 1.0,
        "strong_line": 1.25,
        "memory_candidate": False,
        "reconstruction_candidate": False,
        "geometry_score": 0.0,
        "geometry_support_threshold": 0.15,
        "near_candidate_ratio": 0.85,
        "localized_near_candidate": False,
        "localized_candidate": True,
        "fine_break_candidate": False,
        "fine_break_strong": False,
    }
    values.update(overrides)
    return surface_decision_state(**values)


def test_visible_near_threshold_surface_change_with_geometry_is_not_normal():
    candidate, strong, corroborated = decide(local_score=.912, candidate_line=.962,
                                              geometry_score=.21,
                                              localized_near_candidate=True)

    assert candidate
    assert corroborated
    assert not strong


def test_low_surface_noise_with_same_geometry_remains_normal():
    candidate, strong, corroborated = decide(local_score=.40, candidate_line=.962,
                                              geometry_score=.21)

    assert not candidate
    assert not corroborated
    assert not strong


def test_broad_near_threshold_change_is_not_geometry_corroborated():
    candidate, strong, corroborated = decide(local_score=.90, candidate_line=.962,
                                              geometry_score=.16,
                                              localized_near_candidate=False)

    assert not candidate
    assert not corroborated
    assert not strong


def test_broad_above_threshold_appearance_is_not_a_defect_candidate():
    candidate, strong, corroborated = decide(local_score=1.8,
                                              memory_candidate=True,
                                              reconstruction_candidate=True,
                                              localized_candidate=False)

    assert not candidate
    assert strong  # retained in diagnostics, but production_status will not fail it
    assert not corroborated


def test_extreme_local_patch_does_not_need_whole_tile_percentile_gate():
    candidate, strong, corroborated = decide(local_score=1.05)

    assert candidate
    assert not strong
    assert not corroborated


def test_native_fine_break_can_candidate_or_immediately_fail():
    assert decide(fine_break_candidate=True) == (True, False, False)
    assert decide(fine_break_candidate=True, fine_break_strong=True) == (True, True, False)
