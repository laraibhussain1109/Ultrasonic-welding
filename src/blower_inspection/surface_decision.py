"""Pure surface/geometry decision policy, isolated for deterministic testing."""

from __future__ import annotations


def surface_decision_state(*, local_score: float, candidate_line: float, strong_line: float,
                           memory_candidate: bool, reconstruction_candidate: bool,
                           geometry_score: float, geometry_support_threshold: float,
                           near_candidate_ratio: float, localized_near_candidate: bool,
                           localized_candidate: bool,
                           fine_break_candidate: bool,
                           fine_break_strong: bool) -> tuple[bool, bool, bool]:
    """Return candidate, immediate-strong, and geometry-corroborated states.

    A visible structural defect can sit just below a conservative learned-surface
    limit. Independent geometry evidence retains that view as a candidate instead
    of declaring it normal; multi-view persistence still controls final rejection.
    """
    # The fused local map is itself calibrated evidence. Do not require a
    # whole-tile branch percentile as an extra gate: that would recreate the
    # exact Q99.5 blind spot for a tiny but intense one-patch defect.
    # A broad response spanning much of the blower is normally illumination,
    # batch appearance, or residual registration error.  It remains visible in
    # diagnostics but is not eligible for temporal defect confirmation.  A
    # truly small chip/edge break produces a localized component and therefore
    # keeps full authority even when it contains very few pixels.
    direct = local_score >= candidate_line and localized_candidate
    corroborated = (localized_near_candidate
                    and local_score >= candidate_line * near_candidate_ratio
                    and geometry_score >= geometry_support_threshold)
    candidate = direct or corroborated or fine_break_candidate
    strong = ((local_score >= strong_line and memory_candidate and reconstruction_candidate)
              or fine_break_strong)
    return candidate, strong, corroborated


def production_status(*, appearance_candidate: bool, appearance_strong: bool,
                      golden_candidate: bool, golden_structural_strong: bool,
                      geometry_fail: bool, fine_break_strong: bool) -> str:
    """Fuse evidence without allowing appearance certainty to masquerade as structure.

    ``appearance_strong`` is intentionally accepted for diagnostics but is not
    an immediate-fail gate.  Batch/texture changes can strongly activate two
    learned branches at once; location-aware persistence must confirm those.
    """
    del appearance_strong
    if geometry_fail or fine_break_strong or golden_structural_strong:
        return "FAIL"
    if appearance_candidate or golden_candidate:
        return "CANDIDATE"
    return "PASS"
