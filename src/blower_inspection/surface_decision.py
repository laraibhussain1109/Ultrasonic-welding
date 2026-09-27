"""Pure surface/geometry decision policy, isolated for deterministic testing."""

from __future__ import annotations


def surface_decision_state(*, local_score: float, candidate_line: float, strong_line: float,
                           memory_candidate: bool, reconstruction_candidate: bool,
                           geometry_score: float, geometry_support_threshold: float,
                           near_candidate_ratio: float, fine_break_candidate: bool,
                           fine_break_strong: bool) -> tuple[bool, bool, bool]:
    """Return candidate, immediate-strong, and geometry-corroborated states.

    A visible structural defect can sit just below a conservative learned-surface
    limit. Independent geometry evidence retains that view as a candidate instead
    of declaring it normal; multi-view persistence still controls final rejection.
    """
    # The fused local map is itself calibrated evidence. Do not require a
    # whole-tile branch percentile as an extra gate: that would recreate the
    # exact Q99.5 blind spot for a tiny but intense one-patch defect.
    direct = local_score >= candidate_line
    corroborated = (local_score >= candidate_line * near_candidate_ratio
                    and geometry_score >= geometry_support_threshold)
    candidate = direct or corroborated or fine_break_candidate
    strong = ((local_score >= strong_line and memory_candidate and reconstruction_candidate)
              or fine_break_strong)
    return candidate, strong, corroborated
