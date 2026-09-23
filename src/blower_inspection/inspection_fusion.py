"""Auditable PatchCore/geometry production decision rules."""

from dataclasses import dataclass

import numpy as np

from .geometry_inspector import GeometryEvidence


@dataclass(frozen=True)
class ProductionDecision:
    status: str
    reason_codes: tuple[str, ...]
    immediate_failure: bool
    provisional_candidate: bool
    confirmed_mask: np.ndarray


def fuse_patchcore_geometry(patchcore_score: float, patchcore_mask: np.ndarray,
                            geometry: GeometryEvidence, *, glare_score: float,
                            candidate_threshold: float, fail_threshold: float,
                            geometry_candidate_threshold: float = .55,
                            geometry_fail_threshold: float = 1.0,
                            glare_threshold: float = .55) -> ProductionDecision:
    reasons: list[str] = []
    if geometry.broken_fin_score >= geometry_fail_threshold: reasons.append("BROKEN_FIN")
    if geometry.missing_fin_score >= geometry_fail_threshold: reasons.append("MISSING_FIN")
    if geometry.tilted_fin_score >= geometry_fail_threshold: reasons.append("TILTED_FIN")
    if geometry.score >= geometry_fail_threshold:
        reasons.append("GEOMETRY_DEFORMATION")
        return ProductionDecision("FAIL", tuple(reasons), True, False, geometry.defect_mask)
    patch_candidate = patchcore_score >= candidate_threshold
    geometry_candidate = geometry.score >= geometry_candidate_threshold
    if patch_candidate and geometry_candidate:
        return ProductionDecision("FAIL", ("PATCHCORE_ANOMALY", "GEOMETRY_DEFORMATION"), True, False,
                                  patchcore_mask | geometry.defect_mask)
    # The calibrated candidate line is the start of anomalous PatchCore
    # evidence.  Do not require the score to reach the immediate-failure line
    # before forwarding that evidence to the multi-view persistence tracker.
    # Doing so silently turned every score in [candidate, fail) into PASS --
    # exactly the range where a visible defect should be confirmed by a second
    # rotational view rather than ignored.
    if patch_candidate and glare_score < glare_threshold:
        return ProductionDecision("CANDIDATE", ("PATCHCORE_ANOMALY",), False, True, patchcore_mask)
    if patch_candidate and glare_score >= glare_threshold:
        return ProductionDecision("PASS", ("LIKELY_GLARE",), False, False, np.zeros_like(patchcore_mask))
    return ProductionDecision("PASS", (), False, False, np.zeros_like(patchcore_mask))
