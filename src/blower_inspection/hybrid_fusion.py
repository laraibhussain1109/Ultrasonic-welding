"""Deterministic, auditable fusion of TAO, geometry, glare and registration."""

from dataclasses import dataclass, field

import numpy as np

from .geometry_inspector import GeometryEvidence


@dataclass(frozen=True)
class TaoEvidence:
    raw_map: np.ndarray
    calibrated_map: np.ndarray
    anomaly_score: float
    peak_score: float
    defect_mask: np.ndarray
    defect_area: int
    candidate_components: list[tuple[int, int, int, int]] = field(default_factory=list)
    bad_regions: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class HybridDecision:
    status: str
    hybrid_score: float
    reason_codes: tuple[str, ...]
    immediate_failure: bool = False
    provisional_candidate: bool = False
    confirmed_mask: np.ndarray | None = None


def fuse_evidence(tao: TaoEvidence, geometry: GeometryEvidence, *, glare_score: float,
                  registration_valid: bool, geometry_fail_threshold: float = 1.0,
                  geometry_candidate_threshold: float = .55, tao_strong_threshold: float = 1.0,
                  tao_candidate_threshold: float = .7, glare_threshold: float = .55) -> HybridDecision:
    if not registration_valid:
        return HybridDecision("VIEW INVALID", 0.0, ("REGISTRATION_INVALID",))
    if tao.defect_mask.shape != geometry.defect_mask.shape:
        raise ValueError(
            "TAO and geometry evidence masks must use the same fusion coordinate system: "
            f"{tao.defect_mask.shape} != {geometry.defect_mask.shape}"
        )
    g, t = geometry.score / max(geometry_fail_threshold, 1e-6), tao.anomaly_score / max(tao_strong_threshold, 1e-6)
    reasons: list[str] = []
    if geometry.broken_fin_score >= geometry_fail_threshold:
        reasons.append("BROKEN_FIN")
    if geometry.orientation_score >= geometry_fail_threshold:
        reasons.append("TILTED_FIN")
    if geometry.pitch_score >= geometry_fail_threshold:
        reasons.append("MISSING_FIN")
    catastrophic = g >= 1.0
    corroborated = geometry.score >= geometry_candidate_threshold and tao.anomaly_score >= tao_candidate_threshold
    likely_glare = glare_score >= glare_threshold and geometry.score < geometry_candidate_threshold and geometry.periodicity_score < geometry_candidate_threshold
    if catastrophic:
        reasons.append("GEOMETRY_DEFORMATION")
        return HybridDecision("FAIL", max(1.0, g, .7 * g + .5 * t), tuple(dict.fromkeys(reasons)), True, confirmed_mask=geometry.defect_mask)
    if corroborated:
        reasons.append("TAO_CONFIRMED_CHANGE")
        return HybridDecision("FAIL", max(1.0, .65 * t + .65 * g), tuple(reasons), True,
                              confirmed_mask=tao.defect_mask | geometry.defect_mask)
    if tao.anomaly_score >= tao_candidate_threshold:
        if likely_glare:
            return HybridDecision("PASS", min(.99, .45 * t), ("LIKELY_GLARE",))
        return HybridDecision("CANDIDATE", min(.99, .7 * t + .2 * g), ("PERSISTENT_VISUAL_CHANGE",), provisional_candidate=True,
                              confirmed_mask=tao.defect_mask)
    return HybridDecision("PASS", min(.99, max(.45 * t, .65 * g)), tuple(), confirmed_mask=np.zeros_like(tao.defect_mask))
