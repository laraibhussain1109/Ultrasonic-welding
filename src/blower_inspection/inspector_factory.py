"""Inspection backend selection shared by the desktop and command-line apps."""

from __future__ import annotations

from typing import Any

from .config import PartModelConfig
from .tao_inspector import TaoInspector


def phase_locked_components(config: PartModelConfig):
    """Build the production engine components without hiding PLC phase input."""
    from .phase_locked import GoldenBank, PhaseLockedConfig, PhaseLockedStructuralInspector
    from .plc_inspection import PhaseInspectionController

    settings = PhaseLockedConfig(
        inspection_angles=config.inspection_angles,
        settle_delay_ms=config.settle_delay_ms,
        burst_frame_count=config.burst_frame_count,
        minimum_sharpness=config.minimum_sharpness,
        minimum_qualified_frames=config.minimum_qualified_frames,
        temporal_confirmation_frames=config.temporal_confirmation_frames,
        registration_max_translation_ratio=config.registration_max_translation_ratio,
        registration_max_rotation_deg=config.registration_max_rotation_deg,
        registration_min_correlation=config.registration_min_correlation,
        fitment_runout_threshold=config.fitment_runout_threshold,
        fitment_axial_threshold=config.fitment_axial_threshold,
        golden_noise_floor=config.golden_noise_floor,
        golden_candidate_threshold=config.golden_candidate_threshold,
        golden_fail_threshold=config.golden_fail_threshold,
        edge_candidate_threshold=config.edge_candidate_threshold,
        edge_fail_threshold=config.edge_fail_threshold,
        geometry_candidate_threshold=config.geometry_candidate_threshold,
        geometry_fail_threshold=config.geometry_fail_threshold,
        glare_rejection_enabled=config.glare_rejection_enabled,
        efficientad_enabled=config.efficientad_enabled,
    )
    engine = PhaseLockedStructuralInspector(GoldenBank(config.golden_model_root, config.id), settings)
    return engine, PhaseInspectionController(engine, settings, config.result_dir / "phase_locked_sessions")


def inspector_for_model(config: PartModelConfig) -> Any:
    """Create the inspection backend configured for a part model."""
    production_algorithm = config.production_algorithm.strip().casefold()
    algorithm = config.algorithm.strip().casefold()
    if production_algorithm == "phase_locked_structural":
        return phase_locked_components(config)[0]
    if production_algorithm == "patchcore_geometry" or algorithm in {
        "patchcore_primary", "hybrid_patchcore_geometry", "patchcore_geometry"
    }:
        from .patchcore_inspector import PatchCoreInspector

        return PatchCoreInspector()
    if algorithm == "nvidia_tao":
        return TaoInspector()
    if algorithm == "hybrid_patchcore_padim":
        from .anomaly_models import HybridPatchcorePadimInspector

        return HybridPatchcorePadimInspector()
    raise ValueError(
        f"Unsupported inspection algorithm: {config.algorithm!r}. "
        "This installation may be loading an older checkout. Run "
        "`blower-inspection doctor` and reinstall this repository with "
        "`python -m pip install -e .` from its root."
    )
