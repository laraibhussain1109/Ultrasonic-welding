"""Inspection backend selection shared by the desktop and command-line apps."""

from __future__ import annotations

from typing import Any

from .config import PartModelConfig
from .tao_inspector import TaoInspector


def inspector_for_model(config: PartModelConfig) -> Any:
    """Create the inspection backend configured for a part model."""
    production_algorithm = config.production_algorithm.strip().casefold()
    algorithm = config.algorithm.strip().casefold()
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
