"""Inspection backend selection shared by the desktop and command-line apps."""

from __future__ import annotations

from typing import Any

from .config import PartModelConfig
from .tao_inspector import TaoInspector


def inspector_for_model(config: PartModelConfig) -> Any:
    """Create the inspection backend configured for a part model."""
    if config.algorithm == "nvidia_tao":
        return TaoInspector()
    if config.algorithm == "hybrid_patchcore_padim":
        from .anomaly_models import HybridPatchcorePadimInspector

        return HybridPatchcorePadimInspector()
    raise ValueError(f"Unsupported inspection algorithm: {config.algorithm}")
