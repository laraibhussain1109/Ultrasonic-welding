"""Backbone-output normalization independent of OpenCV/timm runtime imports."""

from __future__ import annotations

from typing import Any


def spatial_patch_tokens(output: Any, model: Any, grid: tuple[int, int]) -> Any:
    """Return only B×N×C spatial patch tokens, excluding global prefixes.

    timm DINO variants may return either a mapping with already-separated patch
    tokens or a tensor containing CLS/register tokens before the spatial grid.
    """
    if isinstance(output, dict):
        tokens = output.get("x_norm_patchtokens")
        if tokens is None:
            raise RuntimeError(
                "Configured ViT feature mapping has no x_norm_patchtokens output; "
                f"available keys: {sorted(output)}"
            )
    else:
        tokens = output
    if tokens.ndim == 4:
        tokens = tokens.flatten(2).transpose(1, 2)
    if tokens.ndim != 3:
        raise RuntimeError(
            f"Configured ViT does not expose BxNxC spatial tokens (shape={tuple(tokens.shape)})"
        )
    expected = grid[0] * grid[1]
    actual = int(tokens.shape[1])
    if actual > expected:
        prefix_count = actual - expected
        declared = int(getattr(model, "num_prefix_tokens", prefix_count))
        if declared > actual:
            raise RuntimeError("ViT reports more prefix tokens than its feature output contains")
        # Shape is authoritative because register-token variants can have more
        # leading global tokens than num_prefix_tokens reports.
        tokens = tokens[:, prefix_count:, :]
    if int(tokens.shape[1]) != expected:
        raise RuntimeError(
            "ViT spatial token count does not match its patch grid: "
            f"tokens={int(tokens.shape[1])}, expected={expected}, grid={grid}"
        )
    return tokens
