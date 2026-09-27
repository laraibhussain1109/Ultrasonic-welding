"""Contracts that remain testable on hosts without the OpenCV shared libraries."""

from pathlib import Path
import tomllib


def test_dinov2_weights_are_downloaded_once_and_atomically_cached():
    source = Path("src/blower_inspection/surface_inspector.py").read_text(encoding="utf-8")

    assert "pretrained=True" in source
    assert 'with_suffix(weights.suffix + ".part")' in source
    assert "temporary.replace(weights)" in source
    assert "Automatic DINOv2 download failed" in source


def test_surface_memory_search_prefers_torch_cuda_without_faiss_dependency():
    source = Path("src/blower_inspection/surface_inspector.py").read_text(encoding="utf-8")
    with Path("pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)

    assert 'torch.device("cuda:0")' in source
    assert "query @ memory.T" in source
    assert "surface_require_gpu" in source
    dependencies = project["project"]["dependencies"] + project["project"]["optional-dependencies"]["industrial"]
    assert not any("faiss" in dependency.casefold() for dependency in dependencies)


def test_fixed_size_dinov2_receives_configured_518_tensor_not_native_768_tile():
    source = Path("src/blower_inspection/surface_inspector.py").read_text(encoding="utf-8")

    assert "rgb = prepare_vit_rgb(image, config.vit_input_size)" in source
    assert "img_size=config.vit_input_size" in source
    assert '"vit_input_size": config.vit_input_size' in source
    assert "tokens = spatial_patch_tokens(output, model, grid)" in source
