"""Bound training RAM without changing native quality measurements or sampling."""

import tracemalloc
import json
import weakref
from dataclasses import replace

import cv2
import numpy as np
import pytest
import torch
from torchvision import models

from blower_inspection.anomaly_models import HybridPatchcorePadimInspector, HybridTrainingSettings, _FeatureHook
from blower_inspection.config import PartModelConfig
from blower_inspection.frame_quality import FrameQualityAnalyzer
from blower_inspection.patchcore_inspector import PatchCoreInspector
from blower_inspection.roi_stabilizer import CanonicalROI
from blower_inspection.training_cache import DiskPatchEmbeddings


def _original_quality_metrics(image, mask):
    """The full-resolution calculations used before the allocation fix."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask = np.ones(gray.shape, bool) if mask is None else mask
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    qualified = eroded if np.any(eroded) else mask
    lap = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)[qualified]))
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    tenengrad = float(np.mean((gx * gx + gy * gy)[qualified]))
    local_std = np.sqrt(np.maximum(cv2.blur(gray.astype(np.float32) ** 2, (15, 15)) -
                                  cv2.blur(gray.astype(np.float32), (15, 15)) ** 2, 0))
    return (lap, tenengrad, float(np.mean((gray >= 250)[mask])),
            float(np.mean((gray <= 5)[mask])), float(np.mean(((gray >= 225) & (local_std < 18))[mask])))


@pytest.mark.parametrize("shape", [(1, 19), (37, 53), (529, 193)])
@pytest.mark.parametrize("mask_kind", ["none", "letterbox", "sparse"])
@pytest.mark.parametrize("color", [False, True])
def test_quality_matches_native_resolution_reference_across_strip_seams(shape, mask_kind, color):
    rng = np.random.default_rng(193)
    image = rng.integers(0, 256, (*shape, 3) if color else shape, dtype=np.uint8)
    image[:, :shape[1] // 4] = 235  # Smooth reflection, alongside texture and dark pixels.
    mask = None
    if mask_kind != "none":
        mask = np.zeros(shape, bool)
        if mask_kind == "letterbox":
            mask[shape[0] // 4:max(1, shape[0] * 3 // 4), 2:-2] = True
        else:
            mask[::3, ::3] = True  # Exercises erosion's empty-mask fallback.
    expected = _original_quality_metrics(image, mask)
    result = FrameQualityAnalyzer().analyze(image, mask)
    actual = (result.sharpness, result.tenengrad, result.saturation_ratio, result.dark_ratio, result.glare_ratio)
    assert actual == pytest.approx(expected, rel=3e-7, abs=1e-9)
    reasons = []
    if expected[0] < 60:
        reasons.extend(("MOTION_BLUR", "LOW_SHARPNESS"))
    for value, threshold, reason in zip(expected[2:], (.12, .55, .20),
                                        ("OVEREXPOSED", "UNDEREXPOSED", "EXCESSIVE_GLARE")):
        if value > threshold:
            reasons.append(reason)
    assert result.reasons == tuple(reasons)
    assert result.valid == (not reasons)


@pytest.mark.parametrize("masked", [False, True])
def test_4k_quality_does_not_allocate_full_image_float64_temporaries(masked):
    image = np.random.default_rng(42).integers(30, 220, (2160, 3840, 3), dtype=np.uint8)
    mask = np.ones(image.shape[:2], bool) if masked else None
    if mask is not None:
        mask[:80] = False
    tracemalloc.start()
    try:
        result = FrameQualityAnalyzer().analyze(image, mask)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.valid
    # The old Laplacian selection alone allocated 63.3 MiB, before variance,
    # gradients, glare, masks, or the retained training photos were accounted for.
    assert peak < 64 * 1024 * 1024


@pytest.mark.parametrize("mask", [np.zeros((19, 31), bool), np.ones((19, 30), bool)])
def test_invalid_quality_masks_still_raise(mask):
    with pytest.raises(ValueError, match="valid_mask"):
        FrameQualityAnalyzer().analyze(np.zeros((19, 31), np.uint8), mask)


@pytest.mark.parametrize("count", [13, 67])
def test_disk_embeddings_produce_identical_seeded_coreset(tmp_path, count):
    settings = replace(HybridTrainingSettings(), coreset_candidate_patches=20,
                       max_coreset_patches=8, coreset_ratio=.2)
    inspector = HybridPatchcorePadimInspector(settings, device="cpu")
    rows = torch.from_numpy(np.random.default_rng(42).normal(size=(count, 7)).astype(np.float32))
    expected, expected_count = inspector._build_patchcore_memory(rows, torch)
    path = tmp_path / "patches.bin"
    with DiskPatchEmbeddings(path) as cache:
        for chunk in rows.split(6):
            cache.append(chunk)
        actual, actual_count = cache.build_memory(inspector, torch)
    assert torch.equal(actual, expected)
    assert actual_count == expected_count
    # The returned bank owns its storage after the Windows-compatible mapping
    # and writer are closed; removing the cache cannot invalidate the checkpoint.
    path.unlink()
    assert torch.equal(actual, expected)


def test_disk_embeddings_close_on_failure(tmp_path):
    path = tmp_path / "patches.bin"
    with pytest.raises(ValueError, match="dimensions"):
        with DiskPatchEmbeddings(path) as cache:
            cache.append(torch.zeros(3, 7))
            cache.append(torch.zeros(3, 9))
    assert cache._handle.closed
    path.unlink()


def test_production_training_limits_loaded_crops_and_preserves_checkpoint_contract(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    normal = tmp_path / "normal"
    normal.mkdir()
    rng = np.random.default_rng(42)
    for index in range(48):
        image = (rng.integers(30, 220, (96, 192, 3), dtype=np.uint8) if index < 44
                 else cv2.imread(str(normal / f"good_{index - 44:03d}.png")))
        assert cv2.imwrite(str(normal / f"good_{index:03d}.png"), image)
    config = PartModelConfig("test", "test", normal, tmp_path / "model.pt", tmp_path / "results", 12,
                             yolo_model_path=tmp_path / "yolo.pt", image_size=64,
                             patchcore_section_count=2, patchcore_memory_bank_size=8)
    settings = replace(HybridTrainingSettings(), backbone="resnet18", max_training_images=20,
                       embedding_grid_size=8, coreset_candidate_patches=32)
    model = PatchCoreInspector(settings, device="cpu")
    backbone = models.resnet18(weights=None).eval()
    hook = _FeatureHook(config.patchcore_embedding_layers)
    hook.attach(backbone)
    model._backbone_cache = (backbone, hook, torch)
    crops, live_counts = [], []

    def canonical(frame, _config, _detector):
        crop = cv2.resize(frame, (64, 32), interpolation=cv2.INTER_AREA)
        crops.append(weakref.ref(crop))
        live_counts.append(sum(reference() is not None for reference in crops))
        return CanonicalROI(crop, (0, 0, 192, 96), (0, 0, 64, 32))

    monkeypatch.setattr(model, "_canonical", canonical)
    path = model.train(config)
    assert max(live_counts) <= 2
    assert not any(reference() is not None for reference in crops)
    report = json.loads(path.with_suffix(".training_report.json").read_text())
    assert report["total_source_images"] == 48
    assert report["accepted_images"] == 20
    assert report["nonduplicate_images"] == 44
    assert report["duplicates_removed"] == 4
    assert report["training_limit_removed"] == 24
    assert report["training_images"] == 16
    assert report["calibration_images"] == 4
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["algorithm"] == "patchcore_primary"
    assert checkpoint["version"] == 3
    assert len(checkpoint["memory_bank"]) <= 8
    assert len(checkpoint["geometry_calibrations"]) == 2
    model.validate_ready(config)
