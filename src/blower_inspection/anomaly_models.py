"""Industrial anomaly models for blower fan inspection.

This module implements a production-oriented hybrid of PatchCore and PaDiM:

- PatchCore keeps a coreset memory bank of normal deep patch embeddings and uses
  nearest-neighbour distance at inspection time.
- PaDiM fits a per-patch Gaussian distribution over normal embeddings and uses
  Mahalanobis distance at inspection time.
- Hybrid inference normalizes and fuses both score maps, then applies the same
  fin/weld ROI and sector logic used by the rest of the application.

The implementation intentionally loads heavyweight AI dependencies lazily inside
methods. This lets authentication/configuration tests run on machines that do not
have the GPU stack installed, while deployment environments can install the full
`industrial` extra from `pyproject.toml`.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import json
import math
import os
import pickle
import random
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .camera import crop_component_roi
from .config import PartModelConfig
from .trainer import (
    InspectionResult,
    clean_mask,
    cylindrical_sector_statistics,
    cylindrical_surface_mask,
    inspection_overlay,
    list_images,
)
from .yolo_tracking import YoloByteTrackDetector

HYBRID_MODEL_VERSION = 3


@dataclass(frozen=True)
class HybridTrainingSettings:
    image_size: int = 512
    backbone: str = "wide_resnet50_2"
    embedding_layers: tuple[str, ...] = ("layer1", "layer2", "layer3")
    embedding_grid_size: int = 56
    projection_dim: int = 256
    max_training_images: int = 300
    coreset_ratio: float = 0.08
    max_coreset_patches: int = 4096
    coreset_candidate_patches: int = 80000
    padim_components: int = 128
    patchcore_weight: float = 0.55
    padim_weight: float = 0.45
    batch_size: int = 4
    random_seed: int = 42
    runtime_memory_bank_limit: int = 512


def require_module(module_name: str) -> Any:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise RuntimeError(
            f"Missing dependency '{module_name}'. Install the industrial stack with "
            "`pip install -e .[industrial]` on the deployment PC."
        )
    return importlib.import_module(module_name)


def robust_normalize(score_map: np.ndarray) -> np.ndarray:
    score = score_map.astype(np.float32)
    lo = float(np.percentile(score, 1.0))
    hi = float(np.percentile(score, 99.5))
    if hi <= lo + 1e-6:
        return np.zeros_like(score, dtype=np.float32)
    return np.clip((score - lo) / (hi - lo), 0.0, 1.0)


class _FeatureHook:
    def __init__(self, layers: tuple[str, ...]) -> None:
        self.layers = layers
        self.features: dict[str, Any] = {}
        self.handles: list[Any] = []

    def attach(self, model: Any) -> None:
        modules = dict(model.named_modules())
        for layer in self.layers:
            if layer not in modules:
                raise KeyError(f"Backbone layer '{layer}' not found")
            self.handles.append(modules[layer].register_forward_hook(self._capture(layer)))

    def clear(self) -> None:
        self.features.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _capture(self, layer: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            self.features[layer] = output.detach()
        return hook


class HybridPatchcorePadimInspector:
    """Train and inspect with a PatchCore + PaDiM hybrid model."""

    def __init__(self, settings: HybridTrainingSettings | None = None, device: str | None = None) -> None:
        self.settings = settings or HybridTrainingSettings()
        self.device = device or os.environ.get("BLOWER_INSPECTION_DEVICE")
        self._device_cache: Any | None = None
        self._backbone_cache: tuple[Any, _FeatureHook, Any] | None = None
        self._checkpoint_cache: dict[Path, tuple[float, dict[str, Any]]] = {}
        self._training_roi_ratios: tuple[float, float, float, float] | None = None
        self._training_detector: YoloByteTrackDetector | None = None

    def train(self, config: PartModelConfig) -> Path:
        self._apply_model_settings(config)
        image_paths = list_images(config.normal_image_dir)
        if len(image_paths) < 20:
            raise ValueError(
                f"Industrial hybrid training needs at least 20 normal images in {config.normal_image_dir}; "
                f"found {len(image_paths)}. Use 100+ per model for production validation."
            )
        used_image_paths = self._select_training_images(image_paths)
        torch = require_module("torch")
        self._training_roi_ratios = config.roi_ratios
        self._training_detector = (
            YoloByteTrackDetector(config.yolo_model_path, config.yolo_confidence)
            if config.yolo_model_path is not None
            else None
        )
        embeddings, grid_shape = self._extract_dataset_embeddings(used_image_paths)
        embeddings_np = embeddings.cpu().numpy().astype(np.float32)
        rng = np.random.default_rng(self.settings.random_seed)
        if embeddings_np.shape[1] > self.settings.padim_components:
            selected_dims = np.sort(rng.choice(embeddings_np.shape[1], self.settings.padim_components, replace=False))
        else:
            selected_dims = np.arange(embeddings_np.shape[1])
        memory_bank, memory_candidate_count = self._build_patchcore_memory(embeddings, torch)
        padim_mean, padim_inv_cov = self._fit_padim(embeddings_np, grid_shape, selected_dims)
        config.model_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": HYBRID_MODEL_VERSION,
                "algorithm": "hybrid_patchcore_padim",
                "settings": asdict(self.settings),
                "model_config": self._serializable_model_config(config),
                "trained_at": datetime.now().isoformat(),
                "source_count": len(image_paths),
                "used_source_count": len(used_image_paths),
                "grid_shape": grid_shape,
                "feature_dim": embeddings_np.shape[1],
                "padim_dims": selected_dims.tolist(),
                "patch_count": int(embeddings_np.shape[0]),
                "memory_candidate_count": int(memory_candidate_count),
                "memory_bank_size": int(memory_bank.shape[0]),
                "memory_bank": memory_bank.cpu(),
                "padim_mean": torch.as_tensor(padim_mean, dtype=torch.float32),
                "padim_inv_cov": torch.as_tensor(padim_inv_cov, dtype=torch.float32),
            },
            config.model_file,
        )
        return config.model_file

    def inspect(
        self,
        config: PartModelConfig,
        image: np.ndarray,
        *,
        save_outputs: bool = True,
        crop_to_component: bool = True,
    ) -> InspectionResult:
        torch = require_module("torch")
        if crop_to_component:
            image = crop_component_roi(image, roi_ratios=config.roi_ratios)
        if not config.model_file.exists():
            raise FileNotFoundError(f"Hybrid model has not been trained: {config.model_file}")
        checkpoint = self._load_runtime_checkpoint(torch, config.model_file)
        if checkpoint.get("algorithm") != "hybrid_patchcore_padim":
            raise ValueError(f"Model file is not a hybrid PatchCore/PaDiM checkpoint: {config.model_file}")
        self._apply_checkpoint_settings(checkpoint)
        tensor = self._preprocess_image(image)
        features, _grid_shape = self._extract_embeddings_from_tensor(tensor)
        patchcore_map = self._patchcore_score_map(features, checkpoint["memory_bank"], torch)
        padim_dims = self._checkpoint_dims(checkpoint["padim_dims"])
        padim_map = self._padim_score_map_torch(
            features.squeeze(0),
            checkpoint["padim_mean"],
            checkpoint["padim_inv_cov"],
            padim_dims,
            torch,
        )
        fused_small = (
            self.settings.patchcore_weight * robust_normalize(patchcore_map)
            + self.settings.padim_weight * robust_normalize(padim_map)
        )
        fused = cv2.resize(fused_small, (self.settings.image_size, self.settings.image_size), interpolation=cv2.INTER_CUBIC)
        fused = cv2.GaussianBlur(fused, (9, 9), 0)
        surface_mask = cylindrical_surface_mask(fused.shape)
        # Operators judge the overlay by color: blue/green should remain PASS,
        # while FAIL should require the yellow/red-to-red severity band.  The
        # hybrid map is normalized to 0..1, so keep the configurable threshold
        # but never allow it below the visual yellow/red floor.
        threshold = min(max(config.anomaly_threshold / 10.0, 0.65), 0.95)
        candidate_mask = clean_mask((fused >= threshold) & surface_mask)
        defect_area = int(candidate_mask.sum())
        surface_scores = fused[surface_mask]
        anomaly_score = float(np.max(surface_scores)) if surface_scores.size else 0.0
        bad_sector_ratio, bad_sectors = cylindrical_sector_statistics(
            surface_mask, fused, config.expected_fins, min_bad_score=threshold
        )
        fail_area = max(config.min_defect_area_px, int(0.0004 * fused.size))
        status = "FAIL" if defect_area >= fail_area or bad_sector_ratio >= config.max_bad_sector_ratio else "PASS"
        display_image, defect_boxes = inspection_overlay(
            image,
            fused,
            candidate_mask,
            status=status,
            anomaly_score=anomaly_score,
            min_box_area_px=config.min_defect_area_px,
            score_normalizer=threshold,
        )
        overlay_path = report_path = None
        if save_outputs:
            overlay_path, report_path = self._save_outputs(
                config, image, fused, candidate_mask, status, anomaly_score, defect_area, bad_sector_ratio, bad_sectors
            )
        return InspectionResult(
            status,
            anomaly_score,
            defect_area,
            bad_sector_ratio,
            bad_sectors,
            overlay_path,
            report_path,
            display_image,
            defect_boxes,
        )

    @staticmethod
    def _serializable_model_config(config: PartModelConfig) -> dict[str, Any]:
        """Return checkpoint metadata without pickled ``Path`` objects.

        PyTorch 2.6 defaults ``torch.load`` to ``weights_only=True``. Keeping
        checkpoint metadata to primitives and tensors lets newly trained models
        load with the safer default path instead of requiring arbitrary pickle
        globals such as ``pathlib.WindowsPath``.
        """
        data = asdict(config)
        for key in ("normal_image_dir", "model_file", "result_dir", "yolo_model_path"):
            if data[key] is not None:
                data[key] = str(data[key])
        return data

    @staticmethod
    def _load_hybrid_checkpoint(torch: Any, model_file: Path, map_location: Any = "cpu") -> dict[str, Any]:
        """Load app-created hybrid checkpoints across PyTorch versions.

        New checkpoints are saved with only tensors and primitive metadata and
        therefore load with ``weights_only=True``. Older checkpoints may contain
        dataclass metadata with ``pathlib.WindowsPath``/``Path`` objects, which
        PyTorch 2.6 rejects in weights-only mode. For those legacy files we
        retry with ``weights_only=False`` so operators can keep using models
        trained by previous versions of this application.
        """
        try:
            return torch.load(model_file, map_location=map_location, weights_only=True)
        except TypeError:
            return torch.load(model_file, map_location=map_location)
        except pickle.UnpicklingError as exc:
            if "weights_only" not in str(exc).lower():
                raise
            return torch.load(model_file, map_location=map_location, weights_only=False)

    @staticmethod
    def _checkpoint_array(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def _checkpoint_dims(value: Any) -> list[int]:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return [int(dim) for dim in np.asarray(value).tolist()]

    def _device(self, torch: Any) -> Any:
        if self._device_cache is not None:
            return self._device_cache
        requested = (self.device or "").lower()
        if requested in {"dml", "directml"}:
            torch_directml = require_module("torch_directml")
            self._device_cache = torch_directml.device()
        elif self.device:
            self._device_cache = torch.device(self.device)
        elif torch.cuda.is_available():
            self._device_cache = torch.device("cuda")
        else:
            self._device_cache = torch.device("cpu")
        if getattr(self._device_cache, "type", None) == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return self._device_cache

    def runtime_device_name(self) -> str:
        """Return the active inference device for operator diagnostics."""
        torch = require_module("torch")
        device = self._device(torch)
        if getattr(device, "type", None) == "cuda":
            index = getattr(device, "index", None)
            return f"cuda:{index or 0} ({torch.cuda.get_device_name(index or 0)})"
        return str(device)

    def _apply_model_settings(self, config: PartModelConfig) -> None:
        if config.image_size != self.settings.image_size:
            self.settings = replace(self.settings, image_size=config.image_size)

    def _apply_checkpoint_settings(self, checkpoint: dict[str, Any]) -> None:
        saved = checkpoint.get("settings")
        if not isinstance(saved, dict):
            return
        current = asdict(self.settings)
        next_values = {key: saved.get(key, value) for key, value in current.items()}
        next_settings = HybridTrainingSettings(**next_values)
        rebuild_backbone = (
            next_settings.backbone != self.settings.backbone
            or next_settings.embedding_layers != self.settings.embedding_layers
        )
        self.settings = next_settings
        if rebuild_backbone:
            self._backbone_cache = None

    def _build_backbone(self) -> tuple[Any, _FeatureHook, Any]:
        if self._backbone_cache is not None:
            return self._backbone_cache
        torch = require_module("torch")
        models = require_module("torchvision.models")
        weights_name = "Wide_ResNet50_2_Weights"
        if self.settings.backbone == "resnet18":
            weights_name = "ResNet18_Weights"
        weights = getattr(models, weights_name).DEFAULT
        backbone = getattr(models, self.settings.backbone)(weights=weights)
        backbone.eval().to(self._device(torch))
        hook = _FeatureHook(self.settings.embedding_layers)
        hook.attach(backbone)
        self._backbone_cache = (backbone, hook, torch)
        return self._backbone_cache

    def _load_runtime_checkpoint(self, torch: Any, model_file: Path) -> dict[str, Any]:
        stamp = model_file.stat().st_mtime
        cached = self._checkpoint_cache.get(model_file)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        checkpoint = self._load_hybrid_checkpoint(torch, model_file, map_location=self._device(torch))
        for key in ("memory_bank", "padim_mean", "padim_inv_cov"):
            value = checkpoint.get(key)
            if hasattr(value, "to"):
                checkpoint[key] = value.to(self._device(torch), non_blocking=True).float().contiguous()
        memory_bank = checkpoint.get("memory_bank")
        if hasattr(memory_bank, "shape") and memory_bank.shape[0] > self.settings.runtime_memory_bank_limit:
            indices = torch.linspace(
                0,
                memory_bank.shape[0] - 1,
                steps=self.settings.runtime_memory_bank_limit,
                device=memory_bank.device,
            ).long()
            checkpoint["memory_bank"] = memory_bank.index_select(0, indices).contiguous()
            checkpoint["runtime_memory_bank_size"] = int(checkpoint["memory_bank"].shape[0])
        self._checkpoint_cache[model_file] = (stamp, checkpoint)
        return checkpoint

    def _extract_dataset_embeddings(self, image_paths: list[Path]) -> tuple[Any, tuple[int, int]]:
        torch = require_module("torch")
        backbone, hook, _torch = self._build_backbone()
        chunks: list[Any] = []
        grid_shape = (0, 0)
        for start in range(0, len(image_paths), self.settings.batch_size):
            batch_paths = image_paths[start:start + self.settings.batch_size]
            batch = torch.cat([self._preprocess_path(path) for path in batch_paths], dim=0)
            batch_features, grid_shape = self._extract_embeddings_with_backbone(batch, backbone, hook, torch)
            chunks.append(batch_features.flatten(2).permute(0, 2, 1).reshape(-1, batch_features.shape[1]).cpu())
        return torch.cat(chunks, dim=0), grid_shape

    def _extract_embeddings_from_tensor(self, tensor: Any) -> tuple[Any, tuple[int, int]]:
        torch = require_module("torch")
        backbone, hook, _torch = self._build_backbone()
        return self._extract_embeddings_with_backbone(tensor, backbone, hook, torch)

    def _extract_embeddings_with_backbone(self, tensor: Any, backbone: Any, hook: _FeatureHook, torch: Any) -> tuple[Any, tuple[int, int]]:
        device = self._device(torch)
        autocast = (
            torch.autocast(device_type="cuda")
            if getattr(device, "type", None) == "cuda"
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast:
            hook.clear()
            backbone(tensor.to(device, non_blocking=True))
            maps = []
            target_hw = None
            for layer in self.settings.embedding_layers:
                fmap = hook.features[layer]
                if target_hw is None:
                    target_hw = fmap.shape[-2:]
                elif fmap.shape[-2:] != target_hw:
                    fmap = torch.nn.functional.interpolate(fmap, size=target_hw, mode="bilinear", align_corners=False)
                maps.append(fmap)
            embedding = torch.cat(maps, dim=1)
            embedding = torch.nn.functional.adaptive_avg_pool2d(
                embedding,
                (self.settings.embedding_grid_size, self.settings.embedding_grid_size),
            )
            embedding = self._project_embedding(embedding.float(), torch)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return embedding.contiguous(), tuple(int(v) for v in embedding.shape[-2:])

    def _select_training_images(self, image_paths: list[Path]) -> list[Path]:
        if len(image_paths) <= self.settings.max_training_images:
            return image_paths
        rng = random.Random(self.settings.random_seed)
        selected = sorted(rng.sample(image_paths, self.settings.max_training_images))
        return selected

    def _projection_matrix(self, in_channels: int, torch: Any) -> Any:
        out_channels = min(self.settings.projection_dim, in_channels)
        if out_channels == in_channels:
            return torch.eye(in_channels, dtype=torch.float32)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.settings.random_seed + in_channels + out_channels)
        matrix = torch.randn(out_channels, in_channels, generator=generator, dtype=torch.float32)
        matrix = matrix / torch.sqrt(torch.tensor(float(out_channels), dtype=torch.float32))
        return matrix

    def _project_embedding(self, embedding: Any, torch: Any) -> Any:
        matrix = self._projection_matrix(embedding.shape[1], torch).to(embedding.device)
        return torch.einsum("oc,bchw->bohw", matrix, embedding)

    def _preprocess_path(self, path: Path) -> Any:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unable to read training image: {path}")
        if self._training_detector is not None:
            image = self._training_detector.exact_crop(image)
        else:
            image = crop_component_roi(image, roi_ratios=self._training_roi_ratios)
        return self._preprocess_image(image)

    def _preprocess_image(self, image: np.ndarray) -> Any:
        torch = require_module("torch")
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.settings.image_size, self.settings.image_size), interpolation=cv2.INTER_AREA)
        arr = resized.astype(np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor.pin_memory() if torch.cuda.is_available() else tensor

    def _build_patchcore_memory(self, embeddings: Any, torch: Any) -> tuple[Any, int]:
        total = embeddings.shape[0]
        target = min(max(1, int(total * self.settings.coreset_ratio)), self.settings.max_coreset_patches, total)
        candidate_count = min(total, max(target, self.settings.coreset_candidate_patches))
        rng = random.Random(self.settings.random_seed)
        if candidate_count < total:
            candidate_indices = torch.tensor(rng.sample(range(total), candidate_count), dtype=torch.long)
            candidates = embeddings[candidate_indices]
        else:
            candidates = embeddings
        candidates = candidates.contiguous()
        selected = [rng.randrange(candidate_count)]
        distances = torch.cdist(candidates[selected], candidates).squeeze(0)
        while len(selected) < target:
            idx = int(torch.argmax(distances).item())
            selected.append(idx)
            new_distance = torch.cdist(candidates[idx:idx + 1], candidates).squeeze(0)
            distances = torch.minimum(distances, new_distance)
        return candidates[selected].contiguous(), candidate_count

    def _fit_padim(self, embeddings: np.ndarray, grid_shape: tuple[int, int], selected_dims: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = grid_shape
        patch_count = h * w
        sample_count = embeddings.shape[0] // patch_count
        reshaped = embeddings.reshape(sample_count, patch_count, embeddings.shape[1])[:, :, selected_dims]
        mean = reshaped.mean(axis=0).astype(np.float32)
        inv_cov = np.empty((patch_count, len(selected_dims), len(selected_dims)), dtype=np.float32)
        eye = np.eye(len(selected_dims), dtype=np.float32) * 0.01
        for patch in range(patch_count):
            cov = np.cov(reshaped[:, patch, :], rowvar=False).astype(np.float32) + eye
            inv_cov[patch] = np.linalg.pinv(cov).astype(np.float32)
        return mean, inv_cov

    def _patchcore_score_map(self, features: Any, memory_bank: Any, torch: Any) -> np.ndarray:
        b, c, h, w = features.shape
        flat = features.flatten(2).permute(0, 2, 1).reshape(-1, c)
        distances = torch.cdist(flat, memory_bank.to(flat.device, non_blocking=True))
        nearest = distances.min(dim=1).values.reshape(h, w).detach().cpu().numpy().astype(np.float32)
        return nearest

    def _padim_score_map_torch(self, feature_map: Any, mean: Any, inv_cov: Any, dims: list[int], torch: Any) -> np.ndarray:
        c, h, w = feature_map.shape
        dim_index = torch.as_tensor(dims, dtype=torch.long, device=feature_map.device)
        flat = feature_map.reshape(c, h * w).T.index_select(1, dim_index)
        mean = mean.to(flat.device, non_blocking=True)
        inv_cov = inv_cov.to(flat.device, non_blocking=True)
        delta = flat - mean
        dist_sq = torch.einsum("pd,pde,pe->p", delta, inv_cov, delta).clamp_min(0.0)
        return torch.sqrt(dist_sq).reshape(h, w).detach().cpu().numpy().astype(np.float32)

    def _padim_score_map(self, feature_map: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray, dims: list[int]) -> np.ndarray:
        c, h, w = feature_map.shape
        flat = feature_map.reshape(c, h * w).T[:, dims]
        delta = flat - mean
        distances = np.empty(flat.shape[0], dtype=np.float32)
        for idx in range(flat.shape[0]):
            distances[idx] = math.sqrt(max(float(delta[idx] @ inv_cov[idx] @ delta[idx].T), 0.0))
        return distances.reshape(h, w)

    def _save_outputs(
        self,
        config: PartModelConfig,
        image: np.ndarray,
        score_map: np.ndarray,
        defect_mask: np.ndarray,
        status: str,
        anomaly_score: float,
        defect_area: int,
        bad_sector_ratio: float,
        bad_sectors: list[int],
    ) -> tuple[Path, Path]:
        config.result_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base = config.result_dir / f"{stamp}_{status.lower()}_hybrid"
        display = cv2.resize(image, (self.settings.image_size, self.settings.image_size), interpolation=cv2.INTER_AREA)
        if display.ndim == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        heat = cv2.applyColorMap(np.clip(score_map * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        blended = cv2.addWeighted(display, 0.58, heat, 0.42, 0)
        blended[defect_mask] = (0, 0, 255)
        overlay_path = base.with_suffix(".png")
        cv2.imwrite(str(overlay_path), blended)
        report = {
            "algorithm": "hybrid_patchcore_padim",
            "model_id": config.id,
            "status": status,
            "anomaly_score": anomaly_score,
            "defect_area_px": defect_area,
            "bad_sector_ratio": bad_sector_ratio,
            "bad_sectors": bad_sectors,
            "created_at": datetime.now().isoformat(),
            "overlay_path": str(overlay_path),
        }
        report_path = base.with_suffix(".json")
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        return overlay_path, report_path
