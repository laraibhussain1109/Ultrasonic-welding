# Industrial blower fan inspection system design

## Target hardware

- GPU/CPU: RTX 5070 12 GB, 32 GB DDR5 RAM, Intel Ultra i7 265K.
- Camera: 8.3 MP USB3.0 industrial camera with locked exposure/focus/gain.
- Fixture: rigid nest, repeatable angular key, part-present trigger.
- Lighting: diffuse ring/coaxial light plus optional low-angle crack light.

## Algorithm

The deployed anomaly model is a hybrid of PatchCore and PaDiM.

### PatchCore branch

- Extract CNN patch embeddings from normal images.
- Select a farthest-first coreset memory bank to keep inference fast.
- Score each inspection patch by nearest-neighbour distance to the normal memory bank.

### PaDiM branch

- Use the same CNN patch grid.
- Fit a per-patch multivariate Gaussian distribution over a memory-bounded random projection of normal embeddings.
- Score each inspection patch with Mahalanobis distance.

### Hybrid decision

- Normalize both score maps robustly.
- Fuse maps using configurable weights.
- Bound RAM use by pooling CNN features to a 28×28 grid, projecting to 256 dimensions, fitting PaDiM on 128 selected components, sampling up to 300 training images, and limiting PatchCore memory/candidate patches.
- Resize to the camera display size and smooth.
- Restrict analysis to the annular fin/weld region.
- Fail parts by defect area or abnormal fin-sector distribution.
- Save overlay PNG and JSON result reports.

## UI and roles

The UI is PyQt6, dark, high-contrast, and modeled after the supplied NeuroIris reference image. Admin and operator sessions share the inspection screen, but only admins see the training control.

- `admin`: inspect, capture/load images, train selected/new models.
- `user`: inspect and operate only; training is hidden.

## Four model support

`config/models.json` defines four independent model records. Each record stores:

- model ID and display name,
- normal-image folder,
- hybrid checkpoint path,
- result folder,
- fin count,
- ROI radius ratios,
- thresholds.

## Validation plan

1. Collect 100+ normal parts per model across normal process variation.
2. Train hybrid checkpoints per model.
3. Collect golden known-bad samples: cracked fin, missing weld, wrong weld, damaged fin, contamination, and part-position errors.
4. Tune `anomaly_threshold`, `min_defect_area_px`, and `max_bad_sector_ratio` per model.
5. Run repeatability trials by inspecting the same parts repeatedly across shifts.
6. Freeze camera/lighting settings and version trained checkpoints before line release.
