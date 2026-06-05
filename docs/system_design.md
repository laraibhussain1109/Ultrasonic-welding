# Blower fan inspection system design

## Target hardware

- GPU/CPU: RTX 5070 12 GB, 32 GB DDR5 RAM, Intel Ultra i7 265K.
- Camera: 8.3 MP USB3.0 industrial camera.
- Recommended optics: low-distortion lens, locked focus/aperture, calibrated working distance.
- Lighting: diffuse ring/coaxial light for weld uniformity; optional low-angle side light for hairline cracks.

## Inspection flow

1. Operator logs in.
2. Operator selects one of the four configured part models.
3. Camera captures the fan in a fixed nest.
4. Software aligns the frame to the trained normal template.
5. The model produces an anomaly heatmap in the fin/weld ring.
6. The decision logic fails parts with excessive defect area or abnormal fin-sector response.
7. Overlay PNG and JSON report are stored in the configured result folder.

## User roles

- `admin`: can inspect, capture normal samples, and train models.
- `user`: can inspect and capture images, but cannot train.

The training button is only created for admin sessions in the Tkinter UI.

## Four model support

`config/models.json` contains four part model records (`BF-001` through `BF-004`). Each model has independent:

- normal-image folder,
- trained model file,
- result folder,
- expected fin count,
- anomaly thresholds.

To add a newer model, add another JSON record, place normal images in its `normal_image_dir`, restart the UI, login as admin, select the model, and click **Train New/Selected Model**.

## Algorithm baseline

The current implementation is a normal-only template anomaly detector. It is useful when defect images are rare, which is common at the start of a machine-vision deployment.

Training:

- load all known-good images,
- resize to a stable processing size,
- align images with ECC registration,
- compute per-pixel normal mean and standard deviation,
- save a fin/weld ring mask and metadata.

Inspection:

- normalize and align the incoming image,
- compute z-score deviation from the normal template,
- threshold and clean the defect mask,
- calculate defect area,
- calculate per-sector fin/weld consistency,
- return PASS/FAIL with an overlay.

## Path to a GPU AI upgrade

When confirmed bad samples are available, add a PyTorch module using the same UI and folder structure:

- normal-only convolutional autoencoder or PatchCore for anomaly maps,
- supervised segmentation model for known crack/weld classes,
- ONNX/TensorRT export for deterministic deployment.

The RTX 5070 12 GB is sufficient for training moderate 1024-pixel anomaly models and running real-time inference.

## Validation plan

1. Collect at least 100 known-good photos per model.
2. Collect golden bad samples for crack, missing weld, wrong fin, and contamination.
3. Train each model from good photos only.
4. Tune `anomaly_threshold`, `min_defect_area_px`, and `max_bad_sector_ratio` in `config/models.json`.
5. Run a GR&R-style repeatability study by inspecting the same good and bad samples multiple times.
6. Lock production thresholds after meeting false reject and escape targets.
