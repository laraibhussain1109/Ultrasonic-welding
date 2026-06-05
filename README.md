# Blower Fan Vision Inspection System

Python-based inspection software for ultrasonic-welded blower fan parts. It supports four configured fan models, login roles, normal-image training, USB3 camera capture, and defect detection for cracks or wrongly welded fins.

## Why this design

The available RTX 5070 12 GB GPU is suitable for future CNN/autoencoder upgrades, but the first production baseline should be reliable with the existing normal photos. This project implements a normal-only anomaly inspection workflow:

1. Capture many known-good images for each part model under controlled lighting and fixture position.
2. Train a per-model normal appearance template from those images.
3. During inspection, align the incoming image to the learned normal template.
4. Detect abnormal local deviations as cracks, missing welds, wrong welds, damaged fins, or contamination.
5. Validate fin/weld sector consistency against the configured expected fin count.

This is intentionally camera/fixture friendly: the 8.3 MP USB3 camera should be mounted rigidly with diffuse coaxial or ring lighting, fixed exposure, and a mechanical nest so every fan appears in the same pose.

## Default logins

| Username | Password | Role | Train button |
| --- | --- | --- | --- |
| `admin` | `admin123` | Admin | Visible/enabled |
| `operator` | `operator123` | User | Hidden/disabled |

Change these before production with:

```bash
blower-inspection add-user admin --role admin --password "new-strong-password"
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
python -m blower_inspection.app
```

For CLI training and inspection:

```bash
python -m blower_inspection.cli train BF-001
python -m blower_inspection.cli inspect BF-001 path/to/test_image.png
```

## Folder workflow

Each of the four supplied models has a normal-image training folder in `config/models.json`:

```text
data/training/BF-001/normal
data/training/BF-002/normal
data/training/BF-003/normal
data/training/BF-004/normal
```

For a new model, add a block to `config/models.json`, capture/put normal images in the folder, login as admin, then press **Train New/Selected Model**. Training runs in a background thread and saves `normal_model.npz` under `data/models/<model-id>/`.

## Production recommendations

- Use a rigid nest with part-present sensing and a repeatable angular key.
- Lock camera exposure, gain, focus, white balance, and USB bandwidth.
- Use diffuse lighting for surface cracks and a low-angle secondary light if weld edge cracks are subtle.
- Start with at least 100 normal images per model, including accepted process variation.
- Keep rejected images with masks for later threshold tuning and supervised AI upgrades.
- Validate thresholds with golden good/bad samples before enabling automatic reject.

## Project layout

```text
config/models.json          Four model definitions and thresholds
config/users.json           Role-based login users
src/blower_inspection/app.py  Tkinter operator/admin UI
src/blower_inspection/cli.py  CLI train/inspect/user tools
src/blower_inspection/trainer.py Normal-only model training and inspection
docs/system_design.md       Detailed design and deployment guide
```
