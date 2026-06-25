# NeuroIris Blower Fan Industrial Vision Inspection

Python/PyQt6 inspection software for ultrasonic-welded blower fan parts. The system is designed for an industrial line PC with an RTX 5070 12 GB GPU, 32 GB RAM, Intel Ultra i7 265K, and an 8.3 MP USB3 camera.

## What changed for industry deployment

This is no longer a simple template-difference demo. The production path uses a **hybrid PatchCore + PaDiM anomaly detector** trained from normal images:

1. A pretrained CNN extracts multi-scale patch embeddings from known-good blower fan images.
2. **PatchCore** stores a coreset memory bank of representative normal patches.
3. **PaDiM** fits per-location Gaussian distributions over normal patch features.
4. Inspection fuses PatchCore nearest-neighbour distance and PaDiM Mahalanobis distance into a robust anomaly heatmap.
5. The heatmap is restricted to the fin/weld ring and checked by fin sector to catch cracks, missing welds, wrongly welded fins, and abnormal local surface changes.

This is a practical normal-only approach for factories because it does not require thousands of defect examples before first deployment.

## UI

The operator interface is built with **PyQt6** and follows the supplied dark neon NeuroIris layout:

- top status bar with online state, speed, tolerance, FPS, latency, and clock,
- left system-control panel with model selection, start, calibrate/load image, stop, reset, tolerance, surface speed, and admin training,
- large central camera/overlay viewer with anomaly score bar,
- right panel with PASS/FAIL/STANDBY badge, session statistics, last result, and inspection log.

## Default logins

| Username | Password | Role | Train button |
| --- | --- | --- | --- |
| `admin` | `admin123` | Admin | Visible/enabled |
| `operator` | `operator123` | User | Hidden/disabled |

Change these before production:

```bash
blower-inspection add-user admin --role admin --password "new-strong-password"
```

## Installation on the deployment PC

Install NVIDIA drivers and a CUDA-compatible PyTorch build for the RTX 5070 first, then install the app:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[industrial]
python -m blower_inspection.app
```

For Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[industrial]
python -m blower_inspection.app
```

## Runtime acceleration

The live inspector automatically selects CUDA when a CUDA-capable PyTorch build and NVIDIA driver are available. You can override the runtime device with `BLOWER_INSPECTION_DEVICE` (`cuda`, `cpu`, or `directml`/`dml` when `torch-directml` is installed). During live inspection the CNN backbone, PatchCore distance search, and PaDiM scoring stay cached on the selected accelerator instead of being rebuilt on every frame.

```powershell
$env:BLOWER_INSPECTION_DEVICE = "cuda"
python -m blower_inspection.app
```

The UI also logs the active inference device when live inspection starts. If that log shows `cpu` on the deployment PC, install a CUDA-enabled PyTorch wheel for the RTX 5070 or set `BLOWER_INSPECTION_DEVICE=cuda` after confirming `python -c "import torch; print(torch.cuda.is_available())"` returns `True`.

Each part model stores its own camera mode and default inspection ROI in `config/models.json`. Operators can update both from the GUI without editing code:

- **SET PART ROI** saves the selected model's normalized ROI percentages as that part's new default.
- **CAMERA FPS / RESOLUTION** saves the selected model's camera width, height, and FPS, for example 3840×2160 @ 30 FPS for full 8.3 MP mode or 1920×1080 @ 60 FPS for high-speed mode.

The blower fan ROI is still cropped before inference, but reducing USB/camera bandwidth can avoid a full-frame capture bottleneck when the full 8.3 MP image is not needed.

## Training workflow

Each configured model has its own normal-image folder:

```text
data/training/BF-001/normal
data/training/BF-002/normal
data/training/BF-003/normal
data/training/BF-004/normal
```

For each part model:

> The camera may see the whole table during capture. Live inspection, calibration image loading, CLI inspection, and training now automatically crop each frame to the long dark blower component ROI before the anomaly model runs, so keyboards, rails, cables, and bench clutter are excluded from scoring.

1. Mount the camera rigidly and lock exposure, gain, focus, white balance, and lighting.
2. Capture at least 100 known-good parts; the software enforces a minimum of 20 images and uses a deterministic, memory-bounded sample of up to 300 images for hybrid training.
3. Put images in the model's `normal_image_dir`.
4. Login as `admin`.
5. Select the model and press **TRAIN SELECTED MODEL**.
6. Validate thresholds with known-good and golden bad samples before automatic rejection.

For a new part, add a new record to `config/models.json`, create its normal-image folder, collect normal samples, then train from the admin UI.

## CLI

```bash
python -m blower_inspection.cli list-models
python -m blower_inspection.cli train BF-001
python -m blower_inspection.cli inspect BF-001 path/to/test_image.png
```

## Production recommendations

- Use a mechanical nest with angular keying; do not rely on software alignment for large pose variation.
- Use diffuse ring/coaxial lighting for weld consistency and a low-angle secondary light for hairline cracks.
- Keep a master set of golden PASS/FAIL samples for every model and re-run them after any threshold or lighting change.
- Store failed overlays and JSON reports for process engineering review.
- The hybrid trainer pools deep features to a 28×28 patch grid, projects them to 256 dimensions, caps PaDiM at 128 components, and caps PatchCore memory to prevent multi-gigabyte covariance allocations on line PCs.
- Use line PLC handshaking before enabling automatic reject gates.

## ESP32 fail output

The inspection UI can drive an ESP32 output pin when a part fails inspection. Flash
`firmware/esp32_fail_output/esp32_fail_output.ino` to the ESP32 with the Arduino
IDE or `arduino-cli`. The sketch uses GPIO 4 by default, which is commonly
labeled `D4` on ESP32 development boards. On every detected `FAIL`, the Python
app sends `FAIL` over USB serial and the firmware drives GPIO 4 HIGH. On `PASS`,
`STOP`, or standby/reset states, it drives GPIO 4 LOW.

The app now scans available serial ports and logs the connected port as `ESP32 READY <port>`. If the inspection log says `ESP32 OFFLINE`, confirm the ESP32 appears in Device Manager / `arduino-cli board list` and set the serial port manually:

```bash
export BLOWER_ESP32_PORT=/dev/ttyUSB0
export BLOWER_ESP32_BAUD=115200
python -m blower_inspection.app
```

Windows PowerShell example:

```powershell
$env:BLOWER_ESP32_PORT = "COM7"
$env:BLOWER_ESP32_BAUD = "115200"
python -m blower_inspection.app
```

If you set the wrong `BLOWER_ESP32_PORT`, the app will still try other detected ports by default. Set `BLOWER_ESP32_SCAN_ALL=0` to disable fallback scanning. Many ESP32 boards reset when the serial port opens, so the app waits briefly before sending the first command; override this with `BLOWER_ESP32_SETTLE=0` only if needed.


### Python serial dependency

You do **not** need the Arduino IDE on the deployment PC just to run inspection.
The Arduino IDE or `arduino-cli` is only needed on whichever computer you use to
flash `firmware/esp32_fail_output/esp32_fail_output.ino` onto the ESP32. The
deployment PC only needs the Python package `pyserial` so the app can open the
ESP32 USB COM port.

If the log says `No module named serial`, install `pyserial` into the same Python
environment that launches the app:

```powershell
python -m pip install pyserial
```

If you are running from this source checkout, reinstall the app dependencies:

```powershell
python -m pip install -e .[industrial]
python -m blower_inspection.app
```

Avoid launching as `python -m src.blower_inspection.app`; use
`python -m blower_inspection.app` after installation so the same environment gets
the package dependencies.

For one-off CLI inspections, add `--esp32-output`:

```bash
python -m blower_inspection.cli inspect BF-001 path/to/test_image.png --esp32-output
```

Use level shifting, an opto-isolator, or an interposing relay/PLC input module as
required by the connected machine. Do not connect ESP32 GPIO directly to voltages
above 3.3 V. If the ESP32 log shows `FAIL_OUTPUT=ACTIVE ... LEVEL=HIGH` but the relay does not energize, check whether the relay module is active-low; if it is, change `FAIL_ACTIVE_LEVEL` in the firmware from `HIGH` to `LOW` and re-flash.
