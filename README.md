# NeuroIris Blower Fan Industrial Vision Inspection

Python/PyQt6 inspection software for ultrasonic-welded blower fan parts. The system is designed for an industrial line PC with an RTX 5070 12 GB GPU, 32 GB RAM, Intel Ultra i7 265K, and an 8.3 MP USB3 camera.

## What changed for industry deployment

The production path combines **YOLO + anomalib SuperSimpleNet**:

1. The configured Ultralytics YOLO model finds the exact component region in every training and inference image.
2. SuperSimpleNet trains only on verified normal YOLO crops and learns a discriminative anomaly boundary using synthetic anomalies.
3. At inference, SuperSimpleNet produces a pixel anomaly map for the same exact crop.
4. The map is restricted to the fin/weld surface and evaluated by defect area and fin sector.
5. ByteTrack associates all rotation views with one part so any failed view remains latched.

This replaces the PatchCore/PaDiM memory-bank approach, which was too sensitive to coreset coverage and valid appearance variation. Existing hybrid checkpoints are incompatible and every part model must be retrained.

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

The live inspector automatically selects CUDA when a CUDA-capable PyTorch build is available. Override it with `BLOWER_INSPECTION_DEVICE` (`cuda` or `cpu`). The UI logs the selected inference device.

## YOLO localisation and part tracking

Select each part's YOLO `best.pt` from **YOLO PART MODEL**. The path is persisted in `config/models.json`. Full-FOV training images are YOLO-cropped in memory immediately before anomalib training; live images use the tracked YOLO bounding box. This train/inference symmetry keeps background and fixture variation out of SuperSimpleNet.

The PASS/FAIL result is latched over all rotation views and counted only when the tracked part crosses the configured production line. See `docs/system_design.md` for the full architecture.

## Training workflow

### Prepare normal crops from full-FOV images

**You do not have to run this command before training.** Both the GUI **TRAIN
SELECTED MODEL** action and `python -m blower_inspection.cli train BF-001`
automatically run YOLO on every full-FOV normal image immediately before model
training. Crops are staged temporarily for SuperSimpleNet, and the original images are not modified.

The preparation command below is optional and intended only when you want to
export and visually review the exact crops first.

Keep the original camera captures outside the configured training output, then
use the dataset preparation command. It loads the selected model's saved YOLO
`best.pt`, detects the component in every image, and writes only the exact crop:

```bash
python -m blower_inspection.cli prepare-dataset BF-001 path/to/full-fov-good-images
```

The default output is the model's `normal_image_dir` (for example
`data/training/BF-001/normal`). Use `--output path/to/crops` to review crops in a
staging directory first, `--no-recursive` to ignore subfolders, or `--replace` to
regenerate existing crops. Relative subfolders are retained to prevent duplicate
filenames from overwriting each other. `crop_manifest.csv` lists every written,
skipped, unreadable, or undetected source image. A failed/no-detection image is
not copied into the normal dataset, and the command exits nonzero when any image
fails so an incomplete dataset cannot be overlooked.

If source and output intentionally refer to the same directory, add `--replace`;
each crop is written to a temporary file and atomically replaces its full-FOV
source. This is destructive, so keeping originals and using in-memory training or
a separate staging `--output` is recommended. On Windows, quote every path that
contains spaces:

```powershell
python -m blower_inspection.cli prepare-dataset BF-001 "C:\camera images" --output "C:\normal crops"
```

Review every crop before training: delete crops containing a defective part,
incorrect detection, hand/tool occlusion, or unacceptable blur. Include normal
rotation angles and acceptable lighting variation, but never include defective
parts in the normal folder.

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
2. Capture at least 100 known-good parts; the software enforces a minimum of 20 images and uses those YOLO crops for SuperSimpleNet training.
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
- SuperSimpleNet checkpoints are independent per part number; validate each against golden PASS/FAIL samples.
- Use line PLC handshaking before enabling automatic reject gates.

## ESP32 fail output

The inspection UI can drive an ESP32 output pin when a part fails inspection. Flash
`firmware/esp32_fail_output/esp32_fail_output.ino` to the ESP32 with the Arduino
IDE or `arduino-cli`. The sketch uses GPIO 4 by default, which is commonly
labeled `D4` on ESP32 development boards. On every detected `FAIL`, the Python
app sends a wireless HTTP command to the ESP32 and the firmware drives GPIO 4
HIGH. On `PASS`, `STOP`, or standby/reset states, it drives GPIO 4 LOW.

### Recommended WiFi / hotspot mode

USB serial is no longer required for production triggering. By default the ESP32
creates its own hotspot:

| Setting | Default |
| --- | --- |
| SSID | `NeuroIris-ESP32` |
| Password | `neuroiris123` |
| ESP32 URL | `http://192.168.4.1` |

Deployment steps:

1. Flash the ESP32 firmware once.
2. On the production PC, connect WiFi to the `NeuroIris-ESP32` hotspot.
3. Start the app normally:

```powershell
python -m blower_inspection.app
```

The app defaults to WiFi transport and calls:

- `http://192.168.4.1/ping` to verify the ESP32 is reachable,
- `http://192.168.4.1/fail` when inspection result is FAIL,
- `http://192.168.4.1/pass` when inspection result is PASS or inspection stops.

If you put the ESP32 and production PC on another WiFi network, set the ESP32 URL
before launching:

```powershell
$env:BLOWER_ESP32_TRANSPORT = "wifi"
$env:BLOWER_ESP32_URL = "http://192.168.1.50"
python -m blower_inspection.app
```

For one-off CLI inspections, add `--esp32-output` after setting the same WiFi
environment variables if needed:

```bash
python -m blower_inspection.cli inspect BF-001 path/to/test_image.png --esp32-output
```

### Optional USB serial fallback

USB serial remains available for bench testing or if you explicitly prefer a
cabled trigger path:

```powershell
$env:BLOWER_ESP32_TRANSPORT = "serial"
$env:BLOWER_ESP32_PORT = "COM7"
$env:BLOWER_ESP32_BAUD = "115200"
python -m blower_inspection.app
```

When serial mode is enabled, the app scans available serial ports, sends a `PING`
handshake to the firmware, and only logs `ESP32 READY <port>` after the firmware
replies with `PONG` or `ESP32_FAIL_OUTPUT_READY`. Keep
`BLOWER_ESP32_REQUIRE_HANDSHAKE=1` unless you intentionally replaced the provided
firmware.

### Python serial dependency

You do **not** need the Arduino IDE on the deployment PC just to run inspection.
The Arduino IDE or `arduino-cli` is only needed on whichever computer you use to
flash `firmware/esp32_fail_output/esp32_fail_output.ino` onto the ESP32. The
deployment PC only needs `pyserial` if you choose optional USB serial mode. WiFi
mode uses Python's standard library HTTP client.

If serial mode says `No module named serial`, install `pyserial` into the same
Python environment that launches the app:

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

Use level shifting, an opto-isolator, or an interposing relay/PLC input module as
required by the connected machine. Do not connect ESP32 GPIO directly to voltages
above 3.3 V. If the ESP32 HTTP `/fail` response shows
`FAIL_OUTPUT=ACTIVE ... LEVEL=HIGH` but the relay does not energize, check whether
the relay module is active-low; if it is, change `FAIL_ACTIVE_LEVEL` in the
firmware from `HIGH` to `LOW` and re-flash.

## Production recommendations

- Use a mechanical nest with angular keying; do not rely on software alignment for large pose variation.
- Use diffuse ring/coaxial lighting for weld consistency and a low-angle secondary light for hairline cracks.
- Keep a master set of golden PASS/FAIL samples for every model and re-run them after any threshold or lighting change.
- Store failed overlays and JSON reports for process engineering review.
- SuperSimpleNet checkpoints are independent per part number; validate each against golden PASS/FAIL samples.
- Use line PLC handshaking before enabling automatic reject gates.
