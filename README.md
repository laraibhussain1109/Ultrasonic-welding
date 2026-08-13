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

## YOLO part localisation, rotation handling, and daily counts

The live inspection path uses your own Ultralytics **YOLO12s** `best.pt` model to
locate the exact part crop. In the UI, select the part model, press **YOLO PART
MODEL**, and choose `best.pt`; the path is persisted in `config/models.json` for
that part model. Install the optional dependency before deployment:

```bash
pip install -e .[industrial]
```

Each camera frame is passed to YOLO with Ultralytics `bytetrack.yaml` and the
resulting persistent track ID is used to associate detections. PatchCore/PaDiM
runs only inside the current YOLO bounding box. All views while a cylindrical
part is rotating are accumulated into one physical inspection session; a failed
view is latched and makes the final verdict `FAIL`, even when every later view is
flawless. If brief occlusion changes ByteTrack's ID, an overlapping detection is
reattached to the recent session instead of clearing that failure.

The PASS/FAIL signal and production counters are updated only when the detected
part center crosses the configured counting line (`counting_line_ratio`, default
45% of frame width—slightly left of center—moving left-to-right). Disappearance or rotation in place
does not count or complete a part. Set `counting_direction` to `right_to_left`
when production flows in the opposite direction. The operator must rotate the
complete curved surface before moving the part across the displayed count line.

After upgrading to this version, retrain each hybrid model from its normal-image
folder. Training now uses the same YOLO exact crop as live inference; old hybrid
checkpoints trained on the larger fixed ROI can produce broad false positives and
miss small surface marks because their feature positions do not match the live
crop. The live overlay leaves normal pixels unchanged and colors only confirmed
thresholded anomaly regions.

The hybrid result also includes a high-resolution structural fin-continuity
check. It detects short gaps in horizontal fins that the downsampled deep
features may treat as harmless texture, while suppressing the normal full-height
support ribs. Confirmed gaps are promoted to full overlay severity and participate
in the same latched FAIL verdict. Repeated gap columns and any broad structural
response are rejected as normal part texture, preventing the structural check
from painting or failing the entire blower surface.

Hybrid training also stores a robust median/MAD reference made from all known-good
YOLO crops. Before comparison, broad illumination is removed with divisive
normalization. Live deep anomalies must be corroborated by this fixed normal
reference, so relative heatmap normalization cannot make every good part fail.
Broad bright, low-texture lamp reflections are masked, while sharp white lines,
cracks, and broken edges remain eligible as physical defects. Retraining is
required once to add these reference statistics to an existing checkpoint.

Train the anomaly model on the **YOLO-cropped component only**, not the complete
camera FOV. Full-FOV training wastes PatchCore patches on the table, fixture,
keyboard, and lighting and reduces the pixel resolution available for a small
broken fin. The production profile uses a 640×640 input, an 80×80 embedding grid,
an 8,192-patch training coreset, and up to 1,024 runtime memory patches. This is a
deliberate detail/latency balance for the target RTX GPU and gives small defects
substantially more representation than the old 384×384/56×56 profile. Retrain all
part models after this upgrade because checkpoint version 5 contains the new
resolution and PatchCore geometry.

Daily counters persist in `data/results/daily_statistics.json`. An operating day
runs from local time 07:00 through the next local 07:00; the UI's reset control
reloads those protected daily totals rather than erasing production records.

## Training workflow

### Prepare normal crops from full-FOV images

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
- The hybrid trainer pools deep features to a 28×28 patch grid, projects them to 256 dimensions, caps PaDiM at 128 components, and caps PatchCore memory to prevent multi-gigabyte covariance allocations on line PCs.
- Use line PLC handshaking before enabling automatic reject gates.
