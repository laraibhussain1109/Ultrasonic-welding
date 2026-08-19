# NeuroIris Blower Fan Industrial Vision Inspection

Python/PyQt6 inspection software for ultrasonic-welded blower fan parts. The production anomaly path consumes an **NVIDIA TAO Deploy ONNX export**; PatchCore and SuperSimpleNet are not used for production decisions.

## Safety and quality boundary

No machine-learning detector is literally foolproof. This implementation is intended to be fail-closed, traceable, and suitable for formal line qualification. A model must still pass a documented gauge R&R and locked golden-set validation before it controls a reject mechanism. Use a safety-rated PLC/interlock where the risk assessment requires one.

The TAO runtime adds these production gates:

- TensorRT/CUDA execution is required by default; loss of the GPU provider inhibits inspection instead of silently falling back to a slow CPU path.
- The model must have one fixed-size image input and a valid 2-D, finite anomaly-map output. Ambiguous bindings require explicit configuration.
- Threshold calibration uses reviewed normal parts in the model's native score space, never per-frame min/max normalization.
- The calibration stores the ONNX SHA-256; a changed model cannot run against stale limits.
- Pixel area, whole-image score, and cylindrical-sector limits all participate in PASS/FAIL.
- A failed view is latched across the rotating physical part, and a runtime fault stops inspection and asserts the reject/inhibit output.
- Reports record the algorithm, model digest, thresholds, score, affected area, and sectors.

See [the TAO deployment and validation gate](docs/tao_deployment.md) before commissioning.

## Installation

Install NVIDIA drivers compatible with the qualified TAO/CUDA stack, then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[industrial,tao]'
```

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[industrial,tao]"
```

Start the UI with `python -m blower_inspection.app`.

## Model and dataset workflow

Each part in `config/models.json` points to its own TAO ONNX export, calibration file, normal-image directory, output directory, camera mode, ROI, and YOLO locator. Export a fixed-spatial-shape TAO visual-anomaly model as, for example, `data/models/BF-001/tao_anomaly.onnx`.

**TAO is not downloaded or trained by this application.** Install/launch the
NVIDIA TAO training toolkit separately through NVIDIA NGC when you do not already
have a qualified ONNX export. The project's `[tao]` extra is the deployment
runtime only. See [the installation and two-stage explanation](docs/tao_deployment.md#what-must-be-installed).

The `7.1.0-cosmos-rl` container is not a visual anomaly-training image. Do not
use it to manufacture or rename an ONNX artifact for this application; first
confirm an NVIDIA-supported visual anomaly localization recipe and its export
contract in the deployment guide.

For Docker diagnostics, run `nvidia-smi` directly or through
`/bin/bash -lc "nvidia-smi"`; `bash nvidia-smi` incorrectly treats the executable
as a shell script. The deployment guide includes the complete PowerShell command
and TAO-recommended shared-memory limits.

After GPU verification, run `scripts\tao_probe.ps1` from PowerShell. It saves the
container's actual TAO commands, packages, and anomaly-related files to
`tao_probe.txt`; GPU output alone cannot establish that anomaly training exists.
PyTorch `autograd/anomaly_mode.py` and `test_anomaly_detect_nan` matches are
gradient/NaN debugging code—not visual defect detection. The updated probe
filters those false positives and prints an explicit capability verdict.

The supported TAO design target is now **VisualChangeNet segmentation**. Its
export must accept a golden/reference image plus the inspected image and emit a
spatial change map. This is supervised paired change detection: the existing
good images provide no-change pairs, but real representative defective images
and pixel masks are still required to teach and validate actionable changes.

The commands have intentionally separate meanings: `tao-train` trains weights
inside the TAO Docker image, `tao-export` creates the ONNX artifact, and `train`
calibrates that existing artifact for the line. Running only `train BF-001`
before export now stops immediately with the corrective sequence instead of a
deep `FileNotFoundError` traceback.

Use the component crop for model training. The optional dataset preparation command applies the configured YOLO detector to full-camera known-good images while retaining a manifest:

```bash
python -m blower_inspection.cli prepare-dataset BF-001 path/to/full-fov-good-images
```

When the source is already the configured normal directory and `--output` is
omitted, the command now creates a timestamped sibling backup and safely replaces
each source with its crop. This directly supports the common capture-then-prepare
workflow. Use an explicit separate `--output` if the full frames should remain in
their original folder.

Review every crop. Remove defects, wrong detections, hands/tools, blur, and uncontrolled glare. Split by physical part, not adjacent frames, to prevent validation leakage.

After placing at least 20 reviewed normal images in the configured directory, calibrate the frozen TAO export:

```bash
python -m blower_inspection.cli train BF-001
```

If the ONNX file is elsewhere, import and persist it in the same command:

```bash
python -m blower_inspection.cli train BF-001 --model-file C:\path\to\tao_anomaly.onnx
```

The UI's **CALIBRATE TAO MODEL** button opens an ONNX chooser when the configured
export is missing, instead of failing with “TAO model export not found”.

For `nvidia_tao`, `train` means **calibrate the exported model**; neural-network training remains in NVIDIA's supported TAO container. Use 100+ physical normal parts spanning accepted process, finish, pose, and lighting variation for production qualification.

If output auto-discovery is ambiguous, configure `tao_input_name`, `tao_output_name`, and (when exported) `tao_score_output_name`. `tao_require_gpu` defaults to `true`.

## YOLO localization and rotating-part decisions

The live path uses the configured Ultralytics detector and ByteTrack track ID to crop the current component. All views during rotation belong to one physical-part session. Any failed view is latched; later good views cannot erase it. Production totals and the ESP32 signal update when the tracked center crosses the configured count line, not when a part merely disappears.

Camera exposure, gain, focus, white balance, lighting, fixture, working distance, resolution, and FPS must be locked to the validated setup. The UI can persist per-model ROI, detector, and camera settings.

## CLI

```bash
python -m blower_inspection.cli list-models
python -m blower_inspection.cli train BF-001
python -m blower_inspection.cli inspect BF-001 path/to/test_image.png
```

## Login and records

Default development logins are `admin` / `admin123` and `operator` / `operator123`. Change them before deployment:

```bash
blower-inspection add-user admin --role admin --password 'new-strong-password'
```

Daily totals are stored in `data/results/daily_statistics.json`; an operating day runs from local 07:00 through the following 07:00. Failed overlays and JSON reports are written below each configured `result_dir`.

## ESP32 reject output

Flash `firmware/esp32_fail_output/esp32_fail_output.ino` and configure the serial bridge for the deployment port. The app asserts FAIL for a rejected completed part. It also asserts FAIL/inhibit after a live inference fault; production PLC logic must distinguish and latch equipment faults according to the line risk assessment.

## Tests

```bash
pip install -e '.[dev]'
pytest -q
```
