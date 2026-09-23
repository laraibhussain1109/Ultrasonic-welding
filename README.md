# NeuroIris Blower Fan Industrial Vision Inspection

Python/PyQt6 inspection software for ultrasonic-welded blower fan parts. The
default production decision uses **PatchCore + structural fin geometry +
multi-view confirmation**. NVIDIA TAO VisualChangeNet remains available for
engineering comparison, training, export, and ONNX runtime experiments, but it
does not control PASS/FAIL by default.

> **New installation?** Follow the complete [step-by-step operating guide](docs/getting_started.md)
> for PatchCore model training and starting a live inspection.

The supplied model IDs contain hyphens (`BF-001` through `BF-004`). For example,
BF-002 training images belong in `data/training/BF-002/normal`; a similarly
named `BF002` directory is not read by that model.

If Python reports `Unsupported inspection algorithm: hybrid_patchcore_geometry`,
run `blower-inspection doctor`. A traceback pointing at a different repository
directory means an older editable installation is active; reinstall from this
checkout with `python -m pip install -e ".[industrial,dev]"`.

## Safety and quality boundary

No machine-learning detector is literally foolproof. This implementation is intended to be fail-closed, traceable, and suitable for formal line qualification. A model must still pass a documented gauge R&R and locked golden-set validation before it controls a reject mechanism. Use a safety-rated PLC/interlock where the risk assessment requires one.

The TAO runtime adds these production gates:

- TensorRT/CUDA execution is preferred. The supplied part configurations fall back to CPU when ONNX Runtime cannot activate a GPU provider; set `tao_require_gpu` to `true` for lines that must inhibit inspection instead.
- The model must have one fixed-size image input and a valid 2-D, finite anomaly-map output. Ambiguous bindings require explicit configuration.
- Threshold calibration uses reviewed normal parts in the model's native score space, never per-frame min/max normalization.
- The calibration stores the ONNX SHA-256; a changed model cannot run against stale limits.
- Pixel area, whole-image score, and cylindrical-sector limits all participate in PASS/FAIL.
- A failed view is latched across the rotating physical part, and a runtime fault stops inspection and asserts the reject/inhibit output.
- Reports record the algorithm, model digest, thresholds, score, affected area, and sectors.

See [the step-by-step operating guide](docs/getting_started.md) and qualification
notes before commissioning. See [the TAO deployment guide](docs/tao_deployment.md)
only when using the optional TAO engineering workflow.

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

## Optional TAO engineering workflow (not default production training)

The section below is retained for teams that deliberately train/export TAO for
research comparison. For normal PatchCore production setup, skip this section
and use [the step-by-step operating guide](docs/getting_started.md).

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

Start with `python -m src.blower_inspection.cli tao-init BF-001`; it now downloads
the correct NVIDIA pretrained checkpoint into `data/models/pretrained` **and**
copies the exact TAO 7.1 VisualChangeNet segmentation YAML from the installed
container. Edit its dataset/training settings, then pass it to `tao-train --spec
...`. The model ID is optional for TAO commands and defaults to the configured
active model. `--skip-weights` is available only for intentional offline use.
An existing manual download under the repository is discovered and copied into
the canonical pretrained directory, so the successful NGC download is reused.
NGC's nested versioned copy is retained, and `tao-init` also creates the stable
direct path `data/models/pretrained/changenet_segment_levir_cd.pth` with hash
verification.

TAO's bundled YAML uses `num_epochs: 1` only as a smoke test. The launcher now
overrides the temporary runtime spec to 50 epochs by default (`--epochs N`) and
rejects values below 2. A completed `174/174` display means the one configured
epoch really ran, but class-1 F1/IoU must be nonzero and validated before export.
All CLI training commands stream output immediately and print the completed/total
epoch or image count, percentage, elapsed time, and estimated time remaining.
The desktop application's event log shows the same updates and automatically
keeps the newest update visible while calibration or training is running.

Interrupted TAO training can resume the full trainer state (model, optimizer,
scheduler, and epoch counter) from the highest numbered non-empty checkpoint:

```powershell
python -m src.blower_inspection.cli tao-train BF-001 `
  --spec specs/visual_changenet/bf-001_segmentation.yaml `
  --dataset "C:\Users\Gigabyte\Downloads\Prepare-data\TAO_VCN_DATASET" `
  --results-dir data/results/BF-001/tao `
  --epoch 400 `
  --restore_last_session
```

`--epoch` is an alias for `--epochs` and specifies the **total target**, not the
number of additional epochs. The restore option deliberately selects the
highest epoch number, so a newer accidental restart such as `epoch_010` cannot
hide an older `epoch_350` checkpoint. To continue after `model_epoch_350...`,
choose a target greater than 351 (for example, `--epoch 400`). During resume,
the launcher also overrides any stale vendor `pretrained_model_path` with the
selected mounted checkpoint; paths such as `/results/pretrained/...` therefore
cannot fail before TAO restores the trainer state.
After every training exit—including a TAO failure or Ctrl+C—the launcher replaces
TAO's Windows-visible zero-byte `changenet_model_segment_latest.pth` with a real,
verified copy of the highest completed `model_epoch_...pth`. The epoch files are
never deleted or renamed.

After training, run `tao-export BF-001 --spec
specs/visual_changenet/bf-001_segmentation.yaml --results-dir
data/results/BF-001/tao`, or press **EXPORT TAO MODEL** as an admin. The exporter
finds the completed checkpoint and writes the configured ONNX; **CALIBRATE TAO
MODEL** is the next separate operation.

An already converted dataset outside the repository is supported directly:
pass `--dataset "C:\Users\Gigabyte\Downloads\Prepare-data\TAO_VCN_DATASET"`.
It is mounted read-only and is never recreated, renamed, reorganized, or split.
Training uses an auto-discovered pretrained checkpoint, an explicit
`--pretrained-model <checkpoint.pth>`, or the explicit `--from-scratch` choice.

Run `python -m src.blower_inspection.cli tao-download-weights` to download
NVIDIA's VisualChangeNet segmentation LEVIR-CD trainable v1.0 model through an installed,
authenticated NGC CLI. The nested `changenet_segment_levir_cd.pth` path is found
automatically; `tao-train` can then omit `--pretrained-model`.
If the NGC folder was downloaded manually into the repository root, run
`tao-find-weights`; `tao-train` now searches the project root automatically and
also accepts the downloaded directory itself as `--pretrained-model`.

If an older generated YAML begins with the TAO release/license banner, delete it
and rerun `tao-init`. The copier now bypasses the container entrypoint with
`--entrypoint cat`, and training preflight rejects banner-corrupted YAML with a
direct repair message.

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

For `nvidia_tao`, `train` means **calibrate the exported model**; neural-network training remains in NVIDIA's supported TAO container. Calibration and inspection prefer TensorRT, then CUDA, and the supplied part configurations use ONNX Runtime's CPU provider when no GPU provider can be activated. Use 100+ physical normal parts spanning accepted process, finish, pose, and lighting variation for production qualification.

VisualChangeNet exports with deep-supervision outputs named `output0` through
`output3` and `output_final` are detected automatically; calibration uses the
fused `output_final` map. If a differently named export remains ambiguous,
configure `tao_input_name`, `tao_output_name`, and (when exported)
`tao_score_output_name`. Set `tao_require_gpu` to `true` to disable CPU fallback
for a production line.

## YOLO localization and rotating-part decisions

The live path uses the configured Ultralytics detector and ByteTrack track ID to crop the current component. For every tracked part it collects five camera frames, measures Laplacian sharpness, discards motion-blurred bursts, and sends only the sharpest photograph to inference. The supplied fixed-station profiles require six accepted, structurally distinct photographs before finalizing; any failed view among those six latches the whole part, including failures that occur after an earlier passing view. The burst size, sharpness floor, and minimum accepted-view count can be set per model with `capture_burst_frames`, `minimum_sharpness`, and `minimum_rotation_views`.

All accepted views during rotation belong to one physical-part session. Any failed sharp view is latched; later good views cannot erase it. Production totals and the ESP32 signal update only after the tracked center crosses the configured count line and the minimum view count has been reached, not when a part merely disappears. The displayed TAO anomaly score is calibrated: `1.000` is the fail threshold, rather than an easily misread raw probability such as `0.002`.

After upgrading, run **CALIBRATE TAO MODEL** once for each part model. Calibration version 3 derives its safety margin from robust variation at the normal-set tail instead of the standard deviation of every structured surface pixel; older calibration files are intentionally rejected so an over-lenient threshold cannot remain active.

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

## ESP32 PASS and reject outputs

Flash `firmware/esp32_fail_output/esp32_fail_output.ino`. On every final PASS verdict,
the app immediately pulses ESP32 GPIO 5 HIGH for 0.5 seconds; connect that output to
your PLC input through suitable isolation/level conditioning. Override
`PASS_OUTPUT_PIN` or `PASS_ACTIVE_LEVEL` in the sketch if required. GPIO 4 remains
the reject output: the app asserts it for a rejected completed part and also after
a live inference fault. Production PLC logic must distinguish and latch equipment
faults according to the line risk assessment.

## Tests

```bash
pip install -e '.[dev]'
pytest -q
```

## Legacy TAO hybrid rotating-blower inspection

The optional TAO comparison path retains NVIDIA TAO VisualChangeNet but does not treat a
single change-map pixel as a physical defect. `CALIBRATE` builds a diverse,
structural reference bank from reviewed normal images, qualifies conservative
Euclidean registration, and learns robust fin geometry. Live views use only the
best valid match among three descriptor candidates and run TAO once. Horizontal
fin orientation, pitch, continuity and gradient periodicity are evaluated in a
central cylinder band; persistent vertical support ribs are excluded.

Fusion immediately rejects catastrophic/corroborated geometry, while a TAO-only
change is provisional. Broad, smooth glare with intact geometry is reported as
`LIKELY_GLARE` and does not latch rejection. Invalid registration is an unusable
view; a part with insufficient qualified views fails closed. Operator imagery
keeps the camera pixels and draws red outlines only for confirmed defects.
Configuration is per model through the `hybrid_*`, `registration_*`,
`geometry_*`, `glare_*`, inspection-band, TAO evidence, reference-bank,
`longitudinal_sections`, and persistence fields in `config/models.json`.

The qualified Windows GPU path loads the PyTorch CUDA runtime before ONNX
Runtime, calls `onnxruntime.preload_dlls()`, and executes TAO through
`CUDAExecutionProvider` with `CPUExecutionProvider` as the configured fallback.
TensorRT is an optional future optimization and is not requested by default.

For fixed-nest rotating blowers, `inspection_completion_mode: "minimum_views"`
turns the configured rotation-view count into the physical-part completion
trigger. Conveyor installations can retain `"counting_line"`. The UI labels a
passing sample as `VIEW PASS` and reserves final `PASS` for completed sessions.

The supplied profiles are commissioned for a **60-degree usable surface arc per
view** (`visible_surface_arc_degrees: 60`) and six distinct qualified views per
revolution. This value records the optical coverage; it does not digitally
expand the camera image. Moving the camera farther from the part, selecting a
wider field-of-view lens, and arranging diffuse lighting so the additional
curved surface remains sharp and glare-free are required before changing a
40-degree installation to 60 degrees. After that mechanical/optical change,
capture new normal images across all six rotation phases and retrain/recalibrate
the selected model. Validate that adjacent views overlap—the structural phase
gate rejects duplicate views but does not measure absolute shaft angle.

When a defect is confirmed, its longitudinal section is latched for the entire
part session. The live camera continues to shade that axial band red and labels
it `ROTATE TO VERIFY` even after the defective circumferential surface has
rotated out of view. The marker clears only after YOLO confirms that the part
has left the station; it identifies the axial region to inspect, while the
operator rotates the table to bring the exact surface defect back into view.

Production YOLO crops require confidence strictly above 70%. Supplied blower
profiles use `counting_axis: "y"`, so the displayed counting line runs horizontally
along the image x-axis and crossing is measured from vertical part motion.

Fixed-nest view counting requires structural phase diversity: a stationary
blower produces one view and `WAITING FOR ROTATION`, not eight repeated decisions.
Missing- and tilted-fin reasons require corroborating geometry measurements.
Because pitch and periodicity share an autocorrelation source, missing-fin
classification also requires an independent continuity disruption; global pitch,
periodicity, or orientation noise alone cannot reject a part.

For fixed-nest inspection, `lock_roi_after_confirmation` makes YOLO a startup
localizer rather than a per-frame crop controller. Start Inspection displays the
detected complete-blower box and asks the operator to accept, recapture, or
cancel it. Once accepted, its full-frame coordinates remain immutable for every
rotation view in that inspection session; changing YOLO boxes therefore cannot
zoom into a few fins or expose neighboring/table regions mid-inspection.
