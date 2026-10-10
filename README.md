# NeuroIris Blower Fan Industrial Vision Inspection

The desktop keeps the NeuroIris interface from `main`: login, model selection,
camera controls, tolerance presets/manual input, live preview, training, counters,
and inspection log. Start it with `blower-inspection-ui`.

Production uses **PatchCore with calibrated fin geometry**. After the separate
fitting revolution, the camera selects one qualified still from each of six
60° stops: **60°, 120°, 180°, 240°, 300°, 360°**. Acquisition and YOLO presence
checks run separately from the UI and inference. Every side is queued, and six
valid views are required before a final PASS pulse.

Qualified native images are also inspected continuously between those selected
stills, so a missed/subtle rotation cannot disable defect detection. Video FAIL
evidence latches the affected sections; repeated video images never substitute
for six valid stop views. Windows startup measures camera formats/backends at
the selected resolution, reasserts MJPEG after size/FPS changes, and compares
automatic exposure if retained driver settings deliver a slow stream. See
[camera recovery and physical verification](docs/camera_recovery.md).

The tolerance percentage compares filtered anomalous heatmap pixels with
inspectable area. Borders, padding, support ribs, reflections, small components,
and thin uncorroborated scratches are filtered. Confirmed longitudinal sections
remain highlighted until the physical part leaves the station.

See [operating notes](docs/stationary_hybrid.md) and
[getting started](docs/getting_started.md) for commissioning and capture settings.

## Install and launch

Use Python 3.10+ and run commands from the repository root. On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[industrial]"
blower-inspection-ui
```

Use **YOLO PART MODEL** to select your actual YOLOv12 `best.pt`. This persists its
path in the selected model's `config/models.json` profile. Default development
logins are `admin` / `admin123` and `operator` / `operator123`.

## Production training and files

Put reviewed GOOD full-camera photographs in the selected model's
`normal_image_dir`. Use multiple physical good parts and all six stopped views
under the production camera, fixture, focus, exposure, and lighting. Training
requires at least 20 source images and 10 distinct qualified crops. Source
photographs are preserved.

The original **TRAIN PATCHCORE MODEL** button and this command use the same
production trainer and selected desktop profile:

```powershell
python -m blower_inspection.cli train BF-001
```

If your GOOD/YOLO paths are saved in `config/fixed_inspection.json`, use:

```powershell
python -m blower_inspection.cli train-fixed --model-id BF-001 --settings config/fixed_inspection.json
```

Default `train-fixed` also produces the desktop artifacts and updates the chosen
profile after successful training. For BF-001, the configured defaults are:

Production output paths come from `config/models.json`. The fixed settings'
`patchcore.model_path` is the engineering/import source path.

| Artifact | Path |
| --- | --- |
| Model, frozen backbone, and geometry | `data/models/BF-001/patchcore_primary.pt` |
| Paired GOOD distance thresholds | `data/models/BF-001/patchcore_calibration.json` |
| Training report | `data/models/BF-001/patchcore_primary.training_report.json` |

Training gates quality, removes redundant images, retains coverage across the
whole dataset, and separates bank-building images from calibration images.
Temporary crop/feature files bound RAM use and are removed after success or
failure. Allow temporary disk space for those files. Native-resolution quality
checks use small overlapping strips. New models save the trained frozen weights
and feature settings so desktop reload uses the same features without a new
backbone download. Stop inspection before training a model.

## Reuse a completed spatial model

If an earlier run produced `patchcore_fixed.pt` and only a training report, add
the missing desktop geometry and calibration with:

```powershell
python -m blower_inspection.cli import-fixed BF-001 --settings config/fixed_inspection.json
```

Keep the training settings and GOOD images accessible. Add
`--checkpoint "C:\path\to\patchcore_fixed.pt"` if the source checkpoint is elsewhere.
The import reuses the exact memory bank and frozen backbone, preserves its
rectangular feature preprocessing, and fits the missing production geometry and
GOOD thresholds using the checkpoint's saved image manifests. It preserves the
source checkpoint and writes the paired production files. Shared spatial models
without translation alignment are supported.

The separate raw spatial engineering workflow is explicit:

```powershell
python -m blower_inspection.cli train-fixed --spatial-only --settings config/fixed_inspection.json
python -m blower_inspection.fixed_view_app
```

Those tools are described in [the spatial guide](docs/fixed_patchcore.md).
The desktop checks checkpoint format, geometry, bank/calibration pairing, feature
settings, and thresholds before opening the camera. A spatial checkpoint requires
import rather than a filename change.

## Inspection and diagnostics

For low camera FPS, start with the raw DirectShow/Media Foundation benchmark,
then the separate stage and six-stop audits in
[Windows camera performance](docs/camera_performance.md). The existing GUI shows
separate acquisition, processing and preview rates and read/inference latency.

```powershell
python -m blower_inspection.cli doctor
python -m blower_inspection.cli list-models
python -m blower_inspection.cli inspect BF-001 path/to/test_image.png
```

`doctor` prints the actual imported checkout and model/calibration paths for
each profile. This helps detect an editable install pointing to an older ZIP
folder. Validate model accuracy, tolerances, camera coverage, and line timing
with independent labeled GOOD/NG parts on the production machine.

The six-view controller counts observed motor transitions; the current firmware
does not report encoder positions. Commission the physical camera coverage and
motor's six 60° steps accordingly.

## ESP32 outputs and records

The firmware in `firmware/esp32_fail_output/esp32_fail_output.ino` is the one from
`main`. `/fail` latches GPIO 4 HIGH. `/pass` clears reject. `/pass-pulse` clears
reject and pulses D5 HIGH for 500 ms (GPIO 5 when the board has no D5 mapping).
The desktop emits one PASS pulse after six valid views. FAIL remains latched while
the part is present. The default HTTP address is `http://192.168.4.1`, configurable
with `NEUROIRIS_ESP32_URL`.

Daily totals and optional failed evidence are stored under `data/results`.
An operating day runs from local 07:00 to the next local 07:00.

## Optional TAO engineering tools

TAO/VisualChangeNet export and comparison tools remain available separately;
the supplied production profiles use PatchCore/geometry. See
[TAO deployment](docs/tao_deployment.md) and the available `tao-*` CLI commands.

## Tests

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```
