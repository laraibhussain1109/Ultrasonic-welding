# Stationary six-view inspection in the existing NeuroIris UI

Run `blower-inspection-ui` from the repository root. The login, panels, controls,
styles, training workflow, camera settings, and YOLO selector are the ones from
`main`. Select the trained YOLOv12 checkpoint with **YOLO PART MODEL**. On the
production Windows PC, the supplied path is:

```
C:\Users\Gigabyte\Downloads\AMBER ULTRASONIC PART.v1i.yolov12\runs\detect\train\weights\best.pt
```

The desktop loads the selected model's production checkpoint and paired
calibration from `config/models.json`. The BF-001 defaults are:

```text
data/models/BF-001/patchcore_primary.pt
data/models/BF-001/patchcore_calibration.json
data/models/BF-001/patchcore_primary.training_report.json
```

The **TRAIN PATCHCORE MODEL** button and `train BF-001` use that production
trainer. `train-fixed` now also uses it by default, reading GOOD/YOLO settings
from `config/fixed_inspection.json` and updating the selected desktop profile
after training succeeds:

Production destinations come from the selected `config/models.json` profile.
The fixed settings' `patchcore.model_path` names the engineering checkpoint and
the default `import-fixed` source; it is not the desktop output path.

```powershell
python -m blower_inspection.cli train-fixed --model-id BF-001 --settings config/fixed_inspection.json
```

For an already completed `patchcore_fixed.pt`, add the missing production
geometry and GOOD distance calibration without rebuilding its feature bank:

```powershell
python -m blower_inspection.cli import-fixed BF-001 --settings config/fixed_inspection.json
```

An explicit `--checkpoint "C:\path\to\patchcore_fixed.pt"` overrides the source
path. Preserve the settings used for that training and keep its reviewed GOOD
images accessible. Import prepares only the checkpoint's saved training and
calibration manifests, reuses its exact bank and frozen weights, preserves its
rectangular preprocessing, and writes the production files above. The source
checkpoint is preserved. Shared models without translation alignment are
supported; aligned/angle-specific models remain engineering tools.

`train-fixed --spatial-only` explicitly creates the separate engineering
checkpoint. Renaming that file or its report cannot supply production geometry
or thresholds. `doctor` prints the imported checkout and both desktop asset paths.

## Training large image folders

The training crash reporting `Unable to allocate 63.3 MiB` at the Laplacian
variance check is fixed. Quality analysis now computes the same native-resolution
sharpness, exposure, and glare measurements in strips with overlapping filter
neighborhoods. It does not resize photographs or relax quality thresholds.

`train-fixed` releases each source photo after preparing its canonical ROI rather
than retaining a full-resolution copy of every GOOD image. Both training paths
cache qualified, distinct crops and unsampled patch embeddings in the system
temporary directory. These files are removed on success or failure. Allow space
on that drive for the canonical crops and features during training.

All source images are still quality-checked. The existing duplicate rules and
evenly spaced selection across the whole dataset remain in place, including later
rotational views. Only the configured subset (300 images by default) enters
training/calibration, and their manifests remain disjoint. The seeded coreset
sampler are unchanged. New production checkpoints also save their frozen
backbone, feature settings, and canonical dimensions; a fresh desktop restores
them before inference without downloading new backbone weights. Previously
trained primary checkpoints remain supported. An interrupted quality-gating run
starts again from the first source image.

The approved live ROI includes the same padding as training. Its cropped still
and a full-camera image therefore normalize to the same canonical image. Training
uses a separate backend instance and requires inspection to be stopped. Geometry
and distance calibration finish before previous production assets are replaced.

## Capture and completion

After the initial continuous fitting revolution, the first stopped image locks
HOME without running defect inference. The next six moving/stopped transitions
are assigned 60°, 120°, 180°, 240°, 300°, and 360° in that order. Set
`skip_initial_fit_rotation: false` in `config/models.json` only for machines
without the separate initial fitting revolution; the first observed indexed
movement then leads to the 60° capture.

Camera acquisition and motion/burst processing run in a dedicated worker. The UI
reads the latest preview while a separate worker periodically checks YOLO presence;
neither slow YOLO nor PatchCore inference can block camera acquisition. The FPS
label reports acquired frames rather than UI refreshes.

Motion evidence combines exposure-adjusted frame differences and optical flow
in the fin band. At each stop, the software confirms three stationary frames and
waits `stationary_settle_ms` (200 ms, measured from the first stationary frame).
`capture_burst_frames` (15; limited to 10–30) is the target burst size. A shorter
burst is accepted when it contains at least `stationary_min_burst_frames` (3)
qualified, settled frames. It selects the sharpest qualifying still when the target
is reached, the `stationary_burst_window_ms` (350 ms) window ends, or confirmed
rotation resumes. The last case selects only stationary frames already buffered,
never the new moving frame. Blur, exposure, and glare quality thresholds still apply.

The camera stream is used for capture; defect inference runs only on the selected
still. Similar-looking fins never prevent a new stop from being counted. A long
dwell cannot create multiple views of the same stop.

Each side is queued independently while the preceding still is being processed.
Slow inference cannot overwrite an earlier queued side. Duplicate angles cannot
increase coverage. A stop with too few qualified stationary frames reports
`INSUFFICIENT_STATIONARY_FRAMES`; a stop ending before any settled frame reports
`NO_SETTLED_STATIONARY_FRAMES`. These consume that physical stop as an invalid
attempt; they cannot shift the next angle or become a good view.
All six qualified stops are required for PASS. Six attempts with missing quality,
or premature part removal, finish as FAIL. The queue is drained before a departed
part is finalized.

The current ESP32 firmware provides result outputs, not motor-position feedback.
Camera counting therefore assumes the commissioned machine really makes these
60° steps and that the camera observes every transition. It does not measure an
absolute shaft angle or prove an unseen/reversed step. The visible, usable camera
arc must actually cover 60°; six captures alone cannot recover hidden surface.
If acquisition itself remains at 2 FPS, a one-second dwell may still provide too
few settled frames. Check the camera's negotiated settings and measured acquisition
rate using the existing CAMERA FPS / RESOLUTION control on the production machine.

## Heatmap area tolerance and hybrid checks

The calibrated raw PatchCore distance threshold creates the anomaly mask without
per-image min/max normalization. Padding, outer edges, fixture/support ribs, and
reflection regions are excluded from the inspectable pixels. Reflection pixels
remain eligible where local fin deformation provides corroboration. Components
smaller than `min_defect_area_px` are removed. Thin elongated components are
filtered as scratches unless geometry corroborates them; the defaults are
`scratch_max_width_px: 2` and `scratch_min_aspect: 8` in canonical heatmap pixels.
Set the width to zero to disable that scratch filter.

```
anomaly percentage = 100 × retained anomaly pixels / inspectable pixels
```

The original 1%, 3%, 5%, 8% buttons and MANUAL input now set this area threshold
directly. For example, a 2% retained anomaly passes a 3% tolerance and exceeds a
1% tolerance. A nonempty area equal to the threshold exceeds it; zero anomalies
pass even with zero tolerance. The percentage does not rescale the component-size
filter, nor is it a percentage of an image score or count of sections.

An above-tolerance heatmap plus local geometry corroboration, or strong calibrated
PatchCore evidence, fails immediately. Weaker above-tolerance evidence is a
candidate and requires recurrence in the same longitudinal section across views.
Geometry alone cannot override the operator's heatmap-area tolerance. Actual
localization and the raw heatmap remain available on the inspection result.

Confirmed section locations accumulate for the physical part. The live preview
highlights the entire longitudinal band, including after the flaw rotates out of
sight and after final FAIL. Locations clear only at confirmed part departure.
The live label remembers the captured angles for each confirmed section, and the
final log records failed angles and sections. The existing LAST RESULT panel
reports the current captured angle and measured area %.

## Existing firmware outputs

`firmware/esp32_fail_output/esp32_fail_output.ino` from `main` is unchanged:

| Result/action | HTTP route | Serial equivalent | Output |
| --- | --- | --- | --- |
| FAIL | `/fail` | `FAIL` | GPIO 4 active HIGH, latched |
| Clear reject | `/pass` | `PASS` / `RESET` | GPIO 4 inactive; no PASS pulse |
| Final PASS | `/pass-pulse` | `PASS_PULSE` | Clear reject and pulse D5 HIGH for 500 ms |

D5 uses the board's `D5` mapping when defined, otherwise GPIO 5. The Python UI uses
the asynchronous HTTP bridge, default `http://192.168.4.1`, configurable with
`NEUROIRIS_ESP32_URL`. There is exactly one PASS pulse after six qualified views,
and no pulse for an individual good view. A failure remains latched while the
part is present. No firmware change is required.

The cloud environment can validate the capture, mask, aggregation, UI wiring,
and output calls using synthetic images. Production accuracy and motion thresholds
still require the actual camera, existing model assets, and labeled GOOD/NG parts
on the Windows machine.

Starting inspection now validates the PatchCore checkpoint, calibration path,
section count, model/calibration pairing, and thresholds before opening the camera.
A missing file reports its full path instead of surfacing a generic file error
after the first captured view.

The companion gap detector now counts separate interrupted fins rather than
gradient rows, so one genuine break is retained. Broad reflections are masked
through their connected bright boundary to prevent that boundary from becoming
false broken-fin evidence. Requalify geometry calibration and area settings with
the real GOOD/NG set after updating.

Regression coverage includes six short stops during a blocked YOLO check, bounded
burst capture, moving/blurred frame rejection, thread shutdown, model readiness,
heatmap area tolerances, unique angle coverage, retained defect locations, and
final output gating. Test reports are generated outside the checkout in the cloud
workspace and are not repository artifacts.
