# Fixed six-view PatchCore inspection

The default desktop now uses the stationary six-view workflow. The earlier
application, training code, hardware bridge, and experimental backends are
retained. Launch the earlier UI with `blower-inspection-ui --legacy`.

## Pipeline and machine sequence

```text
PLC completes the separate 360° fitting check
    -> begin(part_id, fit_check_complete=true)
    -> motor stops at 60° / 120° / 180° / 240° / 300° / 360°
    -> settle delay -> stationary burst -> quality gate -> best valid frame
    -> trained YOLOv12 component box -> padded, aspect-preserving letterbox
    -> optional bounded translation alignment
    -> original PatchCore backbone/features/coreset nearest-neighbor distances
    -> raw spatial heatmap -> threshold -> exclude borders and letterbox
    -> optional mild morphology -> remove small connected components
    -> retained pixels and percentage of valid ROI -> independent view verdict
    -> exactly six recorded positions -> final PASS or FAIL
```

PatchCore is the only anomaly model in this path. PaDiM, DINOv2, TAO,
VisualChangeNet, distillation, structural geometry votes, and image-score gates
do not participate. The original seeded projection, feature hooks, and coreset
sampler are reused. There is no per-frame heatmap normalization.

The fitting rotation produces no inspection view. Only an explicit stopped
position starts acquisition. The first post-stop frames are excluded by the
settle delay, then 10–30 burst frames are sampled. Invalid frames cannot count.
The selected frame must pass both full-frame and cropped-ROI quality checks.
Missing, weak, empty, boundary-touching, or partly outside detector boxes are
invalid; the application never substitutes the full camera frame.

Motor movement discards an unfinished burst and invalidates in-flight results.
A movement confirmation is required before advancing to another angle. Duplicate
stopped messages cannot create duplicate views. Failed quality/ROI acquisition
keeps the same expected angle and displays `VIEW N — ... FAILED`; send another
stopped event after correcting acquisition. An out-of-order angle cannot skip a
side. Position and full-cycle timeouts are visible, and incomplete cycles cannot
produce PASS. A full-cycle timeout or abort is an equipment FAULT.

Each valid view is evaluated independently. Five good views plus one failed view
produce final FAIL. Five good views alone remain INCOMPLETE. No averaging across
angles is used. The result records the failed angles and their region sizes.

## Installation, weights, training, and launch

Use Python 3.10+ and run from the repository root. The prepared cloud environment
already has a virtual environment. On Windows, create/activate one if needed:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[industrial,dev]"
blower-inspection-ui
```

On Linux:

```bash
source .venv/bin/activate
blower-inspection-ui
```

In **SETTINGS / YOLO / model path**, select your actual trained detector:

```text
C:\Users\Gigabyte\Downloads\AMBER ULTRASONIC PART.v1i.yolov12\runs\detect\train\weights\best.pt
```

Alternatively copy that file to `data/models/yolo/best.pt`, the portable default.
The software loads your YOLO checkpoint directly through Ultralytics. It does not
download a replacement detector or use YOLO for defect classification. Set
`class_id` if the detector contains several classes; `-1` accepts all classes.

Set **PATCHCORE / normal image dir** to a directory containing only reviewed
GOOD full-camera images directly inside that directory. Your Downloads/Pictures
files must first be classified and selected; do not use an unreviewed mixed
folder. At least 20 source images and 10 distinct quality-qualified crops are
required. Capture different physical good parts and all six stopped views under
the same fixture, lighting, camera settings, and detector as inference.

Save settings, then train:

```bash
python -m blower_inspection.cli train-fixed --settings config/fixed_inspection.json
```

Training and inference share `YoloROI`, letterboxing, RGB conversion, ImageNet
normalization (`mean=.485,.456,.406`, `std=.229,.224,.225`), feature extraction,
and optional translation alignment. Padding does not enter the GOOD memory bank
or acceptance denominator. Feature grids retain the rectangular ROI aspect.
The backbone is frozen; only GOOD reference embeddings form the coreset bank.
The final 20% (at least three) distinct crops are held out for a distance
distribution report and do not enter the bank. Split source captures by physical
part before training/validation to avoid leakage between neighboring images.

First training downloads the official pretrained torchvision backbone with its
normal checksum verification. The checkpoint then contains both the frozen
backbone and memory bank, so inference needs no backbone download. The default
output is:

```text
data/models/BF-001/patchcore_fixed.pt
data/models/BF-001/patchcore_fixed.training_report.json
```

The report records accepted/rejected images, disjoint manifests, bank size,
preprocessing, and held-out GOOD distances. It does not automatically set
production tolerances. The checkpoint records a bank hash and detector digest;
changing ROI preprocessing, detector weights/settings, alignment, or input
dimensions requires restoring the training settings or retraining.

Existing `patchcore_primary.pt` and hybrid checkpoints are left untouched. Their
old preprocessing and combined decision rules differ, so they remain usable via
the legacy application rather than being silently loaded with new normalization.
Training the new checkpoint does not overwrite them.

The shared model is the default; six models are optional. To use angle-specific
models, set `angle_specific=true`, configure all six `angle_model_paths`, and put
GOOD source images under `normal_image_dir/view_60`, `view_120`, ..., `view_360`.
Train each configured angle with:

```bash
python -m blower_inspection.cli train-fixed --angle 60 --settings config/fixed_inspection.json
```

Repeat for 120, 180, 240, 300, and 360. A missing angle checkpoint never falls
back silently to the shared model. The old `train BF-001` command is retained
for the old workflow; `train-fixed` is the new spatial workflow.

## Production and machine integration

**PRODUCTION** shows the camera/selected view, part ID, exact 0/6 through 6/6
progress, current position, each view's verdict, red defect regions, and cycle
time. **OPEN CAMERA** runs capture in its own thread. **ARM INSPECTION** validates
both detector and PatchCore checkpoints before accepting cycle events.

Manual commissioning is the default. Confirm the actual completed PLC fitting
rotation, start **NEW PART**, and use **CONFIRM NEXT STOP** only when the motor
has stopped at the indicated position. Use **MOTOR MOVING** between positions.
Manual mode cannot enable physical PASS/FAIL output. These controls exercise the
pipeline; they are not electronic evidence of a PLC/encoder position.

The repository has no plant-specific PLC register map or protocol, and the
mechanical description alone does not supply it. Production automation requires
mapping real fit-check, moving, stopped, and angle signals into the supplied
`tcp_json` gateway contract. The application acts as a TCP client to the
configured gateway. It never commands motor motion or guesses angles from
camera appearance. This avoids the phase-assignment behavior observed in PR #42.

Send one UTF-8 JSON object per line:

```json
{"event":"begin","part_id":"PART_000001","fit_check_complete":true}
{"event":"stopped","angle":60}
{"event":"heartbeat"}
{"event":"moving"}
{"event":"stopped","angle":120}
```

Continue through 360°. Send `{"event":"abort"}` for a machine fault/abort.
Send heartbeat/state messages more frequently than `signal_timeout_s`, including
while inference/storage runs. A lost connection, malformed event, skipped angle,
or lost heartbeat faults the active cycle. The gateway receives:

```json
{"event":"view_result","angle":60,"verdict":"PASS","view_count":1}
```

After six positions it also receives `part_result` with all per-view statistics,
failed angles, and the final verdict. Wait for the view acknowledgment before
moving if the acquisition/inference window has not completed. If one second is
insufficient on the deployment hardware, the PLC must wait or stop the cycle;
the software will not count a stale moving view to make progress appear complete.

Set `machine.interface=tcp_json`, gateway `host`/`port`, and only enable
`output_enabled` after integration and qualification. The existing
`ESP32FailOutputBridge` HTTP protocol and `NEUROIRIS_ESP32_URL`,
`NEUROIRIS_ESP32_TIMEOUT_S`, `NEUROIRIS_ESP32_ENABLED` environment settings are
preserved. The existing serial client and firmware remain unchanged. A failed
view asserts FAIL; PASS pulses only after six passing views and successful result
storage. An incomplete cycle, disk error, or output communication error cannot
produce a successful completion. The PLC must treat the connection/heartbeat and
FAULT state as an inhibit, rather than interpreting absence of FAIL as PASS.

Camera capture, inference, calibration processing, and image/history exports
run off the GUI thread. Machine events remain monitored while inference runs.
Active cycles use a fixed settings snapshot. Stop and wait for workers before
saving changed production settings; engineering tuning never rewrites a recorded
verdict. Worker shutdown is cooperative and waits for in-flight operations.

## Tolerance tuning and raw heatmaps

There are three independent controls:

1. **Heatmap threshold:** raw PatchCore distance at which a pixel becomes a
   candidate (`heatmap >= threshold`). The displayed image score is diagnostic.
2. **Minimum region pixels:** 8-connected candidate components below this area
   are discarded. Pixels exactly equal to the minimum are retained.
3. **Acceptance tolerance:** retained total pixels and percentage determine FAIL.

Borders apply to real letterboxed content, not black padding. The percentage is
`100 * retained_anomaly_pixels / valid_ROI_pixels`, with both letterbox and ignored
edges removed from the denominator. Pixel counts are in the configured canonical
ROI resolution (default 640×256), not sensor pixels. Changing resolution changes
the physical meaning of a pixel and requires retraining and tolerance validation.

The modes are:

| Mode | FAIL condition |
| --- | --- |
| PIXEL | pixels >= pixel threshold |
| PERCENTAGE | percentage >= percentage threshold |
| BOTH | both limits reached |
| EITHER | either limit reached |

Select a recorded view in **PRODUCTION**, then open **ENGINEERING**, or use
**LOAD SAVED VIEW** to load a `view_60`/... directory. Switch among ORIGINAL,
ROI, HEATMAP, OVERLAY, BINARY MASK, FILTERED MASK, and FINAL REGIONS. The original
view can show the YOLO box. Final regions have red outlines and a small red fill;
ignored pixels have a neutral overlay. The heatmap uses a fixed `0` to
`heatmap_display_max` scale, never automatic contrast stretching per image.

Adjust heatmap threshold, minimum component size, pixel tolerance, percentage,
border percentages, morphology, and mode. Masks, counts, regions, percentages,
and the preview verdict update from the cached raw map without PatchCore/YOLO
reruns. The original recorded verdict remains visible. **SAVE TOLERANCES FOR
FUTURE CYCLES** persists the reviewed values after inspection is stopped.

For pixel tuning, compare meaningful defect regions with harmless retained
speckles at the same ROI resolution. For percentage tuning, compare labeled
GOOD/NG distributions after setting the heatmap threshold and noise filter.
Start with BOTH, check false positives and false negatives, and validate small
chips/cracks explicitly; raising every limit until GOOD images pass can hide
defects. Opening/closing are OFF by default and limited to mild 3/5-pixel kernels.
Keep them off unless labeled evidence supports the change.

## Tolerance calibration and reproducible image checks

**TOLERANCE CALIBRATION** loads GOOD and NG full-camera images, with their known
angles. It uses exactly the same quality, YOLO, normalization, model, and tolerance
path. It can also load saved view directories containing cached maps. The table
shows class, anomaly pixels, percentage, largest component, and prediction.
Changing tolerances reuses maps in a worker without inference. Click a sample
to examine every intermediate engineering display. Invalid/unrun images are
reported separately, never counted as passes. FPR/FNR are N/A when a class is
missing. CSV export includes a companion threshold/settings JSON.

For repeatable validation, create a JSON manifest (paths relative to the manifest
are supported):

```json
[
  {"image":"GOOD/good_001.png","actual_class":"GOOD","angle":60},
  {"image":"NG/broken_fin_001.png","actual_class":"NG","angle":300},
  {"saved_view":"saved/PART_1/view_120","actual_class":"GOOD"}
]
```

Run:

```bash
python -m blower_inspection.cli calibrate-fixed validation/manifest.json --output data/results/fixed_calibration
python -m blower_inspection.cli inspect-fixed path/to/full_camera_image.png --angle 60
python -m blower_inspection.cli export-fixed-history data/results/inspection_history.csv
```

Calibration saves intermediate evidence for every valid image, including GOOD,
plus `calibration.json` with class counts, FPR/FNR, settings, invalid/unrun images,
and whether both classes completed. Exit code 2 means incomplete labeled
validation. `inspect-fixed` saves evidence for one image but explicitly reports
the part as INCOMPLETE; it cannot substitute for the six-view cycle.

Production storage defaults to `data/results/fixed_inspection/YYYY-MM-DD/` with
unique part directories and `view_60`/... subdirectories. Per-part and per-view
JSON always records counts/thresholds. Failed views save original, ROI, content
mask, raw `heatmap.npy`, fixed-scale heatmap PNG, overlay, thresholded mask,
filtered mask, and marked result. NPY preserves exact floating-point distances;
the PNG is visualization only. PASS images are optional and disabled by default.
Enable them for model-development comparisons. `history.jsonl` and CSV export
retain per-part cycle time, final result, and every view's diagnostics.

## Default configuration

All parameters are persisted in `config/fixed_inspection.json` and editable on
SETTINGS. The complete defaults are:

| Section | Parameters and defaults |
| --- | --- |
| CAMERA | index 0; width 1920; height 1080; FPS 30; exposure -6; gain 0; burst frames 15 |
| QUALITY | minimum blur 60; max dark 55%; max saturated 12%; mean intensity 10–245; blur weight 1; exposure penalty 1 |
| YOLO | `data/models/yolo/best.pt`; confidence .70; IoU .45; X/Y padding 4%; class ID -1; show ROI true |
| PATCHCORE | `data/models/BF-001/patchcore_fixed.pt`; GOOD dir `data/training/BF-001/normal`; input 640×256; WideResNet50-2; layers 2/3; max bank 8192; coreset ratio .08; max training images 300; shared model; angle paths empty |
| ALIGNMENT | disabled; translation only, max 2% per axis; minimum ECC correlation .8 |
| DISPLAY | fixed heatmap maximum 1.0 raw distance |
| TOLERANCE | heatmap .43; minimum component 500 px; pixel limit 2500 px; percentage .15%; BOTH; top/bottom/left/right ignores 5% each; opening/closing 0 (OFF) |
| MACHINE | 6 views; 60° steps; settle .15 s; acquire .65 s; minimum burst 10 frames; position timeout 2 s; cycle timeout 60 s; signal timeout 1 s; manual interface; gateway 127.0.0.1:8765; output disabled |
| STORAGE | `data/results/fixed_inspection`; save PASS false; save FAIL true; save heatmaps true |

Threshold defaults are starting values for engineering tests, not measured
qualification limits. Exposure/gain support and units depend on the camera driver.

## Verification and deployment limits

Automated tests exercise six positions, missed/duplicate/stale events, movement
during inference, fit-check gating, independent failed-side decisions, image
quality, strict ROI detection, normalization, all tolerance modes, ignored
borders, component sizes, raw-map retention, calibration metrics, persistence,
GUI controls, TCP messages, and worker behavior. A separate test uses real
PyTorch/ResNet computation on synthetic images to train, save, reload, and infer
without network access. Those synthetic tests do not establish defect accuracy.

The cloud has no production camera/PLC, GPU, Windows files, trained YOLOv12 file,
or labeled plant images. Real training/GOOD/NG/new-part accuracy, the one-second
timing budget, position signals, and output behavior still require commissioning
on the Windows inspection machine. Use independent physical GOOD and NG parts
at every angle and check the smallest required defects before enabling outputs.

## Files delivered

Created:

- `config/fixed_inspection.json`
- `src/blower_inspection/fixed_settings.py`
- `src/blower_inspection/stationary_roi.py`
- `src/blower_inspection/patchcore_spatial.py`
- `src/blower_inspection/anomaly_tolerance.py`
- `src/blower_inspection/fixed_views.py`
- `src/blower_inspection/machine_signals.py`
- `src/blower_inspection/fixed_workers.py`
- `src/blower_inspection/inspection_storage.py`
- `src/blower_inspection/tolerance_calibration.py`
- `src/blower_inspection/fixed_view_app.py`
- `src/blower_inspection/fixed_cli.py`
- `tests/test_fixed_inspection.py`
- `tests/test_spatial_patchcore.py`
- `tests/test_fixed_gui_workers.py`
- `docs/fixed_patchcore.md`

Modified: `src/blower_inspection/app.py` (default UI dispatch with `--legacy`),
`src/blower_inspection/cli.py` (new-command dispatch), and `README.md` (entry guide).
Old model registries, source implementations, tests, firmware, and weight files
remain available. A pre-change backup is retained outside the checkout at
`/workspace/backups/Ultrasonic-welding-before-fixed-views.tar.gz`.
