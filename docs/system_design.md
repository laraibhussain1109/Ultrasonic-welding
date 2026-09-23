# System design

## Production flow

1. The USB3 camera supplies a locked-exposure frame.
2. YOLO/ByteTrack locates and identifies the physical rotating component.
3. The component crop is ImageNet-normalized and passed to the NVIDIA TAO ONNX export through ONNX Runtime. Providers are ordered TensorRT, CUDA, then CPU; CPU is used only when the part configuration permits fallback and no GPU provider can be activated.
4. The runtime validates bindings, shapes, and finite outputs and honors each part's `tao_require_gpu` safety policy.
5. A SHA-256-bound normal calibration supplies native pixel and image-score thresholds.
6. Surface ROI, connected-area, and cylindrical-sector gates create a per-view decision and localized overlay.
7. The rotating-part state machine latches any failed view until the part crosses the counting line.
8. JSON evidence, daily totals, and the ESP32/PLC result are updated for the completed physical part.

## Fail-closed conditions

Missing or changed artifacts, stale calibration, insufficient calibration samples, unavailable required execution providers, ambiguous bindings, dynamic spatial input, malformed tensors, and NaN/Inf outputs are equipment faults. Live acquisition stops and the reject/inhibit output is asserted. These conditions never produce PASS.

## Artifact layout

Each part configuration owns a normal-image directory, TAO ONNX export, calibration JSON, result directory, YOLO model, camera profile, and ROI. TAO training artifacts remain outside the runtime repository; only the qualified immutable ONNX export is promoted.

## Qualification

See `docs/tao_deployment.md`. Commissioning requires a locked, part-disjoint validation set, defect-size coverage, false-accept/false-reject confidence limits, latency limits, golden-part challenges, and controlled change management. Software controls support this process but cannot replace it.

## Hybrid evidence architecture

The axis of the inspected cylinder is horizontal. Circumferential fins appear as
approximately horizontal/curved edges; interval vertical structures are support
discs, not fin numbers. The former `cylindrical_sector_statistics` width slicing
was therefore a longitudinal anomaly summary, not a physical mapping of
`expected_fins`. Hybrid geometry uses vertical-gradient periodicity for the
circumferential pattern and explicitly named `longitudinal_sections` for the
axis direction.

Each sharp tracked crop is structurally matched (normalized gradient thumbnail
and edge projections) to a diverse reference bank. The closest three candidates
are conservatively registered by phase correlation plus bounded ECC Euclidean
motion. A failed bound or low correlation yields `VIEW INVALID`, never PASS.
The one best qualified pair enters VisualChangeNet exactly once with its existing
RGB/ImageNet preprocessing. A separate grayscale path measures orientation,
pitch, continuity, isolated gaps, periodicity, support ribs, and smooth glare in
the configured central inspection band.

Calibration uses the configured YOLO detector's exact component crop when its
checkpoint is available, matching live ByteTrack crop geometry. Registration is
solved on a bounded 512-pixel structural working image and its conservative
Euclidean transform is scaled back to the full RGB crop. TAO still receives its
normal registered RGB input through the existing resize/normalization path.

Fusion is deterministic: reject immediately for geometry at its fail threshold,
or corroborated TAO and geometry candidate evidence. TAO-only evidence becomes a
provisional candidate requiring consecutive views. Strong smooth glare suppresses
only TAO-only candidates when geometry and periodicity are normal; it never
suppresses strong geometry. Registration failures consume no valid view and an
all-invalid completed inspection fails closed. Reports include component scores,
reason codes, chosen reference, registration quality, and stage latency.

TAO masks originate on the square inference grid while geometry masks originate
on the native long-blower crop. Before fusion, geometry evidence is mapped with
nearest-neighbour interpolation onto the TAO grid; boolean mask fusion never
operates on arrays from different coordinate systems.

TAO-only persistence also requires a localized mask. Sub-threshold tiny changes
are treated as acceptable minor visual changes, while very broad TAO masks with
normal geometry are treated as unlocalized phase/appearance changes rather than
physical defects. The per-model minimum and maximum candidate-area ratios bound
this gate; geometry corroboration always retains authority regardless of area.

Part completion is configurable. `counting_line` preserves conveyor behavior and
publishes a final result only after line crossing plus the required rotation
views. `minimum_views` supports a fixed inspection nest: it publishes one final
result after the configured number of valid views and suppresses that
ByteTrack ID until the physical part leaves. A passing individual view is shown
as `VIEW PASS`; only session completion is shown and counted as final `PASS`.
If twice the required view count is attempted without enough valid views, the
fixed-nest session completes as a fail-closed `INSUFFICIENT_VIEW_QUALITY` fault.

Before registration or defect scoring, calibrated view-quality gates reject
motion blur and abnormal crop aspect ratio as retryable invalid views. Burst
selection also prevents a transient, small YOLO crop from winning merely because
its Laplacian variance is artificially high. Reflection-normalized geometry and
direct glare-mask exclusion prevent smooth moving highlights from manufacturing
isolated broken-fin evidence; strong surrounding geometry remains authoritative.

YOLO localization requires confidence strictly above 70% before a box can create
a crop or ByteTrack inspection view. The counting-line axis is independent of the
blower axis: supplied models draw a horizontal line parallel to the image x-axis
and evaluate top-to-bottom motion using the tracked box's y-center. Legacy
conveyors can retain a vertical line with `counting_axis: "x"`.

The minimum-view counter advances only for structurally distinct rotation phases.
Repeated frames of a stationary blower are rejected by `RotationPhaseGate` and
the UI displays `WAITING FOR ROTATION`; they cannot be counted as eight views or
latch a repeated false result. Geometry calibration stores minimum physical
reject deltas for orientation, pitch, periodicity, and continuity so a locked
camera with near-zero MAD does not turn a one-pixel measurement change into a
maximum-severity deformation.

Semantic fin reasons require corroboration inside geometry. A pitch anomaly by
itself is not a missing fin because autocorrelation can select an adjacent
harmonic on a complete pattern. Because pitch and periodicity come from the same
autocorrelation signal, they cannot corroborate one another. `MISSING_FIN`
requires pitch and periodicity plus independent continuity disruption;
`TILTED_FIN` requires orientation plus neighboring
structural inconsistency. A single global scalar cannot independently cross the
geometry candidate boundary. Isolated broken-fin evidence remains independently
authoritative because it is localized and rib/glare masked.

### Operator-qualified fixed ROI

Fixed-nest profiles set `lock_roi_after_confirmation: true`. YOLO runs once
after camera warm-up and proposes the complete-blower rectangle. Inspection does
not begin until the operator accepts that preview. Rejected previews are captured
again, and Cancel leaves inspection stopped. The accepted pixel coordinates are
then reused for sharp-frame selection, registration, TAO, and geometry throughout
the session. YOLO is not allowed to resize the crop from one rotation frame to
the next. Part-presence loss is debounced before a new fixed-nest part session is
created; it never changes the accepted rectangle.
# PatchCore-primary production inspection

The production path is now **YOLO presence/localization → canonical ROI → frame
quality → central band → PatchCore plus fin geometry → glare/edge filtering →
rotational confirmation**. TAO VisualChangeNet remains installed and its
training, resume, export, calibration and ONNX/CUDA code is retained, but it has
no production voting authority when `production_algorithm` is
`patchcore_geometry`. `engineering_compare_tao` is reserved for comparison
logging and must never alter the production result.

Training and inference both use `ROIStabilizer`: either a median of recent YOLO
boxes (`yolo_stabilized`) or one calibrated box after YOLO confirms presence
(`fixed_after_detection`). Crops are padded and letterboxed without changing
aspect ratio. A continuous distance-to-edge authority map, central viewing band,
and dilated support-rib mask prevent silhouette and normal rib boundaries from
rejecting a part. The long ROI is reported in longitudinal section scores rather
than treated as one undifferentiated location.

Only qualified, diverse good crops enter the memory bank. Blurry, clipped or
glare-dominated crops are rejected; correlated thumbnails are deduplicated; and
a disjoint 20% good-image subset calibrates fixed raw-distance thresholds. The
calibration records the memory-bank hash, training/calibration manifests,
backbone, layers, timestamp and quality settings. A mismatch fails closed as a
stale calibration. Optional `hard_good_dir` images are validation data and are
not automatically admitted into the memory bank.

At runtime, severe blur produces `VIEW INVALID`; it is never a PASS and never
counts as a qualified rotational view. Short camera exposure with stronger,
diffuse illumination is preferred to neural deblurring. A broad moving highlight
can suppress PatchCore authority and require another view, but cannot suppress
catastrophic geometry. One uncorroborated PatchCore spike is a candidate;
persistent candidate views reject through `RotatingPartInspector`; catastrophic
geometry rejects immediately. Operator output leaves normal pixels unchanged
and draws only confirmed red defect regions.

Typical commands:

```bash
python -m blower_inspection.cli train BF-001
python -m blower_inspection.cli inspect BF-001 path/to/full_camera_image.png
pytest -q
```

Training writes `patchcore_primary.pt`, `patchcore_calibration.json`, and a
quality/rejection report beside the model. Each inference result exposes
`frame_quality`, `patchcore`, `geometry`, and `total` milliseconds in
`latencies_ms`; actual values are hardware- and dataset-dependent and should be
collected on the deployment GPU rather than inferred from unit tests.

Known limitations: support-rib discovery is gradient based and should be checked
against each fixture; calibrated geometry baselines require representative good
parts; the optional segmentation-mask edge path is not required by the current
box-only YOLO; and thresholds cannot be claimed production-ready until measured
against independent GOOD, hard-good and labelled NG physical parts.
