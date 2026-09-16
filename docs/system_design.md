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
