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
