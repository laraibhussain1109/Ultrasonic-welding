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
