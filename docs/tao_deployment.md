# NVIDIA TAO anomaly deployment gate

No vision model can honestly be called “fool proof”. This integration is
designed to **fail closed**, be measurable, and prevent a model artifact from
being promoted without validation. It does not remove the need for controlled
lighting, gauge R&R, golden samples, or a safety-rated PLC interlock.

## 1. Train and export

Train the visual-anomaly model with the NVIDIA TAO release qualified for the
deployment driver/CUDA stack. Use only YOLO-cropped component images and split
them by physical part (never adjacent video frames) into train, calibration, and
locked validation sets. Export a fixed-spatial-shape ONNX model to the configured
`model_file`, for example `data/models/BF-001/tao_anomaly.onnx`.

The export contract is:

* one float image input in NCHW (preferred) or NHWC layout;
* ImageNet-normalized RGB input;
* one two-dimensional anomaly-map output after squeezing batch/channel axes;
* optionally, one scalar image score output.

Set `tao_input_name`, `tao_output_name`, and `tao_score_output_name` in
`config/models.json` when the exported names are not unambiguous. Do not rename
bindings by trial and error.

## 2. Calibrate

Install `pip install -e .[industrial,tao]`, place at least 20 reviewed normal
images in the model's normal folder (100+ parts across normal process variation
is the production recommendation), then run:

```bash
python -m blower_inspection.cli train BF-001
```

For TAO this command does **not** retrain the neural network. It runs the frozen
export on the normal set and atomically writes a hash-bound calibration file.
The 99.9th-percentile normal tails plus a three-sigma guard establish pixel and
image thresholds without normalizing the current inspection frame.

## 3. Validate before enabling reject output

Use a locked set containing all accepted surface finishes, rotation angles,
lighting limits and multiple examples of every actionable defect. Record, per
defect class and size bin:

* false-accept rate and its confidence interval;
* false-reject rate and its confidence interval;
* minimum localized defect area;
* p50/p95/p99 inference latency under full line load;
* camera disconnect, corrupt output, stale calibration, and GPU-loss behavior.

Promotion should require signed acceptance limits agreed by manufacturing and
quality engineering. Challenge the golden set at shift start and after camera,
light, fixture, driver, TAO, ONNX, or threshold changes. Keep the PLC reject in
manual/observe-only mode until this qualification passes.

## 4. Fail-closed behavior

Inspection is inhibited on a missing/mutated model, missing/insufficient
calibration, unavailable GPU execution provider, ambiguous output binding,
dynamic spatial input, invalid shape, or NaN/Inf. These are equipment faults,
not PASS results. The application report stores the algorithm, thresholds, and
model digest for traceability.
