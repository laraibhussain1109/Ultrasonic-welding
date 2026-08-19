# NVIDIA TAO anomaly deployment gate

No vision model can honestly be called “fool proof”. This integration is
designed to **fail closed**, be measurable, and prevent a model artifact from
being promoted without validation. It does not remove the need for controlled
lighting, gauge R&R, golden samples, or a safety-rated PLC interlock.

## What must be installed

TAO is NVIDIA's separate model-training toolkit; it is not an anomaly model and
it is not bundled with this repository. The `tao` Python extra in this project
installs only the **ONNX inference runtime**. It cannot create
`tao_anomaly.onnx` from good images.

If no qualified `.onnx` export exists yet, obtain NVIDIA TAO through NVIDIA NGC
and use the TAO release/container compatible with the workstation's NVIDIA
driver and CUDA environment. Follow the official instructions for that exact
release. Containerized training normally requires an NVIDIA GPU, its driver, a
supported Docker/WSL2 setup, access to NGC, a TAO experiment specification, and
a TAO architecture that produces a spatial anomaly map. TAO is a framework, so
selecting and validating the actual visual-anomaly architecture is still
required.

### Do not use the Cosmos-RL image for this workflow

`nvcr.io/nvidia/tao/tao-toolkit:7.1.0-cosmos-rl` is a Cosmos reinforcement-
learning image. Pulling it proves that Docker can reach NGC, but it does **not**
provide a visual surface-anomaly training recipe and it cannot be used to create
the `tao_anomaly.onnx` artifact expected by this application.

Do not start calibration and do not rename a checkpoint or an arbitrary ONNX
file to `tao_anomaly.onnx`. The runtime contract requires a model that actually
accepts an image and emits a spatial anomaly map.

Before downloading another multi-gigabyte container, use the NVIDIA NGC catalog
and the documentation for the exact TAO release to confirm all of the following:

1. A supported **visual anomaly detection/localization** task or recipe exists.
2. That recipe supports training on normal surface images (and any labels it
   requires).
3. Its export command produces ONNX, not only an encrypted or framework-specific
   checkpoint.
4. The exported model exposes a spatial anomaly-map output compatible with the
   contract below.

If the selected TAO release has no such recipe, TAO Toolkit is not itself a
replacement anomaly algorithm. In that case this repository cannot honestly
train the requested detector with TAO, and a supported NVIDIA model/recipe or a
different validated anomaly architecture must be selected first.

You can safely verify Docker GPU passthrough for the image you already pulled:

```powershell
docker run --rm --gpus all `
  nvcr.io/nvidia/tao/tao-toolkit:7.1.0-cosmos-rl nvidia-smi
```

That is only an environment check. A successful `nvidia-smi` is not model
training and does not make the Cosmos-RL image suitable for defect detection.

### Fix for `nvidia-smi: cannot execute binary file`

Do not run `bash nvidia-smi`. That asks Bash to parse the compiled
`/usr/bin/nvidia-smi` executable as if it were a shell script, which produces the
error shown above. Run the executable directly:

```powershell
docker run --rm --gpus all --shm-size=16g `
  --ulimit memlock=-1 --ulimit stack=67108864 `
  nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt nvidia-smi
```

Or, when a shell is actually needed, pass a command string with `-lc`:

```powershell
docker run --rm --gpus all --shm-size=16g `
  --ulimit memlock=-1 --ulimit stack=67108864 `
  nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt `
  /bin/bash -lc "nvidia-smi"
```

The shared-memory and `ulimit` flags also address the warning printed by the TAO
entrypoint. The successful Cosmos-RL output in the screenshot already confirms
that Docker, the NVIDIA container runtime, and the GPU are connected. Repeating
the check with `7.1.0-pyt` verifies that specific image, but still does not prove
that it contains a supported visual-anomaly recipe.

Next, inspect the tasks actually exposed by the PyTorch image rather than
guessing a training command:

```powershell
docker run --rm --gpus all --shm-size=16g `
  --ulimit memlock=-1 --ulimit stack=67108864 `
  nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt `
  /bin/bash -lc "tao --help || python -m tao --help || find /opt -maxdepth 3 -iname '*anomal*' -print"
```

Only continue if the image/documentation identifies a real visual anomaly
training and export task. Save the complete help output; the exact task name and
experiment-spec schema are needed before this repository can provide a truthful
training command.

The two `nvidia-smi` screenshots only complete the GPU check; they do not run
this task inventory. From the repository root, the included PowerShell helper
runs both checks and saves the information needed for the next decision:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\tao_probe.ps1
```

It writes `tao_probe.txt`. Review or share that text file—not another
`nvidia-smi` screenshot. In particular, the report inventories the `tao`
executable, TAO/Python help, installed relevant packages, and anomaly-related
files. An empty anomaly section means the generic image does not contain the
required recipe.

### Interpreting the supplied TAO 7.1 probe report

The reported `anomaly_mode.py`, `test_anomaly_detect_nan`, and
`test_anomaly_grad_warnings` paths are **PyTorch autograd debugging utilities**.
In PyTorch, “anomaly detection” in this context means detecting NaN values and
invalid operations while differentiating a neural network; it does not inspect
images for surface defects.

Likewise, `onnx`, `onnxruntime-gpu`, and `nvidia_tao_pytorch` being installed
only shows that the base image contains framework/export/runtime libraries. It
does not identify a trainable visual-anomaly task. The decisive lines in the
report are:

```text
tao: command not found
No module named tao
```

and the absence of EfficientAD, a visual-anomaly module, or another documented
surface-anomaly training recipe. Therefore this report does **not** demonstrate
TAO visual anomaly support. Do not proceed to calibration: there is no trained
model to calibrate.

The probe now excludes PyTorch autograd/test false positives, inventories the
top-level `nvidia_tao_pytorch` task modules, and prints an explicit verdict. Run
the updated script once after pulling future TAO images.

For this project there are only two technically valid ways forward:

1. Select an NVIDIA-documented TAO visual-anomaly train/export recipe and use
   the exact container/spec it names; or
2. Change the requirement to a TAO-supported supervised detector/segmenter,
   collect and label representative defective images, and update this
   application for that model's real output contract.

The second option cannot be trained from the 220 good-only images. A generic TAO
base image and an ONNX filename do not create an anomaly algorithm.

There are therefore two distinct operations:

1. **TAO training/export:** performed outside this application; produces a real
   `.onnx` file.
2. **Line calibration:** performed by this application on reviewed good images;
   produces `tao_calibration.json` and does not change neural-network weights.

Passing `data/models/BF-001` to `--model-file` only passes a directory. It does
not ask TAO to train there. Pass the exported file itself, for example:

```powershell
python -m src.blower_inspection.cli train BF-001 --model-file `
  "C:\TAO\exports\bf001_anomaly.onnx"
```

If a directory contains exactly one `.onnx` file, the CLI will now locate it.
If it contains none or more than one, it reports the precise corrective action.

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

To import an export that is not already at the configured path, use
`--model-file path/to/model.onnx`. The UI calibration button likewise asks for
the ONNX export if it is missing.

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
