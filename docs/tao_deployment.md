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

VisualChangeNet is the relevant TAO task when it is present. It is a **paired
change-detection model**, not a good-only anomaly learner. Training examples
must contain a registered reference image, a comparison image, and the target
change label/mask required by the selected VisualChangeNet variant. Good/good
pairs teach the no-change class, but good images alone provide no positive
signal for scratches, broken fins, or missing welds.

The application runtime is therefore VisualChangeNet-specific:

* the ONNX export must have two image inputs (golden reference and inspected
  part), not the previous generic single-image contract;
* segmentation logits are converted to a configurable change-class probability
  map (`tao_change_class_index`, default `1`);
* calibration automatically selects and persists a reviewed normal medoid as
  the golden reference when one is not configured;
* both the model and reference SHA-256 are bound into calibration, so changing
  either inhibits inspection.

For this project there are two technically valid VisualChangeNet data paths:

1. Train VisualChangeNet segmentation with aligned golden/part pairs and
   pixel-level change masks for representative defects; or
2. Train VisualChangeNet classification with paired images and change labels,
   accepting that classification cannot localize defects for a heatmap/box UI.

Neither option can be qualified from the 220 good-only images alone. For the
surface-localization requirement, use the segmentation variant and collect real
defective parts with masks. Synthetic defects may augment training, but must not
replace locked validation on real defects.

There are therefore two distinct operations:

1. **TAO training/export:** performed outside this application; produces a real
   `.onnx` file.
2. **Line calibration:** performed by this application on reviewed good images;
   produces `tao_calibration.json` and does not change neural-network weights.

This distinction explains the `FileNotFoundError` shown by
`python -m ...cli train BF-001`: that command reached calibration, looked for
`data/models/BF-001/visual_changenet.onnx`, and correctly stopped because TAO
training/export had not created it. Finding VisualChangeNet Python files in the
container confirms availability; it does not mean a model has been trained.

The repository now exposes the external tasks explicitly:

```powershell
# Copy TAO 7.1's real segmentation spec out of the installed container.
python -m src.blower_inspection.cli tao-init BF-001

# Edit the copied YAML first. Every dataset/result path it references must be
# under the repository and use /workspace/project/... inside the container.
notepad specs/visual_changenet/bf-001_segmentation.yaml

python -m src.blower_inspection.cli tao-train BF-001 `
  --spec specs/visual_changenet/bf-001_segmentation.yaml `
  --dataset "C:\Users\Gigabyte\Downloads\Prepare-data\TAO_VCN_DATASET" `
  --pretrained-model "C:\path\to\changenet_segment_levir_cd.pth"

python -m src.blower_inspection.cli tao-export BF-001 `
  --spec specs/visual_changenet/bf-001_segmentation.yaml

python -m src.blower_inspection.cli train BF-001 `
  --model-file data/models/BF-001/visual_changenet.onnx
```

`tao-init` copies the exact default spec from the installed container rather
than generating an invented schema. The training command invokes TAO's installed module
`nvidia_tao_pytorch.cv.visual_changenet.entrypoint.visual_changenet train`; the
second invokes its `export` task; only the third performs line calibration.

Before writing a spec, inspect the exact TAO 7.1 schema and bundled examples:

```powershell
docker run --rm --gpus all --shm-size=16g `
  --ulimit memlock=-1 --ulimit stack=67108864 `
  nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt /bin/bash -lc `
  "python -m nvidia_tao_pytorch.cv.visual_changenet.entrypoint.visual_changenet --help; find /usr/local/lib/python3.12/dist-packages/nvidia_tao_pytorch/cv/visual_changenet -iname '*.yaml' -o -iname '*.yml'"
```

Use the segmentation schema shipped by that installed version. Do not invent
configuration keys from a different TAO release.

### Fix for `yaml.scanner.ScannerError` at line 12

Earlier `tao-init` versions ran `cat` through TAO's default Docker entrypoint.
That entrypoint printed the release banner, license URL, GPU warning, and SHMEM
notice to stdout before printing the YAML. All of those lines were accidentally
saved into `bf-001_segmentation.yaml`; the colon in the license URL then caused
PyYAML's `mapping values are not allowed here` error. The banner displayed above
`encryption_key:` in the supplied file confirms this corruption.

The copier now runs Docker with `--entrypoint cat`, strips any pre-YAML output,
and the trainer rejects a banner-corrupted file before launching TAO. Repair an
existing checkout by regenerating the spec:

```powershell
Remove-Item .\specs\visual_changenet\bf-001_segmentation.yaml
python -m src.blower_inspection.cli tao-init BF-001
Get-Content .\specs\visual_changenet\bf-001_segmentation.yaml -First 5
```

The first line must be `encryption_key:`, not `=== TAO Toolkit PyTorch ===`.
The launcher described next safely overrides the default dataset/results/weights
locations in a temporary runtime spec.

### Using the already-prepared external `TAO_VCN_DATASET`

Do not copy or reorganize the dataset. Supply its real Windows location with
`--dataset`. The launcher validates only that `A`, `B`, `label`, and `list`
exist, the required split lists are non-empty, and paired filenames match. It
does not write to the dataset. Docker mounts it read-only at
`/data/TAO_VCN_DATASET` and a temporary runtime copy of the YAML overrides
`dataset.segment.root_dir` accordingly; the original YAML is retained.

Use:

```powershell
python -m src.blower_inspection.cli tao-train BF-001 `
  --spec specs/visual_changenet/bf-001_segmentation.yaml `
  --dataset "C:\Users\Gigabyte\Downloads\Prepare-data\TAO_VCN_DATASET" `
  --results-dir data/results/BF-001/tao `
  --pretrained-model "C:\path\to\changenet_segment_levir_cd.pth"
```

The NVIDIA default spec points to a sample checkpoint under
`/results/pretrained/...`, which is not present automatically. Supply the real
downloaded `.pth` with `--pretrained-model`. If you deliberately accept training
without pretrained weights, replace that option with `--from-scratch`. When
neither is supplied, the CLI searches the documented host locations and fails
with a download command rather than using the missing example path.

The exact checkpoint filename is `changenet_segment_levir_cd.pth`; NVIDIA's NGC
model identifier encoded by the TAO 7.1 default directory name is
`nvidia/tao/visual_changenet_segmentation_levircd:visual_changenet_levircd_trainable_v1.0`. Install and authenticate
the NVIDIA NGC CLI, then let the application download and locate the nested file:

```powershell
python -m src.blower_inspection.cli tao-download-weights
```

The checkpoint is downloaded below `data/models/pretrained` and discovered
recursively, so its NGC-generated version subdirectory does not need to be
guessed. `tao-train` also searches that directory, the dataset parent, and the
current user's Downloads directory automatically. After downloading, the
training command can omit `--pretrained-model`:

```powershell
python -m src.blower_inspection.cli tao-train BF-001 `
  --spec specs/visual_changenet/bf-001_segmentation.yaml `
  --dataset "C:\Users\Gigabyte\Downloads\Prepare-data\TAO_VCN_DATASET" `
  --results-dir data/results/BF-001/tao
```

If `ngc` is not installed, obtain the official NVIDIA NGC CLI, configure its API
key, and rerun the download command. The launcher will never pretend that the
example `/results/pretrained/...` path exists.

An NGC `403 Access Denied` is not necessarily a bad API key. The earlier helper
used the shortened, nonexistent/inaccessible resource
`visual_changenet_levircd:trainable_v1.0`; NGC returned 403 while resolving that
model. The corrected identifier mirrors both components encoded in TAO's default
download directory. Verify access before downloading:

```powershell
ngc registry model info "nvidia/tao/visual_changenet_segmentation_levircd:visual_changenet_levircd_trainable_v1.0"
```

If NGC asks for terms to be accepted, complete that action in the NGC web UI and
retry. If this exact public identifier still returns 403, the account lacks
entitlement or NVIDIA has moved/retired the version; do not work around that by
putting the API key into source code.

**Security:** an API key visible in a screenshot, terminal transcript, or chat
must be treated as compromised. Revoke it in NGC, create a replacement, rerun
`ngc config set`, and remove/redact the screenshot. The application never needs
the key as a CLI argument and never stores it in this repository.

The screenshot shows two separate CLI validation errors:

* `tao-train BF-001` omitted mandatory `--spec` because TAO cannot train without
  dataset/model/optimizer settings;
* `tao-train --spec ...` omitted `model_id` in the previous CLI version.

The model ID is now optional and defaults to `active_model`, so either ordering
works after `tao-init` creates the file:

```powershell
python -m src.blower_inspection.cli tao-train BF-001 --spec specs/visual_changenet/bf-001_segmentation.yaml --dataset "C:\Users\Gigabyte\Downloads\Prepare-data\TAO_VCN_DATASET" --from-scratch
python -m src.blower_inspection.cli tao-train --spec specs/visual_changenet/bf-001_segmentation.yaml --dataset "C:\Users\Gigabyte\Downloads\Prepare-data\TAO_VCN_DATASET" --from-scratch
```

Passing `data/models/BF-001` to `--model-file` only passes a directory. It does
not ask TAO to train there. Pass the exported file itself, for example:

```powershell
python -m src.blower_inspection.cli train BF-001 --model-file `
  "C:\TAO\exports\bf001_visual_changenet.onnx"
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
