# Step-by-step: train a model and start inspection

This guide covers the default production workflow: **YOLO localization → stable
ROI → frame-quality gate → PatchCore + fin geometry → temporal confirmation**.
TAO is optional engineering functionality and is not required to train or run
the default production detector.

## 1. One-time installation

1. Install Python 3.10 or newer, Git, the qualified NVIDIA driver, and CUDA
   support appropriate for the PyTorch build on the inspection PC.
2. Open a terminal in the repository root.
3. Create and activate a virtual environment:

   **Linux:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   **Windows PowerShell:**

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install the application and industrial dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -e '.[industrial,dev]'
   ```

   In PowerShell, use double quotes: `pip install -e ".[industrial,dev]"`.
5. Confirm that the configured models load:

   ```bash
   blower-inspection list-models
   ```

6. Confirm that PyTorch can see the NVIDIA GPU:

   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
   ```

   CPU operation is useful for setup, but final throughput must be qualified on
   the deployment GPU.

## 2. Configure a blower model

Each blower type has one entry in `config/models.json`. Before collecting data,
verify at least:

1. `normal_image_dir` points to the known-good full-camera training images.
2. `patchcore_model_file` and `patchcore_calibration_file` point to writable
   artifact locations unique to that blower model.
3. `yolo_model_path` points to the Ultralytics `best.pt` that detects the entire
   blower. It must not point to a PatchCore or TAO file.
4. Camera width, height, FPS, and `yolo_confidence` match the production setup.
5. Keep `production_algorithm` set to `patchcore_geometry` and
   `engineering_compare_tao` set to `false` for production.
6. For a fixed fixture, use `roi_mode: fixed_after_detection` and
   `lock_roi_after_confirmation: true`. Use `yolo_stabilized` only when the part
   position genuinely moves.

The UI can persist the YOLO checkpoint and camera settings with **YOLO PART
MODEL** and **CAMERA SETTINGS**. Restarting training after changing the camera,
YOLO model, ROI, backbone layers, or PatchCore settings is required; do not reuse
the old calibration.

## 3. Collect model-training images

1. Put only physically known-good blowers on the production fixture.
2. Use the same camera, lens, focus, working distance, lighting, exposure,
   resolution, and fixture that inspection will use.
3. Capture full camera frames while the blower rotates.
4. Include multiple physical good blowers, circumferential phases, acceptable
   production variation, and acceptable minor reflections.
5. Do not capture every adjacent video frame. Diverse views are more useful than
   hundreds of nearly identical frames.
6. Collect at least 20 source images; 100+ diverse qualified images across
   several physical parts are strongly recommended for line qualification.
7. Copy them into the model's `normal_image_dir`, for example:

   ```text
   data/training/BF-001/normal/
     good_part_01_view_001.png
     good_part_01_view_010.png
     good_part_02_view_003.png
     ...
   ```

Do not include defects, partial blowers, hands, tools, severe blur, clipped
exposure, or uncontrolled glare. Training automatically performs YOLO cropping,
quality rejection, and duplicate reduction; source images are not modified.

## 4. Train and calibrate the PatchCore model

### Command-line method

1. Activate the virtual environment and run from the repository root.
2. Start training for the selected model:

   ```bash
   blower-inspection train BF-001
   ```

   Equivalent module command:

   ```bash
   python -m blower_inspection.cli train BF-001
   ```

3. Watch the progress messages. Training performs, in order:
   - YOLO localization of every full-camera image;
   - canonical aspect-preserving ROI generation;
   - blur, exposure, saturation, and glare quality checks;
   - duplicate/redundancy filtering;
   - a disjoint training/calibration split;
   - WideResNet PatchCore embedding and coreset memory-bank creation; and
   - fixed raw-distance threshold calibration on images excluded from the bank.
4. Confirm these artifacts exist:

   ```text
   data/models/BF-001/patchcore_primary.pt
   data/models/BF-001/patchcore_calibration.json
   data/models/BF-001/patchcore_primary.training_report.json
   ```

5. Read the training report. It lists source, accepted, training, calibration,
   duplicate, and rejection counts. If too few images survive, correct image
   quality or YOLO localization and capture more data; do not lower all quality
   limits merely to force training to finish.

### Desktop UI method

1. Start the UI with `blower-inspection-ui` or
   `python -m blower_inspection.app`.
2. Log in as an administrator. The development login is `admin` / `admin123`;
   change it before deployment.
3. Select the required blower model.
4. Set **YOLO PART MODEL** to the correct `best.pt` if it is not already saved.
5. Verify **CAMERA SETTINGS**.
6. Click **TRAIN PATCHCORE MODEL**.
7. Wait for `TRAINED <model-id>` in the event log. Do not start live inspection
   while model artifacts are still being written.

## 5. Validate before production use

1. Keep calibration images physically independent from memory-bank training
   where possible. The trainer enforces an image-level disjoint split, but the
   commissioning dataset should also be split by physical blower.
2. Put difficult known-good images in the optional `hard_good_dir`. They are for
   false-reject validation and are not automatically added to the memory bank.
3. Test an independent labeled set of GOOD and NG parts.
4. Record GOOD tested, false rejects, False Reject Rate, NG tested, false
   accepts, False Accept Rate, precision, recall, and F1.
5. Include broken, missing, tilted, and bent fins at the smallest actionable
   size, plus acceptable glare and surface variation.
6. Do not enable the ESP32/PLC reject output until thresholds, camera settings,
   view coverage, latency, and repeatability have passed the plant's approval
   process.

## 6. Start a live inspection

1. Mount and connect the production camera. Lock focus, exposure, gain, white
   balance, resolution, and FPS to the qualified values. Shorter exposure and
   stronger diffuse lighting are preferred to software deblurring.
2. Connect/configure the ESP32 only after offline validation. The UI works with
   the output disabled during commissioning.
3. Activate the virtual environment and start the UI:

   ```bash
   blower-inspection-ui
   ```

   or:

   ```bash
   python -m blower_inspection.app
   ```

4. Log in, select the trained blower model, and verify that the model ID matches
   the physical part.
5. Click **START INSPECTION**.
6. For a fixed station, YOLO proposes a yellow complete-blower ROI. Confirm that
   it includes the whole blower and excludes hands/tools. Choose **No** to capture
   again or **Cancel** to stop; never accept a partial box.
7. Start blower rotation. The UI collects a burst, rejects poor frames, selects
   the sharpest qualified frame, and waits for structurally distinct rotational
   phases. `WAITING FOR ROTATION` means the blower must rotate farther; it is not
   an error.
8. Keep the part in place until the required qualified-view count is reached.
   Blurry/invalid views do not count. If no qualified views are obtained, the
   result is a quality fault/recheck, never a silent PASS.
9. Interpret operator states:
   - **VIEW PASS**: the current qualified view is normal; the part is not final yet.
   - **INSPECTING**: more rotational confirmation is required.
   - **VIEW INVALID**: correct blur, exposure, crop, or obstruction and retry.
   - **FAIL LATCHED**: confirmed evidence has rejected the current physical part.
   - **PASS / FAIL**: final result after the required qualified views.
   - **SYSTEM FAULT**: inspection stopped fail-closed; resolve the logged model,
     calibration, camera, or runtime fault before restarting.
10. Remove the completed part only after final PASS/FAIL. The daily statistics,
    result record, and configured ESP32 output update at part completion.

## 7. Inspect one saved image for setup checks

This is useful for diagnostics, but it does not replace rotational confirmation:

```bash
blower-inspection inspect BF-001 path/to/full_camera_image.png
```

Add `--esp32-output` only during intentional output testing. A saved-image result
must not be treated as a production part decision because production requires
multiple qualified rotational views.

## 8. Common startup failures

- **`No yolo_model_path is configured`**: select/save the correct YOLO `best.pt`.
- **Too few qualified training images**: inspect the training report, correct
  blur/exposure/glare/crop problems, and add diverse known-good views.
- **Stale PatchCore calibration**: retrain after changing the memory bank,
  backbone layers, or relevant settings. Never copy an older calibration over a
  new model.
- **`VIEW INVALID` / `MOTION_BLUR`**: reduce exposure time, increase diffuse
  light, reduce rotation speed if the process allows, and verify focus.
- **`WAITING FOR ROTATION`**: the new image is too similar to an already accepted
  phase; rotate farther.
- **No blower detected**: verify the correct YOLO checkpoint, full-part framing,
  confidence threshold, camera connection, and illumination.
- **Persistent edge detections**: verify that the accepted ROI contains the full
  blower and tune `patchcore_edge_ignore_ratio` only with a labeled validation
  set.
- **System fault after model/settings change**: read the event log and retrain so
  the checkpoint and calibration hashes/settings agree.

## 9. Stop inspection safely

1. Let the current part reach a final decision or remove it and handle the
   resulting insufficient-view fault according to the line procedure.
2. Click **STOP CAMERA** before changing the model, camera, ROI, or lighting.
3. Confirm the reject/inhibit output returns to the plant-defined safe idle state.
4. Close the application only after the camera and output have stopped.

