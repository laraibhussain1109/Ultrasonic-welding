# Windows camera throughput and commissioning

The production camera demonstrates 3840×2160 at 30 FPS in Windows Camera.
The reported NeuroIris rate before this change was approximately 1.2 FPS.
Those are operator observations, not raw OpenCV measurements. A later production
GUI shows RAW 2.0 FPS, READ 499.4 ms, INFER 0.0 ms and 1920×1080 YUY2 DSHOW.
See [camera recovery](camera_recovery.md) for measured startup mode selection,
exposure comparison and continuous inspection of subtle-motion footage.
**Standalone raw OpenCV baseline and post-recovery FPS remain unmeasured.** The cloud has no
physical camera and uses CPU-only PyTorch; synthetic tests cannot certify the
Windows driver, USB link, exposure timing, mechanical stops or RTX 5070.

## What profiling found

At the previous PR #46 head, `a576bde6`, `StationaryCameraWorker.run()` executed
`read → full ROI copy → motion preparation/Farneback → native quality → burst
selection` serially. Its FPS included all those operations. A slow quality check
therefore delayed the next camera read and could miss a short stop. Camera
opening and the initial ROI reads occurred in `app.py` on the GUI thread; the
live DirectShow object was then transferred to a different thread.

`stationary_capture.py` converted the entire native ROI to grayscale before
reducing it. Farneback already used a 320×64 image. `app.py` copied and annotated
the full camera image, converted/copied it again, and created a full-resolution
QImage/QPixmap before scaling to the viewer. Inference was already a separate
QThread; it must remain independent of acquisition.

An initial cloud profile of a synthetic 3840×2160 BGR image, native 2957×475 ROI,
30 iterations and four OpenCV threads measured mean ROI copy 0.54 ms, motion
8.17 ms, native quality 37.86 ms, and full-resolution Qt preview 41.51 ms.
This identifies blocking CPU work, but does not account for the physical
camera's open/read/decode latency. Driver/format/exposure causes of the observed
1.2 FPS still require the raw benchmark below.

## Acquisition and evidence storage

Camera open, every read, negotiated-property queries and release now execute
on the acquisition QThread. `USBCamera` rejects cross-thread open/read/release.
The reader publishes one latest native-frame reference and enqueues ordered
processing references; it does no ROI copy, motion, quality, inference or Qt work.

A motion thread crops by reference, reduces the fin band to at most 320×64,
converts that thumbnail to grayscale and applies the existing flow/difference
thresholds plus localized corner tracking. A separate quality thread checks stopped, settled candidates and supplemental video
at native ROI resolution. Candidate copies are shared by the pending quality
check and burst. Quality futures never count as qualified until completed.
Final selection uses the same sharpest-qualified-still rule in synchronous
and asynchronous paths. Valid and invalid results share an ordered FIFO so a
later invalid stop cannot overtake an earlier burst awaiting quality checks.

Application buffers have explicit limits:

| Buffer | Item limit | Native-image byte limit | On saturation |
| --- | ---: | ---: | --- |
| Latest preview | 1 | One actual camera frame | Overwrite preview only; increment counter |
| Ordered motion input | 64 | 512 MiB | Stop/inhibit inspection |
| Quality jobs/burst finalization | 48 | 256 MiB | Stop/inhibit inspection |
| Worker selected stills | 12 | 256 MiB | Stop/inhibit inspection |
| GUI/inference waiting stills | 12 | 256 MiB | Stop/inhibit inspection |

At 4K BGR, the motion byte limit admits 21 queued full frames, regardless of
the larger item limit. The existing burst limit remains at most 30 native ROI
frames; it is finite as well. Active operations, the source/preview frame and
model memory are additional. These are application buffer limits, not total
RAM/VRAM guarantees; a driver may ignore `CAP_PROP_BUFFERSIZE`.

The six angles, initial fitting rotation, three stationary observations,
200 ms settling from the first stable read timestamp, configured burst target,
350 ms preferred window, minimum three qualified production frames, native
quality thresholds and invalid-stop behavior remain intact. Motion timestamps
come from acquisition, not delayed processing time. OpenCV does not provide
reliable exposure timestamps for every live Windows driver; the timestamp
before read remains the existing conservative software reference. A driver
that buffers stale images must be caught during physical commissioning.

Preview drawing, RGB conversion and Qt allocation use a resized image. Only
display images use linear preview interpolation. Inspection retains the native
ROI pixels. GUI panels, controls, stylesheet, model/training/calibration files,
geometry/anomaly/tolerance rules and ESP32 result outputs are unchanged.

## 1. Establish the raw OpenCV baseline first

Activate the installed environment in the repository root. Close Windows Camera
and other applications holding this camera, then run:

```powershell
python -m blower_inspection.camera_benchmark --index 0 --seconds 10 --output camera-raw-default.json
python -m blower_inspection.camera_benchmark --index 0 --seconds 10 --auto-exposure 0 --output camera-raw-production.json
```

The default matrix probes DirectShow and Media Foundation at 1920×1080 and
3840×2160 with driver default, MJPG (**MJPEG**), YUY2 and NV12. OpenCV has no
portable supported-format enumerator: probing reports accepted settings and
the actual delivered shape/backend/FOURCC. A probe that falls back to another
resolution or format is not evidence of support for the requested combination.
Unknown FOURCC remains unknown. Each probe runs in a separate process, with a
timeout for unsupported formats that hang inside the driver.

The first run leaves exposure untouched. The second reproduces the previous
production request for auto-exposure=0 for comparison. Exposure values/accepted properties are
reported because backend-specific manual/default exposure can limit frame rate.
Recovery no longer forces that setting; model calibration remains unchanged.
If these two runs differ, inspect the camera's driver settings and measure again
before choosing an operating mode.

Each successful probe reports delivered frames/elapsed time, read mean/p50/p95/
maximum latency, actual resolution, FOURCC, backend, failed reads and delivery
gaps longer than 1.5 requested frame periods. Driver frame-position gaps are
reported only when that property actually increases; zero/unsupported positions
and timestamps remain unavailable. OpenCV positions can be delivered-frame
counters, not sensor sequence numbers. Timing gaps and position gaps are
evidence, not a certified sensor-drop count. Static identical pictures also
cannot prove the camera is exposing fresh frames; verify with a moving target.
`CAP_PROP_FPS=30` and a successful property setter never establish 30 FPS.

Repeat the winning matched mode for a longer sample, for example:

```powershell
python -m blower_inspection.camera_benchmark --backends MSMF --formats MJPG --resolutions 3840x2160 --seconds 30 --output camera-raw-4k.json
```

Use the existing **CAMERA FPS / RESOLUTION** control to select the desired native
resolution. The tracked profile defaults are still 1920×1080. Backend/format
can be selected for this process without changing model artifacts:

```powershell
$env:NEUROIRIS_CAMERA_BACKEND = "DSHOW"  # or MSMF after measuring it
$env:NEUROIRIS_CAMERA_FOURCC = "MJPG"   # or another measured FOURCC / DEFAULT
```

## 2. Measure processing and inference separately

Use a saved native camera still and the actual approved ROI coordinates from
the inspection log. The coordinates below are an example at 4K; replace them
with this station's native coordinates:

```powershell
python -m blower_inspection.pipeline_profile --image "C:\path\to\camera-still.png" --roi 307 799 2957 475 --iterations 30 --legacy --output stages-before.json
python -m blower_inspection.pipeline_profile --image "C:\path\to\camera-still.png" --roi 307 799 2957 475 --iterations 30 --model-id BF-001 --output stages-after.json
```

`--legacy` reproduces the previous full-ROI motion preparation and full-resolution
Qt path for an offline comparison. ROI copying, motion, native blur/exposure/
glare checks, Qt preview and optional production inference are timed separately.
It does not change or retrain a model. `--opencv-threads 1` can make comparisons
more consistent on a CPU-limited diagnostic machine; it only affects that tool.
`--synthetic` explicitly labels generated data and cannot certify a camera.

Verify GPU support with the installed Windows interpreter:

```powershell
python -c "import torch; print('torch', torch.__version__); print('CUDA', torch.cuda.is_available()); print('GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('architectures', torch.cuda.get_arch_list())"
python -m blower_inspection.cli doctor
```

PatchCore already chooses CUDA when available and moves its backbone, input
features and memory bank to that device. An explicit `BLOWER_INSPECTION_DEVICE`
override still takes precedence. Confirm the runtime DEVICE entry and an actual
inference run on the RTX 5070, including support for its GPU architecture.
The raw benchmark imports neither PyTorch nor Qt and does not require CUDA.

## 3. Audit physical six-stop capture, then the full GUI

Start the diagnostic before the fitting revolution, using the same native ROI.
Wait for the diagnostic's “Camera ready” message, then run the machine's existing fit/home plus six 60° stops, each with its one-second
pause:

```powershell
python -m blower_inspection.camera_pipeline_benchmark --model-id BF-001 --roi 307 799 2957 475 --width 3840 --height 2160 --seconds 60 --output camera-six-stop-audit
python -m blower_inspection.camera_pipeline_benchmark --model-id BF-001 --roi 307 799 2957 475 --width 3840 --height 2160 --seconds 60 --inference --output camera-six-stop-inference
```

Run the mechanical sequence afresh for each command. These diagnostics use the
profile's existing rules and send no ESP32 outputs. They save 060.png through
360.png at native ROI resolution, validity/sharpness/burst evidence, actual
camera format, per-stage rate/latency samples and optional inference results.
They return failure for missing/invalid captures, queue faults or incomplete
requested inference. Confirm the actual camera resolution matches the ROI;
resolution fallback is rejected. Native still acquisition occurs during stops;
quality/inference completion may occur later without delaying camera reads.

Then start the normal interface and enable one JSON sample per GUI tick:

```powershell
$env:NEUROIRIS_PERFORMANCE_LOG = "$PWD\camera-gui-after.jsonl"
blower-inspection-ui
```

The original FPS header now reports raw completed-read FPS. The existing LAST
RESULT panel shows RAW, PROC and GUI rates, mean READ and INFER latency, dropped
evidence, overwritten previews and actual resolution/compression/backend.
Hover that panel or the FPS header for quality FPS/latency, motion latency,
read p95, render latency and queue depth/peak details. PROC counts motion input
frames; quality counts stationary candidates and supplemental video. GUI counts frame updates,
not monitor refreshes. INFER measures the entire production `inspect()` call,
including PatchCore/registration/quality/geometry; it is not GPU kernel time.
Rates use completed operations and wall-clock timestamps and decay during stalls.
Preview overwrites are expected when display is slower than acquisition; losing
ordered inspection evidence is a fault.

Compare raw FPS with motion+quality, with inference added, and with the GUI added,
using the same measured camera mode, lighting, exposure, ROI and test duration.
Inspect all six saved images for sharpness, angle order, whole-component coverage
and glare. Test one missing/blurred stop and a slow inference: neither may produce
the final PASS pulse. Verify the GUI responds while the camera keeps reading and
confirm the unchanged ESP32 PASS/FAIL pulses on the physical station.

## Before/after evidence and outstanding results

The paired [cloud stage report](performance/camera-cloud-comparison.json) uses
the same synthetic 4K still/native ROI, 30 iterations and one OpenCV thread:

| Cloud CPU stage, mean ms (initial throughput change) | Previous path | `92ba748` path |
| --- | ---: | ---: |
| ROI copy (standalone measurement) | 0.93 | 1.48 |
| Motion preparation + flow | 12.37 | 21.78 |
| Native quality (same analyzer) | 68.73 | 55.77 |
| Resize/color/Qt preview | 110.62 | 7.36 |
| Motion/quality blocking the reader | Both stages inline | Neither stage inline |

This shared-cloud sample demonstrates the cheaper preview allocation/conversion
path. It does **not** establish faster motion processing or quality processing;
color-area resizing has a different cost and shared CPU scheduling varies.
Native quality code and thresholds are unchanged. Those stages are independent
of reads now. Do not turn these stage timings into a claimed camera FPS.

| Physical result | Before | After |
| --- | --- | --- |
| Windows Camera, 4K | Operator reports 30 FPS | Reference observation |
| NeuroIris displayed FPS | Operator reports ~1.2 | Awaiting physical run |
| Raw OpenCV 4K FPS/read latency/format | Not measured | Awaiting raw benchmark |
| Integrated 4K acquisition vs raw baseline | Not measured | Awaiting physical run |
| Six one-second stops, CUDA inference, responsive GUI, ESP32 outputs | Earlier capture failures reported | Awaiting physical commissioning |

Automated tests exercise native 4K input with six ordered one-second stops and
the 200 ms settling/minimum-three gate, blocked quality and inference, reader
ownership, preview sizing, raw negotiation/FPS misreporting, deferred valid/invalid
ordering, bounded-byte/count overflow and reject behavior. Their camera is paced
synthetic playback; no hardware FPS or production accuracy claim follows.

```powershell
python -m pytest -q tests/test_camera_recovery.py tests/test_camera_throughput.py tests/test_stationary_acquisition.py tests/test_stationary_hybrid.py tests/test_stationary_main_ui.py tests/test_app_source_contract.py tests/test_production_training_contract.py
```
