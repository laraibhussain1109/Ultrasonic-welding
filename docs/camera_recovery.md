# Recovering slow acquisition and subtle rotation

The production screenshot isolates the bottleneck: **RAW/PROC/GUI 2.0 FPS,
READ 499.4 ms, INFER 0.0 ms, 1920×1080 YUY2 DSHOW**, with
`INSUFFICIENT_STATIONARY_FRAMES`. These are production GUI observations, not
a standalone raw benchmark. **Post-recovery physical FPS remains unmeasured.**
The cloud has no camera, motor, ESP32 or CUDA GPU. Synthetic tests cannot certify
Windows negotiation, fresh mechanical views or RTX 5070 execution.

Update/download `codex/fixed-six-view-patchcore` from PR #46, reinstall the edited
checkout if needed, and restart NeuroIris. Existing primary checkpoints and
calibration remain usable; this camera update requires no retraining. The original
GUI layout and camera settings control are retained. Select the desired native
resolution there; 4K is not silently replaced by 1080p.

## Measured camera mode selection

Windows startup probes native delivery in isolated lightweight processes:
DSHOW/MJPG, MSMF/MJPG and the two driver-default formats. If these are slow, it
also compares automatic exposure (DSHOW `.75`, MSMF `1`). Each sample warms up
five reads and measures two seconds; hung probes have an eight-second driver
timeout and an outer process-tree timeout. Startup can take over a minute when
many modes time out; the GUI remains responsive. A native matched mode measuring
at least 90% of requested FPS ends the search early; otherwise the highest
measured native rate wins. A driver's reported FPS never substitutes for delivered
FPS. Inspect the FPS tooltip/JSONL `format.mode_probes` and the startup log.
Short startup samples rank modes provisionally; repeat a longer standalone
measurement and observe sustained RAW FPS with a moving target.

FOURCC is requested before negotiation and reasserted **after width/height/FPS**,
because some drivers reset it to YUY2 when those properties change. The old forced
auto-exposure `0` is removed. Drivers may retain manual exposure; automatic trials
are selected only by measured native reads. Whether format negotiation or exposure
caused this station's slow mode still needs physical comparison. Capture
creation/read/release occur on the owning acquisition thread; probe processes
close their separate handles before the production handle opens. Wrong native
resolution on production reopen/read inhibits inspection.

To allow automatic comparison, remove previous process-specific pins before
starting the GUI (the model/artifact configuration is unaffected):

```powershell
Remove-Item Env:NEUROIRIS_CAMERA_BACKEND -ErrorAction SilentlyContinue
Remove-Item Env:NEUROIRIS_CAMERA_FOURCC -ErrorAction SilentlyContinue
Remove-Item Env:NEUROIRIS_CAMERA_AUTO_EXPOSURE -ErrorAction SilentlyContinue
$env:NEUROIRIS_CAMERA_AUTOSELECT = "1"
blower-inspection-ui
```

Explicit backend/FOURCC pins remain supported. With both pinned, exposure is
left untouched unless `NEUROIRIS_CAMERA_AUTO_EXPOSURE` is also set. Set
`NEUROIRIS_CAMERA_AUTOSELECT=0` only for a deliberately measured manual mode.

First establish raw FPS with processing disabled, as described in the
[performance guide](camera_performance.md). Compare exposure separately if needed:

```powershell
python -m blower_inspection.camera_benchmark --backends DSHOW MSMF --formats MJPG DEFAULT --resolutions 1920x1080 3840x2160 --seconds 10 --output camera-raw-default.json
python -m blower_inspection.camera_benchmark --backends DSHOW --formats MJPG --resolutions 3840x2160 --seconds 10 --auto-exposure 0.75 --output camera-raw-dshow-auto.json
python -m blower_inspection.camera_benchmark --backends MSMF --formats MJPG --resolutions 3840x2160 --seconds 10 --auto-exposure 1 --output camera-raw-msmf-auto.json
```

Use actual frame shape/FOURCC/backend, delivered FPS and read latency to compare.
Setters returning True do not establish support. The full default matrix also
tests YUY2/NV12. For the chosen mode, repeat a 30-second raw sample, then compare
motion/quality, inference and GUI stages using the same mode and lighting.

## Subtle motion and continuous evidence

Supplemental motion detection follows coherent corner tracks in individual axial
sections, with forward/backward match checks and exposure normalization. A bounded
four-frame thumbnail history accumulates slow/subpixel changes. The configured
difference/flow thresholds are unchanged; subtle local motion no longer has to
move 20% of the fin band. Exact stills/brightness-only changes bypass unnecessary
flow. Angle assignment follows observed moving/stopped transitions, never an
elapsed-time schedule. The station confirmed no PLC/ESP32 stop/index input.
Camera evidence cannot prove an entirely unobserved step or an encoder position.

**Continuous inspection no longer waits for a rotation event.** Native ROI frames
are quality-checked at up to five additional samples/s between stop bursts; valid
stationary quality results are reused too. Qualified images enter one latest
video slot, and the existing PatchCore/geometry inspector consumes them as fast
as it can. This is continuous sampled video, not a claim that every sensor frame
is inferred. Ordered six-stop stills have priority, and only one inference runs
at a time. Mandatory evidence is never overwritten by supplemental video.

Supplemental video has one pending native input and one qualified output. A
quality operation and active inference can each hold an additional image. A ROI
view can retain its native parent frame; memory remains bounded independently of
run length. Video overwrite/qualification counts appear in performance tooltips
and JSONL, separately from loss of mandatory evidence.

A calibrated, valid video FAIL latches reject and its entire affected axial
sections. Unlabelled video PASS/CANDIDATE results do not increment angle counts,
promote same-side recurrence into multiple views, or emit final PASS. The GUI
labels video coverage as unconfirmed. Removal/stop after unfinished fixed-view
inspection remains FAIL; invalid frames never become valid through this path.
At 2 FPS, a one-second pause cannot reliably supply three qualified photographs
after 200 ms settling. Video inference adds defect evidence, but raw acquisition
must improve before this cycle can be commissioned for PASS.

Training, artifact filenames/pairing, PatchCore calibration, geometry rules,
heatmap area tolerance and native quality thresholds remain unchanged. Firmware
outputs remain `/fail` latched reject, `/pass` clear reject and `/pass-pulse`
D5 HIGH for 500 ms. No firmware or result-output bridge changes are required.

## Physical validation

Keep the part stationary initially: qualified VIDEO results should appear even
before a rotation transition, with `0/6` angle coverage and no PASS pulse. Introduce
a reviewed defective part and confirm its calibrated FAIL latches the expected
section. Then run fit/home and six mechanical 60° stops. The existing 200 ms
settling and minimum-three-qualified-frame gates still apply.

The diagnostic sends no machine outputs. Replace the example ROI with the actual
native coordinates from the operator-approved ROI log. **Wait for “Camera ready”
before starting the fitting revolution**, since startup includes camera probing.

```powershell
python -m blower_inspection.camera_pipeline_benchmark --model-id BF-001 --width 3840 --height 2160 --roi 307 799 2957 475 --seconds 60 --output capture-only
python -m blower_inspection.camera_pipeline_benchmark --model-id BF-001 --width 3840 --height 2160 --roi 307 799 2957 475 --seconds 60 --inference --continuous --output capture-with-video
```

The second run separately records `video_checks` and mandatory `inspections`;
video checks never satisfy `six_valid_views` or `inference_complete`. Examine all
six native PNGs for sharpness, order, coverage and reflections. Try a slow model,
a deliberately missing/blurred stop and part removal. Confirm no incomplete case
emits PASS in the GUI, the reader continues during inference, the interface stays
responsive, and physical ESP32 outputs preserve their existing behavior.

| Production observation | Before recovery | After recovery |
| --- | --- | --- |
| GUI raw rate / read latency | 2.0 FPS / 499.4 ms | Awaiting physical run |
| Negotiated mode in screenshot | 1920×1080 YUY2 DSHOW | Awaiting measured native mode selection |
| Missing view | 0/6 valid; insufficient stationary frames | Awaiting physical six-stop audit |
| Video defect checks without strong motion | Not dispatched | Implemented; production accuracy pending |
| 4K raw OpenCV baseline, CUDA inference, outputs | Not measured here | Awaiting physical commissioning |

The [paired recovery stage report](performance/camera-recovery-cloud.json) uses
the same static synthetic 4K image/native ROI, 30 iterations and one OpenCV thread.
Mean legacy/recovery times were ROI copy **0.58/2.55 ms**, motion **9.20/4.35 ms**,
native quality **39.41/36.72 ms**, and Qt preview **55.35/10.32 ms**. The faster
motion sample benefits from the exact-still fast path; moving/noisy footage has
additional tracking work and must be profiled separately on the station. Shared
CPU scheduling explains variation in unchanged ROI/quality operations. These
stage measurements cannot be converted into a claimed physical camera FPS.

Validation: 87 focused recovery/capture/UI/training-contract checks passed.
The full regression run had 327 passes and the same nine failures as the prior
baseline (336 cases); the subsequently added continuous-audit check also passed.
The full-suite failures concern legacy Windows checkpoint pickles, existing
ROI/empty-fixture detection, dataset error text and six TAO expectations. UI
builders/login/stylesheet match `main` by AST, and offscreen desktop smoke passed.

Automated checks cover localized steps below both old global motion thresholds,
six ordered sharp stops, exposure-only drift/sensor noise, native-resolution
fallback rejection, measured-rate mode selection, continuous quality rejection,
reader independence, video fail latching and mandatory-still priority. These
are regression evidence, not certification of physical 360° coverage.
