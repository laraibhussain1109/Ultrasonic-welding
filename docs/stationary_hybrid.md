# Stationary six-view inspection in the existing NeuroIris UI

Run `blower-inspection-ui` from the repository root. The login, panels, controls,
styles, training workflow, camera settings, and YOLO selector are the ones from
`main`. Select the trained YOLOv12 checkpoint with **YOLO PART MODEL**. On the
production Windows PC, the supplied path is:

```
C:\Users\Gigabyte\Downloads\AMBER ULTRASONIC PART.v1i.yolov12\runs\detect\train\weights\best.pt
```

The existing PatchCore-primary checkpoint and its geometry calibration remain
the production model. This change does not require a different checkpoint
format or route production through the separate `train-fixed` tools.

## Capture and completion

After the initial continuous fitting revolution, the first stopped image locks
HOME without running defect inference. The next six moving/stopped transitions
are assigned 60°, 120°, 180°, 240°, 300°, and 360° in that order. Set
`skip_initial_fit_rotation: false` in `config/models.json` only for machines
without the separate initial fitting revolution; the first observed indexed
movement then leads to the 60° capture.

Motion evidence combines exposure-adjusted frame differences and optical flow
in the fin band. At each stop, the software confirms three stationary frames,
waits `stationary_settle_ms` (200 ms), collects `capture_burst_frames` (15; limited
to 10–30), and selects the sharpest frame passing blur, exposure, and glare
quality checks. The camera stream is used for capture; defect inference runs
only on the selected still. Similar-looking fins never prevent a new stop from
being counted. A long dwell cannot create multiple views of the same stop.

Each side is queued independently while the preceding still is being processed.
Slow inference cannot overwrite an earlier queued side. Duplicate angles cannot
increase coverage. An interrupted burst or invalid image consumes that physical
stop as an invalid attempt; it cannot shift the next angle or become a good view.
All six qualified stops are required for PASS. Six attempts with missing quality,
or premature part removal, finish as FAIL. The queue is drained before a departed
part is finalized.

The current ESP32 firmware provides result outputs, not motor-position feedback.
Camera counting therefore assumes the commissioned machine really makes these
60° steps and that the camera observes every transition. It does not measure an
absolute shaft angle or prove an unseen/reversed step. The visible, usable camera
arc must actually cover 60°; six captures alone cannot recover hidden surface.

## Heatmap area tolerance and hybrid checks

The calibrated raw PatchCore distance threshold creates the anomaly mask without
per-image min/max normalization. Padding, outer edges, fixture/support ribs, and
reflection regions are excluded from the inspectable pixels. Reflection pixels
remain eligible where local fin deformation provides corroboration. Components
smaller than `min_defect_area_px` are removed. Thin elongated components are
filtered as scratches unless geometry corroborates them; the defaults are
`scratch_max_width_px: 2` and `scratch_min_aspect: 8` in canonical heatmap pixels.
Set the width to zero to disable that scratch filter.

```
anomaly percentage = 100 × retained anomaly pixels / inspectable pixels
```

The original 1%, 3%, 5%, 8% buttons and MANUAL input now set this area threshold
directly. For example, a 2% retained anomaly passes a 3% tolerance and exceeds a
1% tolerance. A nonempty area equal to the threshold exceeds it; zero anomalies
pass even with zero tolerance. The percentage does not rescale the component-size
filter, nor is it a percentage of an image score or count of sections.

An above-tolerance heatmap plus local geometry corroboration, or strong calibrated
PatchCore evidence, fails immediately. Weaker above-tolerance evidence is a
candidate and requires recurrence in the same longitudinal section across views.
Geometry alone cannot override the operator's heatmap-area tolerance. Actual
localization and the raw heatmap remain available on the inspection result.

Confirmed section locations accumulate for the physical part. The live preview
highlights the entire longitudinal band, including after the flaw rotates out of
sight and after final FAIL. Locations clear only at confirmed part departure.
The live label remembers the captured angles for each confirmed section, and the
final log records failed angles and sections. The existing LAST RESULT panel
reports the current captured angle and measured area %.

## Existing firmware outputs

`firmware/esp32_fail_output/esp32_fail_output.ino` from `main` is unchanged:

| Result/action | HTTP route | Serial equivalent | Output |
| --- | --- | --- | --- |
| FAIL | `/fail` | `FAIL` | GPIO 4 active HIGH, latched |
| Clear reject | `/pass` | `PASS` / `RESET` | GPIO 4 inactive; no PASS pulse |
| Final PASS | `/pass-pulse` | `PASS_PULSE` | Clear reject and pulse D5 HIGH for 500 ms |

D5 uses the board's `D5` mapping when defined, otherwise GPIO 5. The Python UI uses
the asynchronous HTTP bridge, default `http://192.168.4.1`, configurable with
`NEUROIRIS_ESP32_URL`. There is exactly one PASS pulse after six qualified views,
and no pulse for an individual good view. A failure remains latched while the
part is present. No firmware change is required.

The cloud environment can validate the capture, mask, aggregation, UI wiring,
and output calls using synthetic images. Production accuracy and motion thresholds
still require the actual camera, existing model assets, and labeled GOOD/NG parts
on the Windows machine.

The companion gap detector now counts separate interrupted fins rather than
gradient rows, so one genuine break is retained. Broad reflections are masked
through their connected bright boundary to prevent that boundary from becoming
false broken-fin evidence. Requalify geometry calibration and area settings with
the real GOOD/NG set after updating.

Validation for this revision: 23 new regression tests pass, and the three existing
fin-gap/glare regressions now pass. Full suite: 242 passed, 9 pre-existing failures,
251 executed, no skips. The remaining failures concern legacy checkpoint loading,
ROI/presence helpers, dataset error text, and optional TAO expectations. Test details
were saved to `/workspace/onboarding/stationary-hybrid-tests.xml` in the cloud
workspace; this generated report is not a repository artifact.
