# Industrial inspection system design

## Detection and region of interest

Ultralytics YOLO is the single source of component localisation in both training
and inference. Training images may contain the complete camera field of view;
immediately before training, each image is passed through the configured YOLO
model and only its exact component bounding box is staged for anomaly learning.
At runtime YOLO and ByteTrack locate the same component crop and associate all
rotation views with one physical part.

Using identical localisation on both paths prevents the anomaly model from
learning table, fixture, or camera-position shortcuts and gives the weld surface
the full input resolution.

## SuperSimpleNet anomaly model

The deployed anomaly detector is anomalib **SuperSimpleNet**. It replaces the
PatchCore/PaDiM memory-bank ensemble, whose nearest-neighbour scores generalized
poorly to valid surface, lighting, and rotation variation. SuperSimpleNet learns
a discriminative normal/anomalous boundary using synthetic anomalies generated
from normal component crops. This removes the runtime patch memory bank and its
sensitivity to coreset coverage.

Every part number has an exported `.pt` inference model, a retained Lightning
`.ckpt` training checkpoint, and a JSON metadata sidecar. `TorchInferencer`
loads only the exported `.pt` model. Checkpoints created by the first
SuperSimpleNet release are automatically exported on first inspection, so a
completed 100-epoch training run does not need to be repeated. Old hybrid
checkpoints are deliberately not loaded.

## Runtime decisions

The anomalib pixel anomaly map is resized to the YOLO crop, restricted to the
cylindrical inspection surface, and converted from its broad surface baseline
to a localized defect score. A fixed minimum contrast span prevents sensor noise
from being amplified when the map is nearly uniform. The localized map is then
cleaned morphologically and evaluated for both minimum defect area and bad
fin-sector ratio. OpenCV BGR crops are explicitly converted to RGB before
anomalib inference so live preprocessing matches training. A failure is latched
across all tracked views of a rotating part. Production counters and ESP32
output are updated only by the existing count-line/session state machine.

A connected anomaly response covering more than 50% of the inspected surface is
classified as broad model/background drift and is not drawn as a physical
defect. This prevents normal fin edges from joining into one component-sized red
contour. The independent fin-continuity detector runs after this guard, retaining
localized broken-fin evidence even when a broad neural response is suppressed.
The continuity detector bridges missing spans up to approximately one support
bay (14% of crop width) and only suppresses gap columns repeated across many fin
rows; this lets a single complete broken-fin bay survive the geometry filters.

The operator tolerance buttons represent allowed anomalous surface coverage.
At 0%, any confirmed region that passes the minimum noise-area filter fails;
1% fails once confirmed defect pixels cover 1% of the component; 3% fails at 3%;
and 20% allows up to 20%. They do not change neural confidence, so selecting a
tolerance has a predictable monotonic effect. The separate minimum pixel count
continues to reject isolated noise even at 0%.

Pixel selection uses anomalib's exported `pred_mask`, including the threshold
calibrated by the trained model. The raw heatmap is retained for score display,
but it is no longer binarized with an application-defined cutoff that can select
normal high-contrast component edges instead of a lower-contrast physical fault.

## Boundary controls

Anomaly scoring excludes the outer 12% at each end of the YOLO component crop.
The remaining scoring mask is eroded inward by 8 pixels before either neural or
structural scores are evaluated. This prevents receptive fields and mask pixels
from straddling the component/background boundary, and removes the optically
unstable molded tips from PASS/FAIL coverage. These defaults are configurable per
part as `scoring_end_exclusion_ratio` and `scoring_mask_erosion_px`.

YOLO confidence cannot by itself remove a jagged anomaly contour: YOLO supplies
the rectangular component crop, while the red contour comes from anomaly-mask
pixels. Nevertheless, the YOLO training set should include all valid tip poses,
lighting conditions, and rotations so its bounding box remains stable. The
SuperSimpleNet normal set should likewise contain acceptable edge and near-edge
appearance variation; SuperSimpleNet does not use a PatchCore memory bank, but
normal-set coverage still determines what appearance it learns as acceptable.

## Deployment controls

- Lock camera exposure, gain, focus, and lighting.
- Train from at least 20, preferably 100 or more, verified normal parts spanning
  all acceptable rotation and appearance variation.
- Validate each checkpoint against a versioned golden PASS/FAIL image set.
- Retrain when camera geometry, illumination, YOLO weights, or the part changes.
