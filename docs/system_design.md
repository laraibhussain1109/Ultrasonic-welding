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

Every part number has an independent `.ckpt` file and a JSON metadata sidecar.
Old hybrid checkpoints are deliberately not loaded: every part model must be
retrained after this migration.

## Runtime decisions

The anomalib pixel anomaly map is resized to the YOLO crop, restricted to the
cylindrical inspection surface, cleaned morphologically, and evaluated for both
minimum defect area and bad fin-sector ratio. A failure is latched across all
tracked views of a rotating part. Production counters and ESP32 output are
updated only by the existing count-line/session state machine.

## Deployment controls

- Lock camera exposure, gain, focus, and lighting.
- Train from at least 20, preferably 100 or more, verified normal parts spanning
  all acceptable rotation and appearance variation.
- Validate each checkpoint against a versioned golden PASS/FAIL image set.
- Retrain when camera geometry, illumination, YOLO weights, or the part changes.
