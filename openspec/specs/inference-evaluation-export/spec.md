# Inference, Evaluation, and Export Specification

## Purpose

Current checkpoint-backed sliced inference, full-frame evaluation, and
Ultralytics model export.

## Requirements

### Requirement: Config-sourced model input
Inference, evaluation, and export SHALL currently obtain their model artifact
from `model.weights`; inference SHALL obtain imagery from `data.source_path`,
while evaluation SHALL treat `data.source_path` as a prepared dataset root.

#### Scenario: Evaluation without weights
- **WHEN** `sati eval` runs without `model.weights`
- **THEN** it fails before inference begins

### Requirement: Sliced inference
Inference SHALL build each frame's level stack, emit slices through the same
pixel path used by preparation, predict in batches, recombine detections, and
write final full-resolution detections.

#### Scenario: Slice prediction cache disabled
- **WHEN** `results.cache_slice_predictions` is false
- **THEN** result rows omit raw `slice_prediction` records but retain
  `level_merge`, `final_merge`, and `frame_metadata`

### Requirement: Full-frame evaluation
Evaluation SHALL run inference over the prepared `validation` split and score
only `final_merge` detections against full-frame, full-resolution ground truth.

#### Scenario: Evaluation completes
- **WHEN** validation inference and scoring succeed
- **THEN** `validation_results.parquet`, `_metadata.json`, and `metrics.json`
  are written directly under `output.run_dir`

### Requirement: Scale diagnostics
Evaluation SHALL report overall and per-class COCO-style AP metrics plus
per-contributing-level prediction statistics and ground-truth recall binned by
object long side.

#### Scenario: Per-level output
- **WHEN** a final detection records multiple source levels
- **THEN** it contributes to each source level's diagnostic count

### Requirement: Model export sidecar
Export SHALL invoke the backend's requested format and write
`inference_config.yaml` under `output.run_dir` with slicing, recombination,
model, class-map, and hash-family information.

#### Scenario: Non-local checkpoint
- **WHEN** `model.weights` does not name an existing local file
- **THEN** export fails before invoking the backend
