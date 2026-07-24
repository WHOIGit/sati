## ADDED Requirements

### Requirement: Explicit consumed artifact options
Inference, evaluation, and export SHALL require `--checkpoint PATH`, and
inference SHALL require `--input PATH`; these options SHALL be command I/O and
SHALL NOT mutate the authored experiment config.

#### Scenario: Ad-hoc inference
- **WHEN** a caller runs
  `sati infer experiment=manual --checkpoint best.pt --input ./new-images`
- **THEN** the checkpoint and imagery are recorded in invocation provenance
  while model initialization and raw dataset source config remain unchanged

### Requirement: Checkpoint contract reconstruction
Inference, evaluation, and export SHALL reconstruct model, class, target, and
preprocessing defaults from the checkpoint's embedded
`sati_inference_contract` and SHALL fail before a forward pass when the
contract is absent or invalid.

#### Scenario: Legacy development checkpoint
- **WHEN** a checkpoint lacks `sati_inference_contract`
- **THEN** the command fails with an instruction to produce a new canonical
  SATI checkpoint

### Requirement: Recombination override precedence
Infer/eval config or inline overrides SHALL be allowed to change recombination
and result policy from the checkpoint defaults but SHALL NOT change the
checkpoint's model architecture, class mapping, target schema, or
preprocessing contract.

#### Scenario: Post-hoc IoU choice
- **WHEN** evaluation overrides `recombination.cross_level_iou`
- **THEN** it uses the new value and records a new final-merge
  `recombination_hash` while all other checkpoint compatibility hashes remain
  unchanged

### Requirement: Inference output envelope
Inference SHALL write `results/predictions.parquet` and `_metadata.json` under
`infer_outputs.dir`, together with the common config bundle and lifecycle
manifest.

#### Scenario: Inference completes
- **WHEN** sliced inference succeeds on ad-hoc imagery
- **THEN** no file named `validation_results.parquet` is used for that
  non-validation input

### Requirement: Exported model identity
Export SHALL compute `model_hash` from exported model bytes, retain the source
`checkpoint_hash`, and place the model plus a portable
`inference_config.yaml` contract under `export_outputs.models.dir`.

#### Scenario: Same checkpoint in two formats
- **WHEN** one checkpoint is exported to ONNX and another backend format
- **THEN** both exports share `checkpoint_hash` but have independent
  `model_hash` values

## MODIFIED Requirements

### Requirement: Config-sourced model input
Model initialization SHALL remain configuration, but a completed model
consumed by inference, evaluation, or export SHALL come exclusively from the
command's `--checkpoint` option; evaluation SHALL read
`data.prepared.uri`, and inference imagery SHALL come from `--input`.

#### Scenario: Authored initialization checkpoint
- **WHEN** `model.weights.source=checkpoint` is configured for a new training
  run and `sati eval --checkpoint evaluated.pt` is invoked
- **THEN** evaluation consumes `evaluated.pt`, not the training initialization
  URI

### Requirement: Full-frame evaluation
Evaluation SHALL run checkpoint-backed inference over the selected prepared
split, score only `final_merge` detections against full-resolution truth, and
write `results/validation_results.parquet`, `results/_metadata.json`, and
`metrics/metrics.json` under `eval_outputs.dir`.

#### Scenario: Validation evaluation completes
- **WHEN** evaluation and scoring succeed
- **THEN** its config provenance, result table, metadata, metric summary, and
  lifecycle status are discoverable from the eval run root

### Requirement: Model export sidecar
Export SHALL invoke the requested backend format from a canonical checkpoint
and SHALL write an `inference_config.yaml` that contains the complete portable
inference contract plus export format, artifact filename, source
`checkpoint_hash`, and exported `model_hash`.

#### Scenario: Export outside source run
- **WHEN** a canonical checkpoint has been moved away from its source run
- **THEN** export succeeds when `export_outputs.dir` can be resolved explicitly
  and does not require the old run's `resolved.yaml`

## REMOVED Requirements

### Requirement: Completed checkpoint in `model.weights`
**Reason**: `model.weights` describes training initialization; overloading it
with a consumed run artifact obscures provenance and conflicts with portable
checkpoint reconstruction.

**Migration**: Pass canonical completed checkpoints with `--checkpoint`.

### Requirement: Ad-hoc inference imagery in `data.source_path`
**Reason**: Ad-hoc imagery is invocation input, while configured data describes
reusable experiment datasets.

**Migration**: Pass inference imagery with `--input`.
