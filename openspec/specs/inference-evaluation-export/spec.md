# Inference, Evaluation, and Export Specification

## Purpose

Current checkpoint-backed inference, evaluation, and model export.

## Requirements

### Requirement: Explicit consumed artifact options
Infer, eval, and export SHALL require `--checkpoint`; infer SHALL require
`--input`. These are command I/O and SHALL not mutate experiment config.

#### Scenario: Ad-hoc inference
- **WHEN** `sati infer experiment=manual --checkpoint best.pt --input images`
  runs
- **THEN** command provenance records the inputs while experiment policy is
  unchanged

### Requirement: Config-sourced model input
Model initialization SHALL remain config, but a completed model consumed by
infer/eval/export SHALL come only from `--checkpoint`; eval SHALL use
`data.prepared.uri` and infer imagery SHALL come only from `--input`.

#### Scenario: Initialization checkpoint is configured
- **WHEN** eval receives a different explicit checkpoint
- **THEN** eval consumes the explicit completed artifact

### Requirement: Checkpoint contract reconstruction
Infer, eval, and export SHALL reconstruct model, class, target, and
preprocessing defaults from `sati_inference_contract`, failing before a
forward pass when absent or invalid.

#### Scenario: Legacy checkpoint
- **WHEN** a checkpoint lacks the contract
- **THEN** the command requests a new canonical SATI checkpoint

### Requirement: Recombination override precedence
Infer/eval SHALL accept changes to recombination and result policy, including the
merge method, the match metric, the matching mode, and either stage threshold,
and SHALL reject changes to model architecture, class mapping, target schema, or
preprocessing.

#### Scenario: Post-hoc threshold choice
- **WHEN** the cross-level threshold is overridden
- **THEN** final recombination identity changes while compatibility identities
  remain fixed

#### Scenario: Post-hoc strategy choice
- **WHEN** the merge method or match metric is overridden at infer/eval time
- **THEN** the override is accepted, recorded on merge rows, and moves only the
  recombination identity

### Requirement: Sliced inference
Inference SHALL use the preparation pixel path, predict slices, recombine, and
write full-resolution detections.

#### Scenario: Slice cache disabled
- **WHEN** slice-prediction caching is false
- **THEN** tall results omit raw slice rows and retain merged/frame rows

### Requirement: Inference output envelope
Inference SHALL write `results/predictions.parquet` and `_metadata.json` under
`infer_outputs.dir`, with config provenance and a lifecycle manifest.

#### Scenario: Inference completes
- **WHEN** ad-hoc inference succeeds
- **THEN** it does not use the validation-table filename

### Requirement: Full-frame evaluation
Evaluation SHALL infer over a prepared split, score only full-resolution
`final_merge` rows, and write `results/validation_results.parquet`,
`results/_metadata.json`, and `metrics/metrics.json` under `eval_outputs.dir`.

#### Scenario: Evaluation completes
- **WHEN** evaluation succeeds
- **THEN** config, results, metrics, and lifecycle state are discoverable from
  the eval root

### Requirement: Scale diagnostics
Evaluation SHALL report overall/per-class COCO metrics, per-contributing-level
statistics, and recall binned by object long side.

#### Scenario: Multi-level source
- **WHEN** a final detection records multiple source levels
- **THEN** it contributes to each corresponding diagnostic count

### Requirement: Exported model identity
Export SHALL compute `model_hash` from exported bytes, retain
`checkpoint_hash`, and place the model and portable `inference_config.yaml`
under `export_outputs.models.dir`.

#### Scenario: Two export formats
- **WHEN** one checkpoint is exported twice
- **THEN** exports share checkpoint identity and have independent model hashes

### Requirement: Model export sidecar
The sidecar SHALL contain the complete portable contract plus format, artifact
filename, source checkpoint hash, and exported model hash.

#### Scenario: Detached source checkpoint
- **WHEN** a canonical checkpoint is outside its old run
- **THEN** explicit export output resolution is sufficient
