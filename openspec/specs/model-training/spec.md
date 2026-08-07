# Model Training Specification

## Purpose

Current stock-Ultralytics training boundary and canonical training artifacts.

## Requirements

### Requirement: Backend protocol
The system SHALL isolate framework operations behind a backend protocol and
provide stock Ultralytics without subclassing its dataset, trainer, or loss. The
prediction contract SHALL carry the per-slice non-maximum-suppression parameters
explicitly — the match threshold and whether suppression ignores class — rather
than deferring them to the backend library's own defaults.

#### Scenario: Core install
- **WHEN** Ultralytics is not installed
- **THEN** non-model modules remain importable

#### Scenario: Prediction parameters are supplied, not defaulted
- **WHEN** slices are predicted
- **THEN** the per-slice suppression threshold and class-agnostic flag come from
  configuration, and the backend library's defaults for them are never relied upon

#### Scenario: Library default drift
- **WHEN** the backend library changes its own suppression defaults
- **THEN** prediction behavior is unchanged, because the values are pinned in
  configuration

### Requirement: Structured model initialization
Model config SHALL separate architecture from initialization weights, whose
source is `none`, `library`, or `checkpoint`; completed consumed checkpoints
SHALL not use this initialization block.

#### Scenario: Library initialization
- **WHEN** weights source is `library`
- **THEN** its stable asset name contributes to `config_hash`

### Requirement: Stock materialized training
`sati train` SHALL use `data.prepared.uri`, verify prepared identities,
initialize according to structured model policy, and pass resolved
model/training/runtime values plus the configured augmentation knobs to
Ultralytics. Augmentation SHALL be deactivated by default; the system SHALL
never forward a knob whose value is null.

#### Scenario: Default seed
- **WHEN** no seed override is authored
- **THEN** `runtime.seed` is passed and recorded

#### Scenario: Default augmentation
- **WHEN** no augmentation is authored
- **THEN** every knob is passed to Ultralytics at its off-value and no run
  inherits Ultralytics' stock augmentation

#### Scenario: Opt into stock augmentation
- **WHEN** the `augmentation/ultralytics` group is composed
- **THEN** Ultralytics' stock augmentation values are forwarded

### Requirement: Backend-native artifact isolation
Ultralytics-native output SHALL remain under `backend/ultralytics/` as
non-contract debugging material.

#### Scenario: Backend layout changes
- **WHEN** Ultralytics adds an unknown file
- **THEN** it remains isolated under the backend directory

### Requirement: Current training location
Ultralytics SHALL train under
`<training_outputs.dir>/backend/ultralytics/`; canonical artifacts SHALL be
exposed through `checkpoints/`, `metrics/`, and `figures/ultralytics/`.

#### Scenario: Training completes
- **WHEN** training succeeds
- **THEN** no public `<run>/train/weights/best.pt` contract is created

### Requirement: Canonical checkpoint artifacts
Training SHALL embed the portable contract into best and last checkpoints,
then place them under `checkpoints/` as
`<role>.<first-six-checkpoint-hash>.pt`.

#### Scenario: Best checkpoint finalization
- **WHEN** a best checkpoint is finalized
- **THEN** it reloads, its filename matches its final bytes, and metadata
  inventories it

### Requirement: Portable checkpoint contract
Canonical checkpoints SHALL embed model, preprocessing, recombination, class,
target, identity, schema-version, source-run, and config information sufficient
for portable inference.

#### Scenario: Detached checkpoint
- **WHEN** a checkpoint is copied away from its run
- **THEN** infer/eval/export reconstruct defaults without source `resolved.yaml`

### Requirement: Checkpoint byte identity
`checkpoint_hash` SHALL hash post-embedding bytes and feed the filename,
manifest, metadata, result provenance, and export metadata.

#### Scenario: Embedded metadata changes bytes
- **WHEN** the inference contract is embedded
- **THEN** the final artifact rather than backend source is hashed

### Requirement: Canonical epoch metrics
Training SHALL map native epoch CSV output to stable one-row-per-epoch
`metrics/metrics.csv`, namespacing backend-only columns.

#### Scenario: Native metrics retained
- **WHEN** metrics are normalized
- **THEN** native CSV remains only under `backend/ultralytics`

### Requirement: Copied Ultralytics training figures
When enabled, training SHALL copy every native figure to
`figures/ultralytics/` with its filename unchanged.

#### Scenario: Figures disabled
- **WHEN** figures are disabled
- **THEN** no canonical figures are written

#### Scenario: Native figure copy
- **WHEN** Ultralytics emits `results.png`
- **THEN** SATI writes `figures/ultralytics/results.png`
