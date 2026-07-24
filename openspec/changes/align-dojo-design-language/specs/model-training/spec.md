## ADDED Requirements

### Requirement: Structured model initialization
The model config SHALL separate architecture (`source`, `name`, resolved input
shape) from initialization weights (`source: none | library | checkpoint`,
optional `uri`), and consumed completed checkpoints SHALL NOT use this
initialization block.

#### Scenario: Library initialization
- **WHEN** `model.weights.source` is `library`
- **THEN** the configured stable library asset initializes training and its
  name contributes to `config_hash`

### Requirement: Canonical checkpoint artifacts
Training SHALL finalize best and last checkpoints under `checkpoints/`, embed
the portable SATI inference contract before hashing, and name each artifact
`<role>.<first-six-checkpoint-hash>.pt`.

#### Scenario: Best checkpoint finalization
- **WHEN** the backend produces a best checkpoint
- **THEN** the canonical checkpoint reloads successfully, its filename prefix
  matches its final byte hash, and it is inventoried in
  `checkpoint_metadata.json`

### Requirement: Portable checkpoint contract
Each canonical checkpoint SHALL embed `sati_inference_contract` with buildable
model config, effective preprocessing/slicing, default recombination,
class mapping, target schema, compatibility hashes, schema versions, source
run ID, and config hash.

#### Scenario: Bare canonical checkpoint
- **WHEN** a canonical checkpoint is copied away from its producing run
- **THEN** infer/eval/export can reconstruct model and preprocessing defaults
  from the checkpoint without reading the producing `resolved.yaml`

### Requirement: Canonical epoch metrics
Training SHALL write `metrics/metrics.csv` with one row per completed epoch
and stable SATI metric names mapped from backend output; backend-only extra
columns SHALL be namespaced.

#### Scenario: Ultralytics split metrics
- **WHEN** Ultralytics writes its native epoch results CSV
- **THEN** SATI emits one canonical row for each epoch and retains the native
  CSV only under `backend/ultralytics`

### Requirement: Canonical training figures
When `training_outputs.figures.enabled` is true, training SHALL place every
available recognized backend plot under `figures/` using stable semantic names
and SHALL inventory missing optional figures without fabricating them.

#### Scenario: Figures disabled
- **WHEN** `training_outputs.figures.enabled` is false
- **THEN** no canonical training figures are written

### Requirement: Backend-native artifact isolation
Framework-native training artifacts SHALL be written under
`backend/ultralytics/` and SHALL be treated as non-contract debugging
artifacts rather than canonical checkpoint, metric, or figure paths.

#### Scenario: Backend layout changes
- **WHEN** a supported Ultralytics release adds an unrecognized native file
- **THEN** the file remains under `backend/ultralytics` and does not change
  SATI's canonical run layout

## MODIFIED Requirements

### Requirement: Stock materialized training
`sati train` SHALL train against `data.prepared.uri`, verify prepared and
preprocessing identities, initialize from the structured model weight policy,
and pass resolved model/training/runtime values plus only explicitly authored
YOLO augmentation values to stock Ultralytics.

#### Scenario: Default seed
- **WHEN** training starts without an inline seed override
- **THEN** it passes `runtime.seed` to Ultralytics and records that seed in the
  resolved config and config hash source

### Requirement: Current training location
The Ultralytics backend SHALL train under
`<training_outputs.dir>/backend/ultralytics/`; SATI SHALL expose canonical
artifacts only through sibling directories `checkpoints/`, `metrics/`, and
`figures/`.

#### Scenario: Training completes
- **WHEN** stock Ultralytics training succeeds
- **THEN** the run root contains the canonical artifact envelope and no public
  `<run>/train/weights/best.pt` contract

### Requirement: Checkpoint byte identity
After embedding the portable inference contract, the system SHALL compute
`checkpoint_hash` from final checkpoint bytes and SHALL use that value in the
filename, run manifest, checkpoint metadata, result provenance, and export
metadata.

#### Scenario: Metadata changes checkpoint bytes
- **WHEN** inference metadata is embedded into the backend checkpoint
- **THEN** SATI hashes the post-embedding artifact rather than the original
  backend file

## REMOVED Requirements

### Requirement: Public Ultralytics `train/` artifact layout
**Reason**: Backend-native directories are unstable and do not provide the
canonical sibling-style checkpoint/metric/figure contract.

**Migration**: Retrain or explicitly retain old checkpoints outside SATI; new
runs use `checkpoints/`, `metrics/`, `figures/`, and `backend/ultralytics/`.
