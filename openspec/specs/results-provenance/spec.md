# Results and Provenance Specification

## Purpose

Current canonical results, compatibility identities, metadata, and inspection.

## Requirements

### Requirement: Tall detection results
The system SHALL write explicit-schema tables under the active `results/`
directory with `frame_metadata`, optional `slice_prediction`, `level_merge`,
and `final_merge`; eval uses `validation_results.parquet` and infer uses
`predictions.parquet`. The shared schema SHALL include the absorbed-class list
on merge rows.

#### Scenario: Stage-specific nulls
- **WHEN** a row type does not use a column
- **THEN** the shared schema writes a typed null

#### Scenario: Schema widened
- **WHEN** merge rows gain the absorbed-class column
- **THEN** the results and sidecar schema version increases

### Requirement: Absorbed class provenance
Merge rows SHALL record every distinct class present in the cluster that produced
them, the winner's class first and the remainder ascending, so that a
class-agnostic merge never makes a class disagreement unrecoverable. The column
SHALL be written in every method and matching mode, and a single-member cluster
SHALL write a one-element list rather than a null.

#### Scenario: Agnostic merge over disagreeing classes
- **WHEN** class-agnostic matching merges detections of different classes
- **THEN** the merge row's class is the winner's and its absorbed-class list
  contains every class in the cluster

#### Scenario: No disagreement
- **WHEN** a cluster has one member
- **THEN** the absorbed-class list has exactly that member's class

### Requirement: Result provenance
Every result row SHALL carry applicable config, dataset, prepared dataset,
target schema, class mapping, model config, preprocessing, checkpoint, and
scope-specific recombination identities, plus frame/split/run/schema context.
The scope-specific recombination identity SHALL cover the merge method, the match
metric, the matching mode, the confidence filter, and the thresholds in scope.

#### Scenario: Final merge provenance
- **WHEN** a final merge row is written
- **THEN** its recombination hash covers both merge scopes

#### Scenario: Metric change moves identity
- **WHEN** two runs differ only in match metric or merge method
- **THEN** their recombination identities differ and every compatibility identity
  is equal

### Requirement: Per-strategy merge identity
Merge rows SHALL name the resolved merge method, and the strategy version carried
in the recombination identity SHALL be that strategy's own version, so adding or
revising one strategy does not appear to change another.

#### Scenario: Suppression baseline unchanged
- **WHEN** a run selects the suppression method
- **THEN** its merge rows name that method at its own unchanged version

### Requirement: Verifiable metadata sidecar
`_metadata.json` SHALL validate strictly, preserve ordered numeric class
mappings, pair hashes with exact versioned sources, and distinguish constants
from row-varying recombination identities.

#### Scenario: Hash recipe mismatch
- **WHEN** stored source/version and hash disagree
- **THEN** validation or verification fails

### Requirement: Target schema compatibility identity
`target_schema_hash` SHALL describe task, target structure, and half-open
top-left `xyxy`, excluding labels.

#### Scenario: Label rename
- **WHEN** only a display name changes
- **THEN** target schema identity stays stable and class mapping identity changes

### Requirement: Prepared dataset identity provenance
Prepared identity SHALL be distinct from logical dataset and preprocessing
identity in products, checkpoints, lifecycle manifests, and results.

#### Scenario: Different target gates
- **WHEN** prepared variants share raw data
- **THEN** results share dataset identity but not prepared identity

### Requirement: Model artifact identity
`checkpoint_hash` SHALL identify canonical checkpoint bytes and `model_hash`
exported portable-model bytes; no checkpoint ID SHALL be generated.

#### Scenario: Exported model
- **WHEN** a model is exported
- **THEN** both byte identities are recorded without interchangeability

### Requirement: Command-aware config identity
`config_hash` SHALL cover only active-command resolved behavior, excluding
output blocks, generated runtime identities, completed input options, and
inactive command fields.

#### Scenario: Output directory changes
- **WHEN** only output placement changes
- **THEN** config hash and ID remain unchanged

### Requirement: Current config identity
The versioned command-aware source SHALL replace behavioral `SatiPath` values
with `<path>`, substitute local initialization bytes or a stable library asset,
and omit generated/runtime/output/inactive fields.

#### Scenario: Checkout relocation
- **WHEN** equivalent behavior resolves under two roots
- **THEN** config identity is equal

### Requirement: Model compatibility projection
`model_config_hash` SHALL cover buildable architecture, resolved input shape,
and output dimensions while excluding initialization, training,
recombination, and output policy.

#### Scenario: Different initialization
- **WHEN** identical architectures use different initialization checkpoints
- **THEN** model hashes match while config hashes may differ

### Requirement: Current model and compatibility identities
The system SHALL provide versioned `dataset_hash`, `prepared_dataset_hash`,
`target_schema_hash`, `class_mapping_hash`, `model_config_hash`,
`preprocessing_hash`, `checkpoint_hash`, optional `model_hash`, and
scope-specific `recombination_hash`.

#### Scenario: Recipe update
- **WHEN** a source definition changes
- **THEN** its recipe version and digest change

### Requirement: Read-only artifact inspection
`sati inspect` SHALL summarize configs, products, slices, results, checkpoints,
and exports without a forward pass or write; JSON uses `--format json`.

#### Scenario: Results inspection
- **WHEN** a canonical results path or run is inspected
- **THEN** tables, identities, hash verification, metrics, and lifecycle are
  summarized
