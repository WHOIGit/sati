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

### Requirement: Shared canonical hash mechanics
Hash sources SHALL be constructed in the resolved config's own key namespace
(field names, not synthetic wrappers), serialized through the canonical-JSON
mechanism shared with the sibling project (recursive key sort, `%.12g` floats,
tokenized non-finite values, compact separators, `sha256:` prefix), and carry a
`version` field rather than an inline recipe string. Field exclusion SHALL be
type-driven: portable-path fields SHALL be identified by `SatiPathMarker` and
generated/derived identities by `HASH_DERIVED`, never by a hand-maintained key
list.

#### Scenario: Equal trees hash identically
- **WHEN** two source trees are structurally equal
- **THEN** they serialize to identical canonical bytes and the same digest

#### Scenario: New derived field
- **WHEN** a field marked `HASH_DERIVED` is added to the schema
- **THEN** it is excluded from every source without editing the hashing module

#### Scenario: Portable path only
- **WHEN** two configs differ only in `SatiPath`-marked fields
- **THEN** those fields are absent from behavioral sources and the digests are equal

### Requirement: Verifiable metadata sidecar
`_metadata.json` SHALL validate strictly, preserve ordered numeric class
mappings, pair hashes with exact sources carrying their `version` field in the
config's key namespace, and distinguish constants from row-varying recombination
identities. It SHALL NOT imply that equal identities guarantee equal detections
while any behavioral configuration is declared unhashed.

#### Scenario: Hash version mismatch
- **WHEN** a stored source and its `version` disagree with the recorded digest
- **THEN** validation or verification fails

#### Scenario: Comparing two runs with equal identities
- **WHEN** a reader compares two runs whose compatibility identities are all equal
- **THEN** the declared unhashed prediction parameters must still be checked in the
  resolved configuration before concluding the runs are behaviorally identical

### Requirement: Target schema compatibility identity
`target_schema_hash` SHALL describe the task type and the target/coordinate
convention (top-left, half-open `xyxy`), excluding labels. The coordinate
convention SHALL be a versioned constant of the source, not authored
configuration; authored `task` config SHALL carry only `type`.

#### Scenario: Label rename
- **WHEN** only a display name changes
- **THEN** target schema identity stays stable and class mapping identity changes

#### Scenario: Coordinate convention is not authored
- **WHEN** a config is authored
- **THEN** it exposes no box-format, bounds, or origin fields and the convention
  is fixed inside the target schema source

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
The versioned command-aware source SHALL be a projection of the resolved config
in its own key namespace: it SHALL omit `SatiPath`-marked portable paths and
`HASH_DERIVED` generated/derived fields, substitute a local initialization
checkpoint's byte hash or a stable library asset name for `model.weights`, and
include only the active command's behavioral blocks. It SHALL NOT wrap fields
under a synthetic `config`/`recipe` envelope.

#### Scenario: Checkout relocation
- **WHEN** equivalent behavior resolves under two roots
- **THEN** config identity is equal

#### Scenario: Bare namespace
- **WHEN** a config-hash source is produced
- **THEN** its top-level keys are config field names, not a `config` wrapper

### Requirement: Model compatibility projection
`model_config_hash` SHALL cover buildable architecture, resolved input shape,
and output dimensions while excluding initialization, training,
recombination, and output policy.

#### Scenario: Different initialization
- **WHEN** identical architectures use different initialization checkpoints
- **THEN** model hashes match while config hashes may differ

### Requirement: Current model and compatibility identities
The system SHALL provide `version`-stamped `dataset_hash`,
`prepared_dataset_hash`, `target_schema_hash`, `class_mapping_hash`,
`model_config_hash`, `preprocessing_hash`, `checkpoint_hash`, optional
`model_hash`, and scope-specific `recombination_hash`, each source built with the
shared canonical mechanics.

#### Scenario: Version update
- **WHEN** a source definition changes
- **THEN** its `version` and digest change

### Requirement: Read-only artifact inspection
`sati inspect` SHALL summarize configs, products, slices, results, checkpoints,
and exports without a forward pass or write; JSON uses `--format json`.

#### Scenario: Results inspection
- **WHEN** a canonical results path or run is inspected
- **THEN** tables, identities, hash verification, metrics, and lifecycle are
  summarized

### Requirement: Declared unhashed behavioral configuration
The per-slice prediction parameters SHALL be recorded in every run's resolved
configuration artifacts while remaining outside every compatibility identity. This
exclusion SHALL be intentional and guarded: the configuration block SHALL appear in
no command's hashed projection, and a test SHALL fail if it is ever added without a
deliberate identity version change.

#### Scenario: Recorded but not identity-bearing
- **WHEN** two runs differ only in a per-slice prediction parameter
- **THEN** the difference is visible in each run's resolved configuration while every
  compatibility identity is equal

#### Scenario: Accidental inclusion is caught
- **WHEN** the prediction block is added to a command's hashed projection
- **THEN** the guard test fails, forcing the identity version change to be deliberate
