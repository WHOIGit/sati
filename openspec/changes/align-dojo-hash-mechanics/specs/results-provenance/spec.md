## ADDED Requirements

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

## MODIFIED Requirements

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

### Requirement: Verifiable metadata sidecar
`_metadata.json` SHALL validate strictly, preserve ordered numeric class
mappings, pair hashes with exact sources carrying their `version` field in the
config's key namespace, and distinguish constants from row-varying recombination
identities.

#### Scenario: Hash version mismatch
- **WHEN** a stored source and its `version` disagree with the recorded digest
- **THEN** validation or verification fails

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

### Requirement: Current model and compatibility identities
The system SHALL provide `version`-stamped `dataset_hash`,
`prepared_dataset_hash`, `target_schema_hash`, `class_mapping_hash`,
`model_config_hash`, `preprocessing_hash`, `checkpoint_hash`, optional
`model_hash`, and scope-specific `recombination_hash`, each source built with the
shared canonical mechanics.

#### Scenario: Version update
- **WHEN** a source definition changes
- **THEN** its `version` and digest change
