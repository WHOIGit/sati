# Results and Provenance Specification

## Purpose

Current canonical detection result table, metadata sidecars, configuration
identity, compatibility hashes, and read-only artifact inspection.

## Requirements

### Requirement: Tall detection results
The system SHALL write detection results as one explicit-schema Parquet table
discriminated by `record_type`: `frame_metadata`, optional
`slice_prediction`, `level_merge`, and `final_merge`.

#### Scenario: Stage-specific nulls
- **WHEN** a row type does not use a schema column
- **THEN** the column is written as a typed null rather than inferred or
  omitted from the table schema

### Requirement: Result provenance
Every result row SHALL carry full-frame identity, dataset split, run ID,
schema version, and the applicable config, model, checkpoint, dataset,
class-map, preprocessing, and recombination hashes.

#### Scenario: Final merge provenance
- **WHEN** a `final_merge` row is written
- **THEN** its recombination hash covers both within-level and cross-level
  recombination settings

### Requirement: Verifiable metadata sidecar
The `_metadata.json` sidecar SHALL validate against a strict Pydantic schema,
preserve numeric class-map insertion order, and store each compatibility hash
beside the exact versioned source object from which it was computed.

#### Scenario: Sidecar tampering
- **WHEN** a stored hash source is modified without updating its hash
- **THEN** metadata hash verification fails

### Requirement: Current config identity
`config_hash` SHALL hash the resolved root config after replacing every
`SatiPath` value with `"<path>"`, removing hash-derived fields such as
`run_id`, and replacing a real local `model.weights` path with the file's byte
hash while leaving a hub asset name literal.

#### Scenario: Checkout relocation
- **WHEN** the same authored config is resolved under two filesystem roots
- **THEN** its `config_hash` is identical

### Requirement: Current model and compatibility identities
The system SHALL provide deterministic `model_config_hash`,
`checkpoint_hash`, `dataset_hash`, `class_map_hash`, `preprocessing_hash`, and
scope-specific `recombination_hash` values using versioned canonical-JSON
recipes.

#### Scenario: Checkpoint changes
- **WHEN** checkpoint bytes change
- **THEN** `checkpoint_hash` changes even if the checkpoint filename does not

### Requirement: Read-only artifact inspection
`sati inspect` SHALL summarize configs, prepared datasets, slices, results, and
models without performing a model forward pass or modifying the inspected
artifact.

#### Scenario: Results inspection
- **WHEN** a result Parquet file and sidecar are inspected
- **THEN** the report includes row counts by record type, run IDs, hash values,
  sidecar verification, and metrics when present
