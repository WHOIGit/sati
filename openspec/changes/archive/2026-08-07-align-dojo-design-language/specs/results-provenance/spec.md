## ADDED Requirements

### Requirement: Target schema compatibility identity
The system SHALL compute `target_schema_hash` from a versioned source
describing task type, target structure, and half-open top-left `xyxy` box
semantics, excluding class labels.

#### Scenario: Class label rename
- **WHEN** only a class display name changes
- **THEN** `target_schema_hash` remains stable and `class_mapping_hash` changes

### Requirement: Prepared dataset identity provenance
The system SHALL record `prepared_dataset_hash` separately from logical
`dataset_hash` and preprocessing compatibility in prepared products,
checkpoints, lifecycle manifests, and applicable result metadata.

#### Scenario: Different target gates
- **WHEN** two prepared products share raw data but use different target gates
- **THEN** their result rows can group by the same `dataset_hash` while
  provenance distinguishes their `prepared_dataset_hash` values

### Requirement: Model artifact identity
The system SHALL use `checkpoint_hash` for canonical training checkpoint bytes
and `model_hash` for exported portable-model bytes; there SHALL be no generated
checkpoint ID.

#### Scenario: Exported artifact
- **WHEN** an ONNX model is produced from a checkpoint
- **THEN** export metadata records both byte identities without treating them
  as interchangeable

### Requirement: Command-aware config identity
`config_hash` SHALL cover only the active command's resolved behavioral
configuration and SHALL exclude command output blocks, generated runtime
identities, completed model/input options, and inactive command-only fields.

#### Scenario: Figure directory changes
- **WHEN** only an output directory or output path template changes
- **THEN** `config_hash` and `config_id` remain unchanged

### Requirement: Model compatibility projection
`model_config_hash` SHALL cover the buildable model architecture, resolved
input shape, and output dimensions, and SHALL exclude initialization weights,
training/optimizer settings, recombination, and output policy.

#### Scenario: Same architecture, different initialization
- **WHEN** two training configs build the same model shape from different
  initialization checkpoints
- **THEN** they have equal `model_config_hash` values, different
  `config_hash` values, and their trained checkpoints are distinguished by
  `checkpoint_hash`

## MODIFIED Requirements

### Requirement: Tall detection results
The system SHALL write explicit-schema tall detection tables under the active
command's `results/` directory, discriminated by `frame_metadata`, optional
`slice_prediction`, `level_merge`, and `final_merge`; evaluation SHALL use
`validation_results.parquet` and ad-hoc inference SHALL use
`predictions.parquet`.

#### Scenario: Stage-specific nulls
- **WHEN** a row type does not use a schema column
- **THEN** the column is written as a typed null in the shared schema

### Requirement: Result provenance
Every result row SHALL carry full-frame identity, dataset split, run ID,
schema version, and applicable `config_hash`, `dataset_hash`,
`prepared_dataset_hash`, `target_schema_hash`, `class_mapping_hash`,
`model_config_hash`, `preprocessing_hash`, `checkpoint_hash`, and
scope-specific `recombination_hash`.

#### Scenario: Final merge provenance
- **WHEN** a `final_merge` row is written
- **THEN** its recombination hash covers both merge stages and its remaining
  compatibility hashes identify the exact logical data, prepared product,
  target, labels, model, preprocessing, and checkpoint

### Requirement: Verifiable metadata sidecar
The `_metadata.json` sidecar SHALL validate strictly, preserve ordered numeric
class mappings, store every compatibility/provenance hash beside its exact
versioned `*_source`, and distinguish table-level constants from row-varying
recombination identities.

#### Scenario: Hash recipe mismatch
- **WHEN** a sidecar source names a different recipe version than its stored
  hash contract
- **THEN** sidecar validation or hash verification fails

### Requirement: Current config identity
`config_hash` SHALL hash a versioned command-aware source after replacing
behavioral `SatiPath` values with `"<path>"`, substituting a local
initialization checkpoint's byte hash or stable library asset name, and
removing generated/runtime/output/inactive fields.

#### Scenario: Checkout relocation
- **WHEN** the same command behavior and logical data are resolved under two
  filesystem roots
- **THEN** `config_hash` is identical and data location differences are
  represented outside it

### Requirement: Current model and compatibility identities
The system SHALL provide versioned deterministic `dataset_hash`,
`prepared_dataset_hash`, `target_schema_hash`, `class_mapping_hash`,
`model_config_hash`, `preprocessing_hash`, `checkpoint_hash`, `model_hash`
when exported, and scope-specific `recombination_hash`.

#### Scenario: Recipe update
- **WHEN** a hash source definition changes
- **THEN** its recipe version and resulting hash change, and old/new equality
  is never implied

### Requirement: Read-only artifact inspection
`sati inspect` SHALL summarize configs, prepared datasets, slices, results,
checkpoints, and exported models without a forward pass or write;
machine-readable output SHALL be selected by `--format json`.

#### Scenario: Results inspection
- **WHEN** a canonical result path or run directory is inspected
- **THEN** the report locates the table and sidecar, inventories record types
  and identities, re-verifies hash sources, and includes metric and lifecycle
  summaries when present

## REMOVED Requirements

### Requirement: Dataset hash over emitted slice file identities
**Reason**: Logical dataset identity must remain comparable across preparation
and preprocessing variants; prepared-product identity now covers emitted
artifacts and preparation policy.

**Migration**: Regenerate prepared products and results with the new
`dataset_hash` and `prepared_dataset_hash` recipes.

### Requirement: Model config hash includes initialization weights
**Reason**: Initialization provenance belongs to `config_hash`, while
`model_config_hash` represents the architecture needed to interpret trained
parameters.

**Migration**: Recompute model compatibility metadata through new canonical
training/export paths.
