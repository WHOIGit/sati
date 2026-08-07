## ADDED Requirements

### Requirement: Distinct raw and prepared data references
The config SHALL represent ingestible source data under `data.source` and the
reusable SATI-prepared product under `data.prepared`; preparation/conversion
SHALL consume the former and training/evaluation SHALL consume the latter.

#### Scenario: One experiment prepares and trains
- **WHEN** `data.prepared.uri` is omitted
- **THEN** it resolves to `prepare_outputs.dir`, so the same named experiment
  can prepare and then train without changing the meaning of a data field

### Requirement: Logical dataset identity
Preparation SHALL compute `dataset_hash` from canonical full-frame,
annotation, split, class-index, and source-identity information without
including slice policy, emitted slice files, or pixel payloads.

#### Scenario: Slicing policy changes
- **WHEN** overlap or nominal slice size changes for the same logical source
- **THEN** `dataset_hash` remains stable

### Requirement: Prepared dataset identity
Preparation SHALL compute `prepared_dataset_hash` from `dataset_hash` plus the
complete versioned preparation recipe, and SHALL derive
`prepared_dataset_id` deterministically from that hash.

#### Scenario: Training target gate changes
- **WHEN** `slicing.min_area_ratio` or `slicing.size_gate_px` changes
- **THEN** `prepared_dataset_hash` and the default prepared output directory
  change while `dataset_hash` does not

### Requirement: Prepared product completion manifest
A prepared product SHALL contain `prepare_manifest.json` recording lifecycle
status, dataset and preparation identities, schema versions, artifact paths,
and row/file counts; only `status=complete` is reusable.

#### Scenario: Interrupted preparation
- **WHEN** preparation stops before its manifest is marked complete
- **THEN** a later command refuses to consume or silently reuse the product

### Requirement: Verified prepared product reuse
`sati prepare` SHALL verify and reuse an existing complete product with the
same `prepared_dataset_hash`; a mismatched or incomplete target SHALL require
an explicit `--clobber`.

#### Scenario: Repeat preparation
- **WHEN** the same source and preparation config resolve to an existing valid
  complete product
- **THEN** preparation exits successfully without re-emitting slice pixels

## MODIFIED Requirements

### Requirement: Materialized prepared dataset
`sati prepare` SHALL write a reusable prepared product to
`prepare_outputs.dir`, including the common `config/` provenance bundle,
canonical `manifests/`, the derived `yolo/` tree, and
`prepare_manifest.json`.

#### Scenario: Default prepared output
- **WHEN** no explicit `prepare_outputs.dir` is authored
- **THEN** the directory is rendered under
  `./.cache/prepared_datasets/<prepared-dataset-id>` relative to the
  invocation CWD

### Requirement: Prepared-tree compatibility gate
The generated `yolo/data.yaml` and prepared metadata SHALL carry
`dataset_hash`, `prepared_dataset_hash`, and `preprocessing_hash`; training
SHALL refuse any mismatch with the selected prepared product and embedded
checkpoint/model contract before loading the model.

#### Scenario: Wrong prepared product
- **WHEN** training's resolved preparation or preprocessing identity differs
  from the selected prepared tree
- **THEN** training fails with both expected and observed identities

## REMOVED Requirements

### Requirement: Prepared dataset at generic run directory
**Reason**: Prepared data is a reusable, deterministic product and must not
share invocation-scoped `output.run_dir` semantics.

**Migration**: Regenerate the prepared dataset under
`prepare_outputs.dir`; old development products are not discovered.
