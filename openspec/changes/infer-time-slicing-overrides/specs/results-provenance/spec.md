## MODIFIED Requirements

### Requirement: Result provenance
Every result row SHALL carry applicable config, dataset, prepared dataset,
target schema, class mapping, model config, preprocessing, effective
preprocessing, checkpoint, and scope-specific recombination identities, plus
frame/split/run/schema context. The scope-specific recombination identity SHALL
cover the merge method, the match metric, the matching mode, the confidence
filter, and the thresholds in scope. The preprocessing identity SHALL describe
the pyramid the checkpoint was trained on; the effective preprocessing identity
SHALL describe the pyramid the run actually used, and the two SHALL be equal
whenever no slice-geometry override is in effect.

#### Scenario: Final merge provenance
- **WHEN** a final merge row is written
- **THEN** its recombination hash covers both merge scopes

#### Scenario: Metric change moves identity
- **WHEN** two runs differ only in match metric or merge method
- **THEN** their recombination identities differ and every compatibility identity
  is equal

#### Scenario: Re-sliced run is self-describing
- **WHEN** a result row is read without access to its run directory
- **THEN** comparing its effective preprocessing identity to its preprocessing
  identity establishes whether the run re-sliced, without consulting any other
  artifact

#### Scenario: Unoverridden run
- **WHEN** a run uses the checkpoint's own pyramid
- **THEN** its effective preprocessing identity equals its preprocessing identity

### Requirement: Current model and compatibility identities
The system SHALL provide `version`-stamped `dataset_hash`,
`prepared_dataset_hash`, `target_schema_hash`, `class_mapping_hash`,
`model_config_hash`, `preprocessing_hash`, `effective_preprocessing_hash`,
`checkpoint_hash`, optional `model_hash`, and scope-specific
`recombination_hash`, each source built with the shared canonical mechanics.
`preprocessing_hash` and `effective_preprocessing_hash` SHALL share one source
definition and one `version`, differing only in whether that source is built from
the checkpoint contract's preprocessing or from the resolved runtime
preprocessing.

#### Scenario: Version update
- **WHEN** a source definition changes
- **THEN** its `version` and digest change

#### Scenario: Shared source definition
- **WHEN** the preprocessing source definition is revised
- **THEN** both the preprocessing and effective preprocessing identities change
  version and digest together

### Requirement: Verifiable metadata sidecar
`_metadata.json` SHALL validate strictly, preserve ordered numeric class
mappings, pair hashes with exact sources carrying their `version` field in the
config's key namespace, and distinguish constants from row-varying recombination
identities. It SHALL record the effective preprocessing identity with its exact
source alongside the checkpoint's preprocessing identity. It SHALL NOT imply that
equal identities guarantee equal detections while any behavioral configuration is
declared unhashed.

#### Scenario: Hash version mismatch
- **WHEN** a stored source and its `version` disagree with the recorded digest
- **THEN** validation or verification fails

#### Scenario: Comparing two runs with equal identities
- **WHEN** a reader compares two runs whose compatibility identities are all equal
- **THEN** the declared unhashed prediction parameters must still be checked in the
  resolved configuration before concluding the runs are behaviorally identical

#### Scenario: Re-sliced run sidecar
- **WHEN** a run overrides slice geometry
- **THEN** the sidecar carries the effective preprocessing source describing the
  pyramid used, and it verifies against the recorded effective digest

### Requirement: Read-only artifact inspection
`sati inspect` SHALL summarize configs, products, slices, results, checkpoints,
and exports without a forward pass or write; JSON uses `--format json`. Config
comparison SHALL group slicing fields by whether they are pinned by a checkpoint
contract or overridable at infer/eval time, so that a difference in slice geometry
is not presented as a compatibility break.

#### Scenario: Results inspection
- **WHEN** a canonical results path or run is inspected
- **THEN** tables, identities, hash verification, metrics, and lifecycle are
  summarized

#### Scenario: Comparing a re-sliced run against its baseline
- **WHEN** two resolved configs differ only in level policy
- **THEN** the difference is reported under the overridable slice-geometry group
  and no pinned compatibility identity is reported as differing
