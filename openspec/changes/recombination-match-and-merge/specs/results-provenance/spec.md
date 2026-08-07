## ADDED Requirements

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

### Requirement: Per-strategy merge identity
Merge rows SHALL name the resolved merge method, and the strategy version carried
in the recombination identity SHALL be that strategy's own version, so adding or
revising one strategy does not appear to change another.

#### Scenario: Suppression baseline unchanged
- **WHEN** a run selects the suppression method
- **THEN** its merge rows name that method at its own unchanged version

## MODIFIED Requirements

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
