## MODIFIED Requirements

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
