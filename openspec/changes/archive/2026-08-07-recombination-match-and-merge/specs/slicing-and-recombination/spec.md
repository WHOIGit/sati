## ADDED Requirements

### Requirement: Two-stage recombination strategies
The system SHALL confidence-filter detections, match them within each level, then
map winners to full resolution and match across levels. Each stage SHALL select
cluster winners by score with a deterministic tie-break, and SHALL either
discard the non-winning members (`nms`) or replace the winner's box with the
axis-aligned envelope of the whole cluster (`greedy_nmm`), keeping the winner's
score and class in both modes. The default method SHALL be `nms`; merging SHALL be
available as a selectable option.

#### Scenario: Fragment outranks whole detection
- **WHEN** a slice-clipped fragment and the whole detection of the same object
  match, and the fragment carries the higher score
- **THEN** under `greedy_nmm` the merged box covers the whole object regardless of
  which member won

#### Scenario: Default suppression mode
- **WHEN** no method is configured
- **THEN** each surviving detection's box is exactly one input box

#### Scenario: Final detection
- **WHEN** a detection survives cross-level matching
- **THEN** its final box is in full-frame full-resolution coordinates and records
  its contributing source levels

### Requirement: Bounded merge growth
A merge SHALL replace the winner's box with the cluster envelope only when the
envelope's area is at most a configured multiple of the winner's own area;
otherwise the winner SHALL retain its own box. Matched members SHALL be consumed
either way, so the bound never changes how many detections survive — only their
geometry. The bound SHALL be recorded in the recombination identity regardless of
the selected method.

#### Scenario: Large false positive contains a small true detection
- **WHEN** a small high-scoring detection matches a much larger low-scoring box
  whose envelope would exceed the bound
- **THEN** the surviving detection keeps its own box and the larger box is still
  removed as a duplicate

#### Scenario: Bound does not change detection count
- **WHEN** the same input is recombined at two different growth bounds
- **THEN** the number of surviving detections is equal

### Requirement: Configurable match metric
Cluster membership SHALL be decided by a configurable metric —
intersection-over-union or intersection-over-smaller — compared strictly greater
than the stage threshold. The default SHALL be intersection-over-smaller, because
a slice-clipped fragment is a strict subset of its whole detection and therefore
cannot exceed any usable intersection-over-union threshold.

#### Scenario: Nested fragment
- **WHEN** one detection is wholly contained in another
- **THEN** intersection-over-smaller matches them and
  intersection-over-union equals their area ratio

#### Scenario: Threshold boundary
- **WHEN** the metric equals the threshold exactly
- **THEN** the pair does not match

### Requirement: Class-agnostic matching by default
Matching SHALL ignore class by default, so that one object cannot retain two
detections that differ only in class, and the highest-scoring member's class
SHALL become the merged detection's class. Class-aware matching SHALL remain
selectable.

#### Scenario: Duplicate with disagreeing class
- **WHEN** two detections of the same object carry different classes and match
- **THEN** one detection survives, carrying the higher-scoring member's class

#### Scenario: Class-aware ablation
- **WHEN** class-agnostic matching is disabled
- **THEN** a detection only matches detections of its own class

## REMOVED Requirements

### Requirement: Two-stage NMS recombination
**Reason**: Both stages now select over clusters formed by a configurable match
metric and may merge rather than suppress, so "NMS" no longer names the
behavior. Superseded by "Two-stage recombination strategies", "Configurable
match metric", and "Class-agnostic matching by default".

**Migration**: None for callers — the stage boundaries, coordinate conventions,
and provenance columns are unchanged. Authored configs SHALL rename
`within_level_iou`/`cross_level_iou` to `within_level_threshold`/
`cross_level_threshold`; no alias remains.
