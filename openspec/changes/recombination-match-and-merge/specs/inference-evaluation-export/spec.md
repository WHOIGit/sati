## MODIFIED Requirements

### Requirement: Recombination override precedence
Infer/eval SHALL accept changes to recombination and result policy, including the
merge method, the match metric, the matching mode, and either stage threshold,
and SHALL reject changes to model architecture, class mapping, target schema, or
preprocessing.

#### Scenario: Post-hoc threshold choice
- **WHEN** the cross-level threshold is overridden
- **THEN** final recombination identity changes while compatibility identities
  remain fixed

#### Scenario: Post-hoc strategy choice
- **WHEN** the merge method or match metric is overridden at infer/eval time
- **THEN** the override is accepted, recorded on merge rows, and moves only the
  recombination identity
