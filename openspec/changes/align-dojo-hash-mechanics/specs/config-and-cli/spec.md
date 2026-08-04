## MODIFIED Requirements

### Requirement: Resolved config artifact
Resolution SHALL be command-aware and side-effect-free until a mutating command
passes collision checks. Persisted `config/resolved.yaml` SHALL be the
authoritative immutable worker config; `composed.yaml` is branching provenance
and `resolved.json` is a rendering of the same resolved tree. Config inspection
SHALL NOT ingest the raw dataset: dataset-derived identities and any path that
depends on them SHALL be reported as deferred rather than computed.

#### Scenario: Read-only config inspection
- **WHEN** `sati inspect config experiment=manual` resolves a config
- **THEN** it reports effective identities and paths without creating them

#### Scenario: Inspection without the dataset
- **WHEN** `sati inspect config` runs and the raw dataset is absent
- **THEN** it still reports `config_hash`/`config_id` and active output paths and
  marks dataset-derived identities as deferred

### Requirement: Config comparison
`sati inspect config-compare` SHALL compare configs and group differences by
contributing hash source plus `other`, keyed in the resolved config's own dotted
namespace. Because hash sources share that namespace, a difference read by a
behavioral hash SHALL appear under that hash's group only, and `other` SHALL be a
true residual containing exactly the differences no behavioral hash reads.

#### Scenario: Path-only difference
- **WHEN** configs differ only by hash-excluded portable paths
- **THEN** the difference is under `other` and behavioral hashes remain equal

#### Scenario: Behavioral difference is not residual
- **WHEN** configs differ in a field a behavioral hash reads
- **THEN** the difference appears under that hash's group and not under `other`

#### Scenario: Shared field
- **WHEN** a difference is read by more than one hash source
- **THEN** it appears under every group that reads it and not under `other`
