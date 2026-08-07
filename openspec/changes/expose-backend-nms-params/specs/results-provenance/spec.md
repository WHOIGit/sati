## ADDED Requirements

### Requirement: Declared unhashed behavioral configuration
The per-slice prediction parameters SHALL be recorded in every run's resolved
configuration artifacts while remaining outside every compatibility identity. This
exclusion SHALL be intentional and guarded: the configuration block SHALL appear in
no command's hashed projection, and a test SHALL fail if it is ever added without a
deliberate identity version change.

#### Scenario: Recorded but not identity-bearing
- **WHEN** two runs differ only in a per-slice prediction parameter
- **THEN** the difference is visible in each run's resolved configuration while every
  compatibility identity is equal

#### Scenario: Accidental inclusion is caught
- **WHEN** the prediction block is added to a command's hashed projection
- **THEN** the guard test fails, forcing the identity version change to be deliberate

## MODIFIED Requirements

### Requirement: Verifiable metadata sidecar
`_metadata.json` SHALL validate strictly, preserve ordered numeric class
mappings, pair hashes with exact versioned sources, and distinguish constants
from row-varying recombination identities. It SHALL NOT imply that equal identities
guarantee equal detections while any behavioral configuration is declared unhashed.

#### Scenario: Hash recipe mismatch
- **WHEN** stored source/version and hash disagree
- **THEN** validation or verification fails

#### Scenario: Comparing two runs with equal identities
- **WHEN** a reader compares two runs whose compatibility identities are all equal
- **THEN** the declared unhashed prediction parameters must still be checked in the
  resolved configuration before concluding the runs are behaviorally identical
