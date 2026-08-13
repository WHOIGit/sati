## MODIFIED Requirements

### Requirement: Resolution stack policies
The system SHALL support `auto` and `start_then_geometric` level policies, and
level index zero SHALL mean full resolution whenever full resolution is
present. The level policy SHALL be evaluated per command from that command's
resolved configuration, and SHALL NOT be fixed by the checkpoint for commands
that build their pyramid at runtime.

#### Scenario: Stack starts below full resolution
- **WHEN** `start_then_geometric.start_scale` is less than one
- **THEN** the first emitted level is numbered one

#### Scenario: Runtime policy differs from the trained policy
- **WHEN** an infer or eval command resolves a level policy differing from the one
  recorded in the checkpoint contract
- **THEN** the pyramid is planned from the resolved policy, and emitted slices
  retain the emitted size fixed by the contract
