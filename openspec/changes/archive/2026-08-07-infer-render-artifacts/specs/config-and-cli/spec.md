## MODIFIED Requirements

### Requirement: Current root schema
The strict root SHALL contain `experiment`, `task`, `runtime`, `data`,
`slicing`, `model`, `training`, `recombination`, `prediction`, `results`,
`rendering`, `output_root`, and the five command output blocks; legacy
`output.run_dir`, root `run_id`, and `training.seed` SHALL be absent. The
`rendering` block SHALL carry a global switch, per-artifact-kind toggles, and a
portable path to the rendering library's own style configuration, whose contents
the root schema SHALL NOT re-validate.

#### Scenario: Legacy output key
- **WHEN** an authored config contains `output.run_dir`
- **THEN** strict validation fails

#### Scenario: Style configuration contents
- **WHEN** a config points at a rendering style file
- **THEN** the root schema validates only that it is a path, leaving its keys to
  the rendering library
