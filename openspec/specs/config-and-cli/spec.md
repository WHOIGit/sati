# Configuration and CLI Specification

## Purpose

Current Hydra composition, strict configuration, runtime resolution, and the
installed `sati` command surface.

## Requirements

### Requirement: Strict composed configuration
The system SHALL compose configuration through the Hydra Compose API, validate
the composed tree against strict Pydantic models with unknown fields forbidden,
and pass only a typed `RootConfig` beyond the configuration boundary.

#### Scenario: Unsupported key
- **WHEN** an authored YAML or inline override contains a key absent from the
  current Pydantic schema
- **THEN** configuration fails before any pipeline work begins

### Requirement: Hybrid experiment root selection
The system SHALL use packaged `config.yaml` when no experiment is selected and
SHALL promote `experiment=<name>` to the full Hydra root
`experiment/<name>.yaml` when present.

#### Scenario: Local experiment with packaged groups
- **WHEN** a caller supplies `--config-dir configs
  experiment=v1/manual/fullres_bbox`
- **THEN** the local experiment is the primary root and its missing config
  groups fall through to packaged defaults

### Requirement: Current config search precedence
The system SHALL search explicit `--config-dir`, then local
`./config_defaults`, then packaged `sati/config_defaults`, with the directory
owning the selected primary root moved first to satisfy Hydra.

#### Scenario: Selector-less local shadow
- **WHEN** a caller expects local `./config_defaults` fragments to shadow
  packaged groups without selecting an experiment
- **THEN** the local tree must include its own primary `config.yaml`

### Requirement: Inline override dash rule
The CLI SHALL treat `--dashed` tokens as Typer options and dash-free
`key=value` tokens as Hydra overrides.

#### Scenario: Training override
- **WHEN** a caller runs `sati train training.epochs=10`
- **THEN** `training.epochs=10` reaches Hydra composition as a config override

### Requirement: Current root schema
The current root config SHALL contain `data`, `slicing`, `model`, `training`,
`recombination`, `results`, `output.run_dir`, and derived `run_id`.

#### Scenario: Derived runtime values
- **WHEN** the authored config omits `model.imgsz` and `run_id`
- **THEN** resolution sets `model.imgsz` from `slicing.nominal_slice_size` and
  generates a timestamp-plus-random-suffix `run_id`

### Requirement: Resolved config artifact
Config-aware commands SHALL support `--resolved-config`; loading that artifact
applies only supplied inline overrides, generates a fresh `run_id`, and writes
`resolved.yaml` directly under `output.run_dir` unless invoked read-only.

#### Scenario: Read-only config inspection
- **WHEN** config inspection resolves an authored config
- **THEN** it performs no filesystem write

### Requirement: Current command surface
The installed `sati` CLI SHALL expose leaf commands `prepare`, `convert`,
`train`, `eval`, `infer`, and `export`, plus `inspect config`, `dataset`,
`slices`, `results`, and `model`.

#### Scenario: Inspect help
- **WHEN** a caller runs `sati inspect` without a subcommand
- **THEN** help lists the five artifact-specific inspection commands
