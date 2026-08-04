# Configuration and CLI Specification

## Purpose

Current Hydra composition, strict configuration, command-aware resolution, and
the installed `sati` command surface.

## Requirements

### Requirement: Strict composed configuration
The system SHALL compose through the Hydra Compose API, validate against strict
Pydantic models with unknown fields forbidden, and pass only a typed
`RootConfig` beyond the configuration boundary.

#### Scenario: Unsupported key
- **WHEN** authored YAML or an inline override contains an unsupported key
- **THEN** configuration fails before pipeline work begins

### Requirement: Hybrid experiment root selection
The system SHALL use packaged `config.yaml` when no root is selected and SHALL
promote `experiment=<name>` to the full root `experiment/<name>.yaml`; named
experiments SHALL be the standard project workflow and carry mandatory
`experiment.name`.

#### Scenario: Project experiment
- **WHEN** a caller runs `sati train experiment=v1/manual/fullres_bbox`
- **THEN** `./configs/experiment/v1/manual/fullres_bbox.yaml` is selected
  without requiring `--config-dir configs`

### Requirement: Current config search precedence
The system SHALL search repeatable explicit `--config-dir` entries, then
project-local `./configs`, then packaged `sati/config_defaults`, moving the
directory owning the selected primary root first for Hydra composition.

#### Scenario: Local group shadows packaged group
- **WHEN** a local and packaged group option have the same name
- **THEN** the local option is selected and omitted local groups fall through
  to packaged defaults

### Requirement: Authored root file
Config-aware commands SHALL accept `--config FILE` as an authored Hydra root,
compose its defaults against the normal search path, and apply ordinary inline
overrides afterward.

#### Scenario: External authored root
- **WHEN** a YAML file outside every search directory is passed with `--config`
- **THEN** its parent becomes primary while normal group fallbacks remain
  available

### Requirement: Exclusive root sources
`--config`, `--resolved-config`, and `experiment=<name>` SHALL be mutually
exclusive root sources.

#### Scenario: Conflicting roots
- **WHEN** a caller supplies more than one root source
- **THEN** composition fails naming the conflict

### Requirement: Inline override dash rule
Hydra override grammar SHALL be dash-free; dash-prefixed tokens SHALL be Typer
options and unknown dash-prefixed tokens SHALL fail as CLI errors.

#### Scenario: Mixed options and overrides
- **WHEN** a caller supplies `training.epochs=10 --checkpoint ./best.pt`
- **THEN** only the first token reaches Hydra

### Requirement: Command I/O is not experiment policy
The CLI SHALL represent completed checkpoints, ad-hoc inference inputs,
conversion targets, and artifact format selections as Typer options rather
than experiment configuration.

#### Scenario: Evaluate a checkpoint
- **WHEN** `sati eval experiment=manual --checkpoint ./best.pt` runs
- **THEN** the checkpoint does not mutate model initialization config

### Requirement: Current root schema
The strict root SHALL contain `experiment`, `task`, `runtime`, `data`,
`slicing`, `model`, `training`, `recombination`, `results`, `output_root`, and
the five command output blocks; legacy `output.run_dir`, root `run_id`, and
`training.seed` SHALL be absent.

#### Scenario: Legacy output key
- **WHEN** an authored config contains `output.run_dir`
- **THEN** strict validation fails

### Requirement: Resolved config artifact
Resolution SHALL be command-aware and side-effect-free until a mutating command
passes collision checks. Persisted `config/resolved.yaml` SHALL be the
authoritative immutable worker config; `composed.yaml` is branching provenance
and `resolved.json` is a rendering of the same resolved tree.

#### Scenario: Read-only config inspection
- **WHEN** `sati inspect config experiment=manual` resolves a config
- **THEN** it reports effective identities and paths without creating them

### Requirement: Config comparison
`sati inspect config-compare` SHALL compare configs and group differences by
contributing hash source plus `other`.

#### Scenario: Path-only difference
- **WHEN** configs differ only by hash-excluded portable paths
- **THEN** the difference is under `other` and behavioral hashes remain equal

### Requirement: Current command surface
The CLI SHALL expose `prepare`, `convert`, `train`, `eval`, `infer`, `export`,
and `inspect config|config-compare|dataset|slices|results|model`.

#### Scenario: Inspect help
- **WHEN** `sati inspect` is invoked without a subcommand
- **THEN** help lists all six inspection commands
