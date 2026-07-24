## ADDED Requirements

### Requirement: Authored root file
Config-aware commands SHALL accept `--config FILE` as an authored Hydra root,
compose its defaults against the normal search path, and apply ordinary inline
overrides after composition.

#### Scenario: External authored root
- **WHEN** a YAML file outside every config search directory is passed with
  `--config`
- **THEN** its parent becomes the primary config directory and packaged/local
  groups remain available as fallbacks

### Requirement: Exclusive root sources
`--config`, `--resolved-config`, and `experiment=<name>` SHALL be mutually
exclusive root config sources.

#### Scenario: Conflicting roots
- **WHEN** a caller supplies both `--config experiment.yaml` and
  `experiment=manual`
- **THEN** composition fails with an error naming the conflicting root sources

### Requirement: Command I/O is not experiment policy
The CLI SHALL represent completed checkpoints, ad-hoc inference inputs,
conversion targets, and artifact output selections as Typer options rather
than Hydra config overrides.

#### Scenario: Evaluate a checkpoint
- **WHEN** a caller runs
  `sati eval experiment=manual --checkpoint ./checkpoints/best.pt`
- **THEN** the checkpoint is treated as command input and is not written into
  the authored model-initialization config

### Requirement: Config comparison
`sati inspect config-compare` SHALL compare authored, composed, or resolved
configs and organize differences by their contributing hash source plus an
`other` section.

#### Scenario: Path-only difference
- **WHEN** two configs differ only in portable path values excluded from
  behavioral hashes
- **THEN** the difference appears in `other` and their applicable behavioral
  hashes remain equal

## MODIFIED Requirements

### Requirement: Hybrid experiment root selection
The system SHALL use packaged `config.yaml` when no root is selected and SHALL
promote `experiment=<name>` to the full Hydra root
`experiment/<name>.yaml`; named experiments SHALL be the standard project
workflow and SHALL carry mandatory `experiment.name`.

#### Scenario: Project experiment
- **WHEN** a caller runs
  `sati train experiment=v1/manual/fullres_bbox`
- **THEN** `./configs/experiment/v1/manual/fullres_bbox.yaml` is selected as
  the primary root without requiring `--config-dir configs`

### Requirement: Current config search precedence
The system SHALL search repeatable explicit `--config-dir` entries, then
project-local `./configs`, then packaged `sati/config_defaults`, and SHALL move
the directory owning the selected primary root first for Hydra composition.

#### Scenario: Local group shadows packaged group
- **WHEN** `./configs/training/default.yaml` and the packaged group both exist
- **THEN** the local group is selected while omitted local groups fall through
  to the packaged tree

### Requirement: Inline override dash rule
The CLI SHALL treat Hydra override grammar tokens as dash-free config
overrides and dash-prefixed tokens as Typer options; unknown dash-prefixed
tokens SHALL fail as CLI errors.

#### Scenario: Mixed options and overrides
- **WHEN** a caller supplies
  `training.epochs=10 --checkpoint ./best.pt`
- **THEN** only `training.epochs=10` reaches Hydra and `--checkpoint` is parsed
  by the command

### Requirement: Current root schema
The strict root config SHALL contain `experiment`, `task`, `runtime`, `data`,
`slicing`, `model`, `training`, `recombination`, `results`, `output_root`,
`prepare_outputs`, `training_outputs`, `eval_outputs`, `infer_outputs`, and
`export_outputs`; legacy `output.run_dir`, root `run_id`, and
`training.seed` SHALL be absent.

#### Scenario: Legacy output key
- **WHEN** an authored config contains `output.run_dir`
- **THEN** strict validation fails and names the replacement output blocks

### Requirement: Resolved config artifact
Config resolution SHALL be command-aware and side-effect-free until a mutating
command passes collision checks; a resolved artifact SHALL record generated
IDs, effective hashes, active command, and concrete active output paths.

#### Scenario: Read-only config inspection
- **WHEN** `sati inspect config experiment=manual` resolves a valid config
- **THEN** it reports the same effective IDs, hashes, and paths a mutating
  command would use without creating those paths

### Requirement: Current command surface
The installed `sati` CLI SHALL expose leaf commands `prepare`, `convert`,
`train`, `eval`, `infer`, and `export`, plus `inspect config`,
`config-compare`, `dataset`, `slices`, `results`, and `model`; every command
and group invoked without required input SHALL render help.

#### Scenario: Inspect help
- **WHEN** a caller runs `sati inspect` without a subcommand
- **THEN** help lists all six inspection commands

## REMOVED Requirements

### Requirement: Local `./config_defaults` project shadow
**Reason**: Editable project configs use dojo's `./configs` convention while
`config_defaults` is reserved for immutable packaged resources.

**Migration**: Move authored local YAML from `./config_defaults` to
`./configs`; no compatibility search path remains.

### Requirement: Mutating resolution writes root `resolved.yaml`
**Reason**: A complete config provenance bundle now belongs under
`<active-output>/config/` and resolution must be side-effect-free before
collision checking.

**Migration**: Use `config/resolved.yaml` in newly generated artifacts.
