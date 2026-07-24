## ADDED Requirements

### Requirement: Distinct content and invocation identities
The system SHALL distinguish fresh invocation IDs from deterministic content
identities: `run_id` identifies one invocation, `config_id` is derived from
`config_hash`, `prepared_dataset_id` is derived from
`prepared_dataset_hash`, and there SHALL be no `run_hash`.

#### Scenario: Repeat the same experiment
- **WHEN** the same resolved behavioral config is invoked twice
- **THEN** both invocations have the same `config_hash` and `config_id` but
  different default `run_id` values

### Requirement: Dojo-style template rendering
The system SHALL render command output paths using resolved dotted config
tokens plus `{timestamp}`, `{coolname:<source>}`, `{coolname:noseed}`, and
`{source_run_dir}`, with absolute, CWD-relative `./`, and parent-relative bare
path semantics matching dojo.

#### Scenario: Training output template
- **WHEN** `training_outputs.dir_template` is
  `{experiment.name}/{timestamp}_{runtime.run_id}` and `output_root` is
  `./runs`
- **THEN** the resolved training directory is an absolute path under
  `./runs/<experiment-name>/`

### Requirement: Active command output
Each mutating command SHALL resolve exactly one of `prepare_outputs`,
`training_outputs`, `eval_outputs`, `infer_outputs`, or `export_outputs` and
SHALL leave inactive invocation output blocks unresolved; train/eval MAY
resolve `prepare_outputs.dir` only as a read-only default locator for
`data.prepared.uri`.

#### Scenario: Training resolution
- **WHEN** a config is resolved for `sati train`
- **THEN** `training_outputs.dir` and its child paths are the only writable
  outputs, `eval_outputs.dir` is not generated, and any resolved
  `prepare_outputs.dir` is treated only as an input reference

### Requirement: Config provenance bundle
Before pipeline work begins, every mutating command SHALL write
`config/composed.yaml`, `config/resolved.yaml`, `config/resolved.json`,
`config/cli.txt`, and `config/overrides.txt` under its resolved output.

#### Scenario: Inline override provenance
- **WHEN** training is invoked with `training.epochs=10`
- **THEN** the composed and resolved artifacts contain the effective value and
  `overrides.txt` records the authored override token

### Requirement: Lifecycle manifest
Every invocation output SHALL contain an atomically updated manifest recording
command, status, timestamps, identities, hashes, resolved artifact paths, and
failure details when applicable.

#### Scenario: Failed invocation
- **WHEN** work fails after the output directory is initialized
- **THEN** the lifecycle manifest ends in `failed` status with the failure
  summary while preserving the config provenance bundle

### Requirement: Collision safety
A mutating invocation SHALL refuse a non-empty resolved output directory by
default; `--clobber` SHALL clear only the validated concrete output directory,
and SHALL NOT accept an unresolved template, filesystem root, workspace root,
or home directory as its target.

#### Scenario: Existing completed run
- **WHEN** a command resolves to a non-empty completed run directory without
  `--clobber`
- **THEN** it fails before modifying that directory

### Requirement: Exact resolved replay
`--resolved-config` SHALL preserve generated IDs, active command, and resolved
paths and SHALL reject behavioral Hydra overrides; branching SHALL use the
stored `config/composed.yaml` through `--config`.

#### Scenario: Replay path
- **WHEN** `sati train --resolved-config <run>/config/resolved.yaml --resume`
  is invoked
- **THEN** it targets the same run ID and training directory

### Requirement: Training-only resume
`--resume` SHALL be supported only for training, require the exact resolved
config and a canonical last checkpoint, and refuse completed runs or
configuration mismatches.

#### Scenario: Resume failed training
- **WHEN** a failed training run has a valid last checkpoint and is invoked
  with its resolved config and `--resume`
- **THEN** training resumes in the same lifecycle context without clobbering
  prior provenance
