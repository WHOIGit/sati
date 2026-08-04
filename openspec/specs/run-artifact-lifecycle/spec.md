# Run and Artifact Lifecycle Specification

## Purpose

Current invocation/product identities, output resolution, provenance bundles,
collision safety, and lifecycle state.

## Requirements

### Requirement: Distinct content and invocation identities
`run_id` SHALL identify an invocation, while `config_id` and
`prepared_dataset_id` derive from their hashes; there SHALL be no `run_hash`.

#### Scenario: Repeat an experiment
- **WHEN** equivalent behavior runs twice
- **THEN** content identities match and default run IDs differ

### Requirement: Dojo-style template rendering
Output paths SHALL support dotted resolved tokens, `{timestamp}`,
`{coolname:<source>}`, `{coolname:noseed}`, and `{source_run_dir}`, with
absolute, CWD-relative `./`, and parent-relative bare path semantics.

#### Scenario: Training template
- **WHEN** a training template uses experiment, timestamp, and run ID
- **THEN** it renders to an absolute directory under `output_root`

### Requirement: Active command output
A mutating command SHALL resolve one active output block and leave inactive
blocks unresolved; train/eval MAY resolve prepare output only as an input
locator.

#### Scenario: Training resolution
- **WHEN** train resolves
- **THEN** only training output paths are writable

### Requirement: Config provenance bundle
Before work, a mutating command SHALL atomically write
`config/composed.yaml`, `resolved.yaml`, `resolved.json`, `cli.txt`, and
`overrides.txt`.

#### Scenario: Inline override
- **WHEN** an override is supplied
- **THEN** effective config and authored token are both preserved

### Requirement: Authoritative resolved configuration
`config/resolved.yaml` SHALL be the immutable, fully composed, validated,
resolved worker configuration; JSON SHALL render the same tree and hash
sources SHALL derive from it.

#### Scenario: Instantiated run
- **WHEN** a run completes or fails
- **THEN** its resolved config remains unchanged

### Requirement: Lifecycle manifest
Each invocation SHALL atomically track command, state, timestamps, identities,
hashes, artifact paths, and failures.

#### Scenario: Failed invocation
- **WHEN** work raises after initialization
- **THEN** the manifest becomes `failed` and provenance remains

### Requirement: Collision safety
Mutations SHALL refuse non-empty outputs by default. `--clobber` SHALL clear
only a validated concrete target, never templates, roots, workspace, or home.

#### Scenario: Existing run
- **WHEN** a non-empty output exists without clobber
- **THEN** the command fails before modifying it

### Requirement: Exact resolved replay
`--resolved-config` SHALL preserve IDs, command, hashes, and paths and reject
behavioral overrides; branching SHALL use stored `composed.yaml` via
`--config`.

#### Scenario: Replay path
- **WHEN** training resumes from its resolved config
- **THEN** it targets the same run and directory

### Requirement: Training-only resume
`--resume` SHALL require training's exact resolved config and canonical last
checkpoint, refusing completed, mismatched, missing, or non-training contexts.

#### Scenario: Resume failed training
- **WHEN** a failed matching run has a valid last checkpoint
- **THEN** it resumes without clobbering provenance
