## Why

SATI already shares dojo's strict Hydra/Pydantic foundation, but its CLI,
configuration roots, identities, and output layout expose a different and
partly accidental workflow. Aligning these contracts now, while SATI permits
breaking changes and before v2 adds streaming datasets and sweeps, will make
the two tools predictable siblings and prevent the overloaded
`output.run_dir` design from hardening.

## What Changes

- **BREAKING** Adopt dojo's config-first CLI grammar: local `./configs`
  automatically shadows packaged groups; `experiment=<name>` selects a full
  experiment root; dash-free dotted assignments are Hydra overrides; and
  `--config`, `--resolved-config`, and repeatable `--config-dir` are the only
  general config-source options.
- **BREAKING** Separate command I/O from experiment configuration. Completed
  checkpoints and ad-hoc inference inputs become explicit options such as
  `--checkpoint` and `--input`; initialization weights remain model config.
- **BREAKING** Replace `output.run_dir` and the root-level `run_id` with
  `experiment`, `runtime`, `output_root`, and command-specific
  `prepare_outputs`, `training_outputs`, `eval_outputs`, `infer_outputs`, and
  `export_outputs` blocks using dojo-style path templates.
- Add a common compose → validate → resolve → collision-check → persist
  lifecycle. Mutating commands write `config/composed.yaml`,
  `config/resolved.yaml`, `config/resolved.json`, `config/cli.txt`, and
  `config/overrides.txt`; `sati inspect config` renders the same resolution
  without side effects.
- **BREAKING** Introduce clear identity roles: fresh `run_id` for an invocation,
  deterministic human-readable `config_id` from `config_hash`, optional
  authored `dataset_id`, byte-derived `checkpoint_hash` with no checkpoint ID,
  and no `run_hash`.
- **BREAKING** Pin command-aware hash source projections and rename
  `class_map_hash` to dojo's `class_mapping_hash`. Keep SATI-specific
  `preprocessing_hash` and scope-aware `recombination_hash`; add
  `target_schema_hash` so detection task/coordinate semantics can evolve
  independently from class labels.
- **BREAKING** Standardize output topology. Prepared datasets remain reusable
  dataset products; training runs contain canonical `config/`,
  `checkpoints/`, `metrics/`, `figures/`, and backend-native artifacts;
  eval/infer runs contain canonical `config/`, `results/`, `metrics/`, and
  `figures/`; exports contain the portable model and metadata together.
- Normalize Ultralytics training output into sibling-style artifacts:
  content-stamped checkpoints, one-row-per-epoch `metrics/metrics.csv`,
  canonical figures, and an isolated backend-native directory rather than an
  opaque nested `train/` directory.
- Preserve SATI's existing domain contracts: prepared Parquet manifests,
  invertible slice affines, two-stage recombination, full-frame scoring, and
  explicit result schemas. Keep v2's zarr/streaming, rectangular slices,
  post-hoc recombination sweeps, and run sweeps as compatible future layers,
  not features of this change.
- Remove superseded development-stage config keys, hash names, paths, and
  layouts without compatibility shims.

## Capabilities

### New Capabilities

- `run-artifact-lifecycle`: Shared identifiers, path-template resolution,
  collision/replay rules, config provenance bundle, and canonical
  command-specific artifact directories.

### Modified Capabilities

- `config-and-cli`: Align config roots, local shadowing, root schema, inline
  overrides, command options, replay sources, and inspection behavior.
- `dataset-preparation`: Separate raw dataset input from the reusable prepared
  dataset product and resolve preparation outputs through the common lifecycle.
- `model-training`: Normalize stock-Ultralytics outputs into canonical
  checkpoints, epoch metrics, figures, and backend-native artifacts.
- `inference-evaluation-export`: Pass completed artifacts as command I/O and
  give infer, eval, and export distinct resolved output contracts.
- `results-provenance`: Align IDs, hash vocabulary and source projections,
  result placement, metadata, and portable model provenance.

## Impact

- Configuration/schema: `src/sati/config_schemas/`,
  `src/sati/config_defaults/`, project `configs/`, and all config-loader
  modules.
- CLI/orchestration: `src/sati/cli.py`, prepare/train/eval/infer/export and
  inspect entrypoints, plus reusable config-source and collision helpers.
- Artifacts/provenance: `src/sati/backends.py`, `train.py`, `evaluation.py`,
  `infer.py`, `export.py`, `hashing.py`, and manifest sidecars/readers.
- Tests and docs: config, hashing, CLI, backend, materialization, inference,
  evaluation, export, inspection, README, design documentation, reports, and
  DOX contracts.
- No new runtime dependency is required; `coolname` may be added if SATI uses
  dojo-compatible deterministic/fresh human-readable IDs rather than a small
  internal word-list implementation.
