## Context

SATI v1 is functionally complete: strict Pydantic schemas, Hydra Compose,
multi-resolution preparation, stock-Ultralytics training, full-frame inference
and evaluation, export, explicit-schema Parquet, and a compatibility-hash
family all exist. Its current interface was assembled one vertical slice at a
time, however, and exposes implementation accidents:

- local experiments under `./configs` require `--config-dir configs`, while
  dojo discovers that directory automatically;
- `data.source_path` means a raw COCO/YOLO source to `prepare`, a prepared
  dataset to `train`/`eval`, and ad-hoc imagery to `infer`;
- `model.weights` means training initialization and a completed model consumed
  by infer/eval/export;
- every command writes to `output.run_dir`, and config resolution itself
  creates that directory and writes one `resolved.yaml`;
- Ultralytics owns the nested `train/` artifact shape, while eval/infer write
  Parquet, metadata, and metrics directly at the run root;
- `run_id` is generated but does not determine a common run topology, and
  there is no human-readable deterministic partner for `config_hash`;
- several SATI hash names and source projections differ from dojo even where
  the concepts are identical.

Dojo's shipped P2 platform provides the sibling precedent: dash-free Hydra
overrides, sparse Typer options for command I/O, local config-group shadowing,
strict schemas, compose/validate/resolve separation, deterministic hashes,
fresh invocation IDs, rendered output blocks, collision guards, config
provenance, canonical checkpoint/metric/result/figure directories, and
portable inference contracts.

The target is a shared design language, not identical domain models. SATI
retains `prepare`, detection records, slice affines, two-stage recombination,
full-frame scoring, and its future zarr/streaming path. The separate
`sati-synth` interface is not changed.

## Goals / Non-Goals

**Goals:**

- Make `sati train experiment=...` and dotted inline overrides feel identical
  to their dojo equivalents.
- Give every command one unambiguous distinction between configuration and
  command I/O.
- Make prepared datasets reusable products rather than accidental run
  directories.
- Give mutating invocations predictable, inspectable, collision-safe artifact
  layouts.
- Pin the purpose, source, inclusion, and exclusion rules for every ID and
  hash.
- Normalize useful Ultralytics outputs without hiding the backend-native
  artifacts needed for debugging.
- Make a checkpoint or exported model carry enough information to reconstruct
  SATI preprocessing and model behavior without the producing run directory.
- Preserve structural seams required by v2 zarr pyramids, streaming training,
  rectangular slices, recombination sweeps, and run sweeps.

**Non-Goals:**

- Implement any v2 feature, including zarr, streaming training, rectangular
  slices, new recombination strategies, or sweeps.
- Change slicing geometry, affine semantics, NMS behavior, coordinate
  conventions, COCO/YOLO conversion semantics, or metric definitions.
- Make `sati-synth` use Hydra or SATI's config loader.
- Add remote storage, an experiment registry, Aim, MLflow, or a database.
- Preserve old config keys, hash names, generated paths, or artifact layouts.
- Make Ultralytics-native files part of SATI's stable public schema.

## Decisions

### 1. Align concepts and grammar, not command taxonomy

SATI keeps leaf commands `prepare`, `convert`, `train`, `eval`, `infer`, and
`export`, plus the `inspect` group. Training paradigm does not become a CLI
subcommand. This matches dojo's principle that configuration selects behavior
while allowing SATI's domain pipeline to remain legible.

All config-aware commands share one parser:

- dash-free `key=value`, `group=option`, `+key=value`, and `~key` tokens are
  Hydra overrides;
- dash-prefixed tokens are Typer options;
- `--config`, `--resolved-config`, and `--config-dir` select config sources;
- command inputs/outputs that are not reusable experiment policy are options.

Unknown dash-prefixed options fail as CLI errors rather than leaking into
Hydra. `--config`, `--resolved-config`, and `experiment=...` are mutually
exclusive root sources; ordinary inline overrides remain legal with any root
source.


### 2. Use packaged defaults plus an automatic local `./configs` shadow

Search precedence becomes:

1. repeatable explicit `--config-dir` entries, in CLI order;
2. project-local `./configs`;
3. packaged `sati/config_defaults`.

`experiment=<name>` is intercepted and promoted to the full primary root
`experiment/<name>`. The directory owning that root becomes Hydra's primary
directory and all remaining directories become search-path fallbacks. A
selector-less invocation may still use packaged `config.yaml` as a low-friction
starter, but authored project work uses named experiments.

This changes only the local shadow name; packaged defaults remain
`config_defaults` to distinguish immutable package resources from editable
project config.

Dojo's compositor also supports a selector-less `config.yaml` or `config.yml`
when one exists in its search path; its current checkout simply ships neither
under `./configs` nor packaged defaults. SATI deliberately ships its packaged
starter because it is useful for one-off preparation and does not weaken the
named-experiment workflow.

### 3. Split the root config by responsibility

The proposed top-level tree is:

```yaml
experiment:
  name: manual_fullres_bbox
task:
  type: object_detection
runtime:
  seed: 0
  run_id: "{coolname:noseed}"
  config_id: null               # resolved-only
  command: null                 # resolved-only

data:
  source:
    format: coco
    uri: ./datasets/source.json
    image_root: ./datasets/images
    split_file: ./configs/splits.toml
  prepared:
    uri: null                   # defaults to resolved prepare_outputs.dir

slicing: {...}
model:
  architecture:
    source: ultralytics
    name: yolo26n
  weights:
    source: library             # none | library | checkpoint
    uri: null                   # initialization only
  num_classes: 4
  imgsz: auto
training: {...}
recombination: {...}
results: {...}

output_root: ./runs
prepare_outputs: {...}
training_outputs: {...}
eval_outputs: {...}
infer_outputs: {...}
export_outputs: {...}
```

`data.source` describes ingestible raw data and is active for preparation and
conversion. `data.prepared.uri` describes the reusable SATI-prepared product
and is active for train/eval. When omitted, it resolves to
`prepare_outputs.dir`, allowing one experiment to support
`sati prepare experiment=x` followed by `sati train experiment=x`.

`model.weights` is only initialization policy. A completed checkpoint consumed
by infer/eval/export is command I/O and never overwrites this block.
`runtime.seed` replaces `training.seed` because it controls the invocation, not
only the optimizer loop.

All models remain strict and deferred features remain absent.

### 4. Resolve one active command context

The conductor receives a command context (`prepare`, `convert`, `train`,
`eval`, `infer`, `export`, or an inspect preview) and sequences:

1. select and compose the authored root;
2. apply inline overrides;
3. validate authored fields;
4. resolve derived model/data values and cheap identities;
5. compute the command-aware `config_hash` and deterministic `config_id`;
6. generate or preserve `run_id` according to source mode;
7. render the active writable output block and any read-only input-product
   locator it references;
8. validate resolved-only fields;
9. collision-check for mutating commands;
10. persist the config provenance bundle immediately before work begins.

Inactive invocation output blocks remain authored policy in the composed
artifact but do not acquire concrete `dir` values. Train/eval may resolve
`prepare_outputs.dir` only as the default read-only locator for
`data.prepared.uri`; that does not make it a writable output for those
commands. All output blocks are excluded from `config_hash`.
Resolved artifacts record `runtime.command`, preventing a training-resolved
artifact from being silently executed as an eval run.

`config/resolved.yaml` is the authoritative configuration of an instantiated
prepared product or invocation run. It is the fully composed, validated, and
resolved tree passed to the command worker; it is written atomically before
work and never subsequently changed. `config/composed.yaml` is retained as
pre-resolution provenance and an editable branching input, not as a competing
run configuration. `config/resolved.json` is a machine-readable rendering of
the same tree, not an independently authoritative configuration; `cli.txt`
and `overrides.txt` record invocation provenance only. Every hash-source
projection is derived from the authoritative resolved tree.

`--resolved-config` is exact replay: derived IDs and paths are preserved and
ordinary behavioral overrides are rejected. The only permitted mutations are
explicit lifecycle options such as `--resume` and `--clobber`. Branching a run
uses `config/composed.yaml` as a new `--config` source with new overrides.
The resolved-config loader revalidates the saved resolved tree against the
current schema but does not recompute its derived values, paths, or identities.

This deliberately differs from current SATI, which gives a loaded resolved
config a new `run_id`; exact replay is more understandable and makes the
persisted resolved config the unambiguous run record, as in dojo.

### 5. Use dojo-style path templates with command-specific output blocks

SATI owns `{...}` rendering rather than exposing OmegaConf `${now:...}` in run
paths. Supported tokens include any resolved dotted config path and the fixed
special tokens `{timestamp}`, `{coolname:<source>}`, `{coolname:noseed}`, and
`{source_run_dir}`. Relative path semantics match dojo:

- absolute paths remain absolute;
- `./path` is relative to invocation CWD;
- a bare relative top-level output dir is under `output_root`;
- a bare relative child dir is under its parent output dir.

Defaults:

```yaml
prepare_outputs:
  dir_template: "./.cache/prepared_datasets/{prepared_dataset_id}"
training_outputs:
  dir_template: "{experiment.name}/{timestamp}_{runtime.run_id}"
eval_outputs:
  dir_template: "{experiment.name}/{timestamp}_{runtime.run_id}_eval"
infer_outputs:
  dir_template: "{experiment.name}/{timestamp}_{runtime.run_id}_infer"
export_outputs:
  dir_template: "{source_run_dir}/exports/{timestamp}_{runtime.run_id}"
```

`prepare_outputs` is content-addressed and reusable; the initial packaged
default stores it in the project-local `.cache/prepared_datasets/` directory.
The other blocks are invocation-addressed. If an export's checkpoint has no
discoverable source run directory, `export_outputs` must be overridden or
falls back under `output_root/exports/...`.

### 6. Distinguish invocation IDs, content hashes, and compatibility hashes

All hashes use versioned recipes and SHA-256 over canonical JSON (sorted keys,
no insignificant whitespace, `%.12g` finite floats, named non-finite tokens,
and a `sha256:` prefix). IDs are human-facing strings; hashes are machine
identity. There is no `run_hash`. SATI SHALL use its existing `coolname>=5`
dependency for deterministic and fresh human-readable IDs, matching dojo.

| Field | Role | Source |
|---|---|---|
| `run_id` | fresh invocation ID | authored value/template, default `{coolname:noseed}` |
| `config_id` | deterministic config label | `{coolname:config_hash}` |
| `dataset_id` | optional human dataset name | source/prepared metadata only; no generated fallback |
| `prepared_dataset_id` | deterministic prepared-product label | `{coolname:prepared_dataset_hash}` |
| `config_hash` | command behavior provenance | active resolved behavioral config |
| `dataset_hash` | logical raw/full-frame dataset identity | canonical fullframe/annotation identity without slice pixels |
| `prepared_dataset_hash` | prepared product identity | dataset hash + preparation source |
| `target_schema_hash` | task/output compatibility | task type, box convention, target structure |
| `class_mapping_hash` | label compatibility | ordered class index/name mapping |
| `model_config_hash` | built-model compatibility | architecture, resolved input shape, output dimensions |
| `preprocessing_hash` | slice/input compatibility | effective level/slice/pixel preprocessing |
| `checkpoint_hash` | trained model identity | checkpoint file bytes after inference contract embedding |
| `recombination_hash` | merge behavior | strategy/version and scope-specific parameters |

`config_hash` includes:

- `experiment`, `task`, and all active command behavioral blocks;
- runtime values that can affect results, including the seed;
- active data/model/training/slicing/recombination/result policy.

It excludes:

- `output_root` and every `*_outputs` block;
- `runtime.run_id`, `runtime.config_id`, `runtime.command`, timestamps, and
  generated paths;
- resolved-only cache values and derived display fields;
- completed checkpoint and ad-hoc input options, which have their own hashes;
- inactive command-only blocks.

All `SatiPath` values in behavioral config continue to canonicalize to
`"<path>"`; data content is represented by `dataset_hash`. A local
initialization checkpoint contributes its byte hash to `config_hash`; a
library/hub asset name contributes its literal stable name.

`model_config_hash` excludes initialization weights, optimizer/training
settings, output paths, and recombination. It includes only the architecture
required to interpret trained tensors. Initialization provenance belongs to
`config_hash`; trained bytes belong to `checkpoint_hash`.

`dataset_hash` no longer hashes emitted slice files. It is a cheap logical
identity over canonical fullframes, annotations, splits, class indices, source
relative identities, and declared image metadata; it excludes slice policy,
materialized encoding, and pixels. `prepared_dataset_hash` adds the complete
preparation recipe: dataset hash, slicing level policy, overlap, size/area
gates, stretch policy, image format, materialization schema version, and any
future zarr/materialized mode. This split allows metric comparison by logical
dataset while preventing incompatible prepared trees from being reused.

`preprocessing_hash` keeps the effective pixel-to-model contract and excludes
training-only target gates. `recombination_hash` remains scoped:
`level_merge` covers within-level behavior and `final_merge` covers both
stages.

The rename from `class_map_hash` to `class_mapping_hash` is a clean break.
Sidecars retain exact `*_source` objects for verification.

### 7. Prepared datasets use a reusable product lifecycle

`prepare` performs a read-only ingest/identity preflight before creating its
output directory, computes the cheap logical dataset identity and preparation
recipe before emitting pixels, resolves `prepared_dataset_hash` /
`prepared_dataset_id`, and targets `prepare_outputs.dir`.

A complete prepared product contains:

```text
config/
  composed.yaml
  resolved.yaml
  resolved.json
  cli.txt
  overrides.txt
manifests/
  fullframes.parquet
  annotations.parquet
  slices.parquet
  slice_annotations.parquet
  _metadata.json
yolo/
  data.yaml
  images/...
  labels/...
prepare_manifest.json
```

`prepare_manifest.json` is the completion marker and records status, IDs,
hashes, schema versions, row/file counts, and artifact paths. A matching
complete product is reused after verification. An incomplete or mismatched
directory fails safely; `--clobber` explicitly rebuilds it.

This layout admits v2's `pyramid/` and streaming metadata without changing the
identity or lifecycle. `yolo/` can later become optional while manifests stay
canonical.

### 8. Invocation runs share a canonical envelope

Every train/eval/infer run writes:

```text
config/                  # provenance bundle
run_manifest.json        # always written, status updated atomically
```

The manifest begins with `status: running`, ends with `complete` or `failed`,
and records command, start/end timestamps, IDs, hashes, resolved paths,
artifact inventory, and a concise failure summary. It is a run index, not a
replacement for result metadata.

Non-empty invocation directories fail by default. `--clobber` deletes only
the already-resolved concrete output directory after explicit target
validation. `sati train --resolved-config ... --resume` is the only initial
resume surface; it requires `status: running|failed`, a canonical last
checkpoint, and matching resolved config. Other commands are rerun or
clobbered.

### 9. Normalize training artifacts and retain backend-native output

A completed training run contains:

```text
config/
checkpoints/
  best.<first6-checkpoint-hash>.pt
  last.<first6-checkpoint-hash>.pt
  checkpoint_metadata.json
metrics/
  metrics.csv
figures/
  ultralytics/
    results.png               # native Ultralytics filename
    PR_curve.png              # native Ultralytics filename
    ...                       # every other emitted native figure, unchanged
backend/
  ultralytics/                # non-contract native output
run_manifest.json
```

Ultralytics trains in `backend/ultralytics/`. After fit, SATI embeds the
portable inference contract, hashes the final bytes, and copies/hard-links
canonical best/last checkpoints using the hash-stamped naming convention.
`checkpoint_metadata.json` inventories checkpoints and repeats portable
metadata for inspection without importing torch.

`metrics/metrics.csv` has one row per completed epoch and stable SATI column
names, mapped from Ultralytics' CSV. Backend-specific extra metrics may be
retained with a namespaced column; native `results.csv` remains under
`backend/`.

When figures are enabled, SATI copies every figure emitted by Ultralytics into
`figures/ultralytics/` with its native filename unchanged. SATI initially keeps
Ultralytics' raster formats; canonical renaming, figure inventories, and
standalone HTML regeneration are deferred.

### 10. Make canonical checkpoints and exports portable

Before final checkpoint hashing, SATI embeds
`checkpoint["sati_inference_contract"]` containing:

- buildable model configuration without optimizer state;
- effective slicing/preprocessing configuration;
- default recombination policy;
- ordered class mapping and target schema;
- `config_hash`, `target_schema_hash`, `class_mapping_hash`,
  `model_config_hash`, and `preprocessing_hash`;
- SATI/schema versions and source run ID.

`sati infer`, `sati eval`, and `sati export` accept `--checkpoint PATH` and
reconstruct model/preprocessing defaults from this contract. The active
experiment supplies dataset/input and output policy; authored or inline
recombination overrides are allowed and produce a new
`recombination_hash`. A checkpoint without the contract fails with an
actionable error; development-stage legacy checkpoints are not supported.

Exports place the model and `inference_config.yaml` together under
`export_outputs.dir`. The sidecar is the portable inference contract plus
export format, exported-model byte hash (`model_hash`), source
`checkpoint_hash`, and artifact filename. `checkpoint_hash` continues to name
the source checkpoint, while `model_hash` names exported bytes.

### 11. Put command outputs under semantic subdirectories

Evaluation:

```text
config/
results/
  validation_results.parquet
  _metadata.json
metrics/
  metrics.json
figures/                    # enabled diagnostic plots
run_manifest.json
```

Inference uses the same envelope but may omit `metrics/`; its result table is
`results/predictions.parquet` with the same record taxonomy and sidecar.
Keeping distinct filenames avoids calling arbitrary input “validation.”

Export:

```text
config/
models/
  <artifact>
  inference_config.yaml
export_manifest.json
```

Result tables retain SATI's explicit schema and `record_type` taxonomy. This
change moves them; it does not replace them with dojo's classification schema.

### 12. Inspection uses the same config and provenance vocabulary

`sati inspect config` becomes config-first:

```text
sati inspect config experiment=v1/manual/fullres_bbox
sati inspect config --config path/to/root.yaml --format json
sati inspect config --resolved-config runs/.../config/resolved.yaml
```

It composes, validates, resolves, renders active paths and identities, reports
enabled artifacts, and performs no write. `sati inspect config-compare`
compares authored/composed/resolved configs grouped by hash source and
non-hash fields.

Artifact inspection remains positional (`dataset`, `slices`, `results`,
`model`) but standardizes machine output as `--format json`; the current
`--json` flag is removed.

### 13. v2 and synth remain explicit boundaries

The following v2 additions fit without changing this interface:

- zarr adds prepared-product artifacts and preparation-hash fields;
- streaming selects `training.data_mode` but consumes the same prepared
  manifests/pyramid identity;
- rectangular slices change preprocessing recipe versions, not artifact
  topology;
- recombination sweeps write variants under an eval/infer result envelope and
  group by `recombination_hash`;
- run sweeps expand resolved training configs and index their canonical run
  directories using future `sweep_outputs`.

`sati-synth` remains standalone. Its COCO output is selected by SATI
`data.source`; no shared runtime schema or imports are introduced.

## Risks / Trade-offs

- **[Hash semantics break all prior identities]** → This is intentional during
  development; bump every affected recipe and schema version together, add
  source-object golden tests, and provide no false equality across versions.
- **[Embedding metadata may perturb or invalidate Ultralytics checkpoints]** →
  Prove round-trip loading and prediction before adopting the checkpoint,
  hash only final bytes, and fail training finalization if the canonical
  checkpoint cannot reload.
- **[Pre-computing prepared identity without reading all pixels is weaker than
  a content hash]** → Record hash provenance, retain an optional future
  `dataset_content_hash`, and verify the completed manifest before reuse.
- **[Five output blocks enlarge the schema]** → Keep a shared strict base
  model and command-specific small extensions; inactive blocks are not
  resolved or hashed.
- **[Copying backend artifacts increases disk use]** → Prefer hard links on
  one filesystem and copies otherwise; the backend directory is explicit and
  can later have a retention policy.
- **[Resume behavior is backend-sensitive]** → Limit resume to canonical last
  checkpoints and exact resolved configs; do not promise cross-version resume.
- **[Config-first checkpoint defaults can conflict with authored overrides]**
  → Define precedence explicitly: checkpoint inference contract supplies
  model/preprocessing defaults, config supplies command data/output, and only
  recombination/result policy is overrideable for infer/eval.

## Migration Plan

1. Add the shared ID/hash primitives and golden tests under new recipe names.
2. Replace the strict root schemas and packaged/local YAML trees in one
   breaking change; remove legacy fields rather than aliasing them.
3. Extend the compositor/conductor/resolver with root-source exclusivity,
   command context, path templates, active output resolution, and read-only
   rendering.
4. Introduce lifecycle manifests, collision checks, and config provenance
   persistence; verify all target paths before destructive clobber.
5. Move preparation to the reusable prepared-product lifecycle and pin its
   identity/completion marker.
6. Stage Ultralytics output under `backend/`, embed the inference contract,
   normalize checkpoints/metrics/figures, and implement exact resume.
7. Move eval/infer/export into their canonical envelopes and switch completed
   model/input selection to command options.
8. Update inspection, conversion, all committed experiments, README, design
   docs, reports, DOX, and tests.
9. Run the full fast suite, lint, a real Ultralytics checkpoint round trip,
   and the existing coordinate tripwire.

Rollback during implementation is source-control rollback only. Existing
development artifacts are not migrated and may be deleted or regenerated by
the user.
