# SATI v1 — Scale-Aware Training and Inference (Minimum Viable Path)

**Status:** Design draft · **Last updated:** 2026-07-02
**Scope:** v1 is the smallest coherent system that proves SATI's core idea end-to-end.
Everything deliberately deferred is listed in §14 and specified in `DESIGN-DOC-v2.md`.

SATI is a package for object detection training and inference over **large, medium, and
small pixel-size objects inside very large images**. It applies one nominal slice size
(e.g. 1024×1024) against the full-resolution image **and against a stack of downsampled
full-frames**, so a single sampling window covers many object scales: small objects are
caught in full-res slices, large objects in slices taken from coarse levels. Inspired by
[SAHI](https://github.com/obss/sahi), generalized to multi-resolution and to **training**,
not just inference.

## 1. v1 Design Stance

v1 trades options for simplicity while keeping every *structural* decision that would be
expensive to retrofit:

| kept in v1 (structural) | simplified in v1 (behavioral) |
|---|---|
| Pydantic schemas + Hydra Compose + Typer CLI | one resolution-stack mode (automatic factor) |
| Parquet manifests, full backtraceability | no zarr — levels built in memory, slices **materialized to disk** |
| Tall `validation_results.parquet` + `record_type` | recombination is NMS-only at both stages |
| Compatibility-hash family | every slice is exactly nominal×nominal via **stretch**, never rectangular |
| Two-stage recombination with provenance | stock ultralytics — zero custom datasets/trainers/losses |
| Invertible per-axis affine per slice | no sampling knobs (all slices used once; all negatives kept) |

The load-bearing invariant carried from v1 into v2: **every emitted slice records an
invertible per-axis scale+translate affine to full-res coordinates.** v1's stretch and
v2's native rectangular slices are both just values of that affine, so manifests,
recombination, and results schemas survive the upgrade unchanged.

### Non-goals for v1
- Zarr / on-the-fly slicing, custom dataloaders or trainers, custom loss.
- Explicit scale lists, short-side floors, sampling weights, `negative_sample_ratio`.
- Non-greedy NMM / LSNMS / favorability scoring / edge penalties in recombination.
- Parameter sweeps. Segmentation/keypoints. mmdetection.

---

## 2. Core Concepts & Glossary

- **Full-frame / full-res:** the original source image at native resolution; the unit input
  annotations describe and that final predictions/metrics are reported against.
- **Resolution level:** one entry in the downsample stack, with dims computed from full-res
  directly (floor, min 1 px) at the policy-defined scale. **Level index 0 always means
  full-res (scale 1.0)** — a policy whose stack starts below full-res (e.g.
  `start_then_geometric` with `start_scale < 1`) emits no level 0 and numbers its levels
  from 1, so the index keeps its meaning even when full-res isn't a slicing target.
- **Nominal slice size:** the configured square sampling window (`nominal_slice_size`, e.g.
  1024). In v1 **every emitted slice is exactly nominal×nominal** (see stretch rule, §6.1).
- **Overlap ratio:** fractional slice-to-slice overlap (e.g. 0.10), as in SAHI.
- **Stretch rule (v1):** an axis smaller than the nominal slice is resized (stretched) up to
  nominal length at slice emission; the anisotropic scale is recorded in the slice affine.
- **Recombination:** merging per-slice, per-level predictions into one set of full-frame
  full-res predictions (§6.2).
- **Manifest tables:** the Parquet ledger linking every slice/resize/annotation back to the
  full-frame source via invertible transforms (§4).
- **Bounding box convention:** internal boxes are top-left-origin, half-open `xyxy`:
  `[x1, y1, x2, y2)`; width `x2 − x1`, height `y2 − y1`.
- **Authored vs. resolved config:** *authored* is what a human writes (relative paths, `auto`
  sentinels); `<output>/config/resolved.yaml` is the immutable, fully rendered source of truth
  actually passed to the worker. `composed.yaml` is branchable provenance; `resolved.json`
  renders the same resolved tree.
- **Compatibility hash:** SHA-256 over canonical JSON of a chosen slice of the resolved
  config/dataset, giving every result row content-addressed provenance (§4.7).
- **Inference config:** the YAML exported alongside a checkpoint describing slice size,
  resolution stack, and recombination settings to reproduce at deployment (§10).

---

## 3. Configuration — Hydra ↔ Pydantic

**Pydantic schemas in `sati.config_schemas` are the single source of truth** for valid
options and function I/O typing. Hydra YAML fragments packaged in `sati/config_defaults/`
compose runtime configuration, validated against the schemas before use. If they disagree,
the schema wins.

### 3.1 Principles

- **Schemas are the contract; YAML only fills them in.** Validation happens once, at the
  boundary; everything downstream takes a typed model, never a raw dict/`DictConfig`.
- **Strict by default.** `extra="forbid"`: a typo'd YAML key is an error. **Absent =
  unsupported** — v2 features have no v1 schema fields, so configuring them fails validation
  instead of hitting a stub.
- **Compose small fragments.** One config group per subdirectory (`data/`, `model/`,
  `slicing/`, `recombination/`, `training/`), one option per file, plus an `experiment/`
  entry per runnable job. Experiments are **full config roots** (dojo-style,
  `# @package _global_` with their own defaults list): `experiment=name` on the CLI makes
  `experiment/name` the primary config, bypassing `config.yaml`; without a selector the
  packaged `config.yaml` remains the useful starter root. Named experiments carry
  mandatory `experiment.name` and are the normal project workflow.
- **Three exclusive roots.** Use `experiment=name`, an authored root via `--config`, or
  exact replay via `--resolved-config`; never combine them. Exact replay preserves command,
  IDs, hashes, and paths and accepts no behavioral overrides.
- **Authored vs. resolved.** Resolution is command-aware and side-effect-free. After
  collision checks, a mutating command atomically writes the complete `config/` bundle
  before work begins.
- **One entry point.** CLI, tests, and orchestrators all call the same
  `compose_and_resolve(...)`.
- **Deterministic identity.** The resolved config hashes to a stable `config_hash` (§4.7).

### 3.2 Modules

- `sati.config_schemas` — Pydantic models: `RootConfig` + sub-models, `Literal[...]` enums,
  `Field(ge=/le=)` bounds, `model_validator`s for cross-field invariants. Zero Hydra imports.
- `sati/config_defaults/` — immutable packaged Hydra YAML tree, loaded via
  `importlib.resources`. Search precedence is repeatable explicit `--config-dir`, automatic
  project `./configs`, then packaged defaults. Local groups shadow matching packaged groups
  while omitted groups fall through.
- `sati.config_loader` — compose → validate → resolve; a package mirroring dojo's
  structure: `compositor.py` (all Hydra usage), `resolver.py` (command, identities,
  `SatiPath`, templates, derived values), `conductor.py` (side-effect-free sequencing).
  Drives Hydra through the **Compose API**
  (`initialize_config_dir` + `compose`), never `@hydra.main` or multirun.

**Decided (2026-07-02):** packaged defaults are located by resolving
`importlib.resources.files("sati") / "config_defaults"` to a concrete filesystem path
handed to `initialize_config_dir`. If the resolved traversable is not a real directory
(zip-imported install), `compose_and_resolve` raises immediately with a clear error —
SATI does not support zipped installs. No `SearchPathPlugin`.

### 3.3 v1 config surface (complete)

The entire v1 option set — anything not listed has no schema field:

| key | type / default | meaning |
|---|---|---|
| `experiment.name` | string | mandatory for named experiment roots |
| `task.type` | `object_detection` | task selector; the coordinate convention (top-left, half-open `xyxy`) is a fixed `target_schema_hash` constant, not config |
| `runtime.*` | command/seed plus derived run/config identity | invocation context |
| `data.source.*` | format, URI, optional split/image roots, optional `dataset_id` | ingestible raw dataset |
| `data.prepared.*` | URI plus derived dataset/prepared identities | reusable prepared product |
| `slicing.nominal_slice_size` | int, `1024` | square sampling window |
| `slicing.overlap_ratio` | float `[0,1)`, `0.10` | applied on both axes |
| `slicing.level_policy.kind` | `auto` \| `start_then_geometric`, `auto` | level-stack policy |
| `slicing.level_policy.downsample_factor` | number > 1, `2` | pyramid factor for `auto` (each policy owns its own factor) |
| `slicing.level_policy.start_scale` | float `(0,1]` | first level scale for `start_then_geometric` |
| `slicing.level_policy.subsequent_downsample_factor` | number > 1 | next-level factor for `start_then_geometric` |
| `slicing.level_policy.stop_when_long_side_lte_nominal` | bool, `true` | terminal condition for `start_then_geometric` |
| `slicing.max_levels` | int, optional | hard cap on stack depth |
| `slicing.min_area_ratio` | float `[0,1]`, `0.20` | training-target visibility gate |
| `slicing.size_gate_px` | int, `16` | min shorter side at level, post-downscale (§6.1) |
| `slicing.image_format` | `png` \| `jpeg`, `png` | materialized slice codec |
| `model.architecture.*` | source/name plus resolved shape | buildable architecture |
| `model.weights.*` | `none` \| `library` \| `checkpoint`, optional URI | training initialization only |
| `model.num_classes` | int ≥ 1 | detector output classes |
| `model.imgsz` | int, optional (defaults to `slicing.nominal_slice_size`) | model input size |
| `training.*` | epochs, batch, lr, device | passed through to backend |
| `training.yolo_augment.*` | YOLO train augment kwargs, **off by default** | schema defaults deactivate every knob; compose `augmentation/ultralytics` for stock behavior; per-knob `null` defers that one knob to Ultralytics |
| `recombination.method` | `nms` \| `greedy_nmm`, `nms` | suppress, or union the cluster envelope |
| `recombination.match_metric` | `intersection_over_union` \| `intersection_over_smaller`, IoS | IoU cannot match a clipped fragment (§6.2) |
| `recombination.within_level_threshold` | float, `0.8` | Stage-1 match threshold (metric-neutral name) |
| `recombination.cross_level_threshold` | float, `0.8` | Stage-2 match threshold |
| `recombination.max_merge_growth` | float ≥ 1, `8.0` | envelope area cap, ×winner; inert under `nms` |
| `recombination.confidence_threshold` | float, `0.05` | applied before Stage 1 |
| `recombination.class_agnostic` | bool, `true` | matching mode for both stages |
| `results.cache_slice_predictions` | bool, `false` | persist `slice_prediction` rows (§4.5) |
| `output_root` | path | parent for bare relative output templates |
| `prepare_outputs.*` | dir/template and artifact leaves | reusable product envelope |
| `training_outputs.*` | dir/template and backend/checkpoint/metric/figure leaves | training envelope |
| `eval_outputs.*`, `infer_outputs.*`, `export_outputs.*` | command-specific envelopes | active command output |

---

## 4. Data Model

Source of truth: **Parquet manifests + materialized slice images on disk**. COCO and YOLO
are import/export adapters only.

### Coordinate conventions

One internal geometry convention everywhere:

- origin at the top-left pixel corner; `x` right, `y` down;
- coordinate domains continuous and half-open: `[0, width) × [0, height)`;
- boxes half-open `xyxy`; width `x2 − x1`, height `y2 − y1`, area `w·h`;
- transformed boxes stored as floats; image sizes, slice origins, slice dims as ints.

COCO `xywh` and YOLO normalized `cxcywh` exist only in adapters (plus convenience export
columns); `xyxy` is authoritative for clipping, affines, IoU, and recombination.

### 4.1 `fullframes.parquet`

Source of truth for image membership, dimensions, and split — including zero-annotation frames.

| column | type | notes |
|---|---|---|
| `fullframe_id` | str | canonicalized to string at ingest |
| `source_uri` | str | path/URI of the source image |
| `dataset_split` | str | `training` \| `validation` \| `holdout` |
| `fullres_wh` | int×2 | native width, height |
| `color_mode` | str | `RGB` or `grayscale` |

### 4.2 `annotations.parquet`

Canonical **full-frame, full-res** ground truth; no slice semantics.

| column | type | notes |
|---|---|---|
| `fullframe_id` | str | |
| `ann_id` | int | |
| `class_id` / `class_name` | int / str | |
| `bbox_xyxy` | float×4 | authoritative `[x1, y1, x2, y2)` |
| `bbox_w` / `bbox_h` | float | derived |
| `bbox_rel_cxcywh` | float×4 | YOLO-style convenience, normalized to full-frame |

### 4.3 `slices.parquet`

One row per emitted slice, partitioned by `fullframe_id` / `level`. **`slice_id` is unique
across all levels of a `fullframe_id`**, so `(fullframe_id, slice_id)` identifies a slice;
`level` is redundant-but-stored for queries.

| column | type | notes |
|---|---|---|
| `fullframe_id`, `level`, `slice_id` | str, int, int | level 0 = full-res |
| `dataset_split` | str | copied from `fullframes.parquet` |
| `scale`, `level_wh` | float, int×2 | requested scale; realized level dims |
| `slice_xy`, `slice_wh` | int×2, int×2 | window origin/size **in level coords** (pre-stretch) |
| `emitted_wh` | int×2 | always `(nominal, nominal)` in v1 |
| `is_whole_level` | bool | entire level covered by this one slice |
| `stretched_axes` | list[str] | `x`/`y` axes that were stretched to nominal |
| `slice_uri` | str | materialized image path, relative to the dataset root |
| `affine_slice_to_level` | float×4 | `[sx, sy, tx, ty]`: emitted-pixel → level coords |
| `affine_slice_to_fullres` | float×4 | `[sx, sy, tx, ty]`: emitted-pixel → full-res coords |

Both affines map **emitted-slice pixels** (post-stretch) outward:
`x_level = sx·x_slice + tx`. For an unstretched axis `sx = 1`; for a stretched axis
`sx = slice_w / emitted_w < 1`. `affine_slice_to_fullres` composes the level→fullres scale
(`fullres_w / level_w` per axis) on top. Every transform SATI ever applies to pixels is
captured by these two rows, so predictions and annotations round-trip exactly.

### 4.4 `slice_annotations.parquet`

Derived projections; rows link to `annotations.parquet` via `parent_ann_id`.

| column | notes |
|---|---|
| `slice_ann_id`, `parent_ann_id`, `fullframe_id`, `level`, `slice_id` | keys |
| `bbox_fullres_xyxy` | canonical box, copied for traceability |
| `bbox_level_xyxy` | box mapped to the level |
| `bbox_slice_xyxy` | clipped box in **emitted** (post-stretch) slice pixels |
| `visible_area_frac` | clipped area ÷ level-box area |
| `truncated_edges` | subset of `top`/`bottom`/`left`/`right` |
| `dropped_by_size_gate`, `dropped_by_min_area_ratio` | why a target was excluded |

Dropped rows are **kept in the table** with their drop flags set — the training exporter
skips them, but the ledger stays complete for audits.

### 4.5 `validation_results.parquet`

Offline eval/inference artifacts in **one tall table discriminated by `record_type`**, with
an **explicit hand-written PyArrow schema** (never Arrow inference — it breaks on
null-leading and list columns). Build the table **column-by-column against the declared
schema**: a row-dict that omits a column yields a typed null; unknown keys are dropped.

`record_type` mirrors the pipeline stages:

- `slice_prediction` — one raw detection per row, `bbox_slice_xyxy` in emitted-slice pixels;
  written **only when `results.cache_slice_predictions` is on**.
- `level_merge` — one detection after Stage-1 merge; `bbox_level_xyxy` in level coords.
- `final_merge` — one detection in full-frame full-res coords (`bbox_fullres_xyxy`); the only
  kind scored against ground truth. Carries contributing `source_levels`.
- `frame_metadata` — one row per source frame (dims, split, `source_uri`).

Every row carries: `fullframe_id`, `dataset_split`, `run_id`, the hash family (§4.7),
`merge_strategy_name` + `recombination_hash` on merge rows, and `schema_version`.

A **`_metadata.json` sidecar** (validated by its own strict Pydantic model) holds
constant-for-the-table semantics: the class-index→name map, column descriptions,
`schema_version`, and each compatibility hash **paired with the `*_source` object it was
computed from** so every hash is independently verifiable. Do not `sort_keys` when dumping
the class map — numeric string keys (`"0"`, `"1"`, `"10"`) must keep insertion order. (This
is the opposite of the hash serializer's setting — see §4.7 — so the sidecar writer and the
hash function must not share a JSON-dump helper.)

### 4.6 Dataset splits

User-provided splits only. Resolved at `prepare`, stored in `fullframes.parquet`, copied
into `slices.parquet`.

- COCO inputs may provide separate train/val/holdout files; YOLO datasets their existing
  layout.
- An optional TOML file (`training = [...]`, `validation = [...]`, `holdout = [...]` lists of
  `fullframe_id`s) disambiguates or overrides. A TOML id missing from the source is an
  **error**; a source frame absent from the TOML is dropped with a **warning**.

### 4.7 Compatibility hashes

All hashes are **SHA-256 over canonical JSON** of a deliberately chosen source object,
using the same mechanics as the sibling `image-classifier-dojo` project so equal source
trees hash byte-identically in both tools: keys sorted, compact separators, floats
normalized through `%.12g` then rendered by `json.dumps` (so `2.0` serializes as `2.0`),
non-finite floats tokenized, `sha256:` prefix. Each source carries a `version` field (not
an inline recipe string); `config_hash` carries no version token — its identity is its
content. Sources project in the config's **own bare key namespace** (field names, no
`recipe`/`command`/`config` wrapper). What each hash *excludes* matters as much as what it
includes.

**Gates** — hard preconditions; a mismatch refuses the operation, no model load needed:

| hash | digests | gates |
|---|---|---|
| `class_mapping_hash` | ordered class-index→name map | comparing **or** ensembling |
| `target_schema_hash` | task `type` plus the target/coordinate convention (top-left, half-open `xyxy`) as a versioned constant, excluding labels | target compatibility |
| `dataset_hash` | logical frames, annotations, splits, classes, source identity; no pixels/slices | metric comparison |
| `prepared_dataset_hash` | logical dataset plus complete preparation recipe | product compatibility |
| `preprocessing_hash` | nominal slice, stack, overlap, stretch policy | ensembling raw predictions |

**Provenance / group-by axes** — identifiers, not gates:

| hash | digests |
|---|---|
| `config_hash` | active-command behavior; no outputs, generated IDs, completed command inputs, or inactive fields |
| `model_config_hash` | buildable architecture/input/output shape; no initialization weights |
| `checkpoint_hash` | canonical `.pt` bytes after embedding the inference contract |
| `model_hash` | exported portable-model bytes |
| `recombination_hash` | strategy name + per-strategy version + match/merge params (metric, thresholds, `max_merge_growth`, `class_agnostic`, confidence), scoped per record_type: within-level config on `level_merge`; within-level **+** cross-level on `final_merge`. Both scopes' sources are stored in the `_metadata.json` sidecar so a row's value is independently verifiable. |

**Decided (2026-08-03):** field exclusion is type-driven, never a hand-maintained list.
Every path-typed schema field is declared as `SatiPath` (an `Annotated[Path, ...]` alias)
and every generated/derived identity as `HashDerived`; the source builder walks the
Pydantic model tree and **pops** any `SatiPath` or `HashDerived` field from the source
before serialization (so a portable-path-only difference never moves a behavioral hash,
and `config_hash` is independent of generated identities — including dataset identities,
which is what lets `sati inspect config` resolve without ingesting the dataset). A plain
`Path`/`str` holding a path in a schema is a bug by definition. Guard tests: (1) hash one
authored config under two fake roots and assert equality; (2) walk `RootConfig` and assert
no field annotates bare `pathlib.Path`; (3) assert every generated identity is `HashDerived`.

**Initialization exception:** a local initialization checkpoint contributes its byte hash
to `config_hash`; a stable library asset contributes its literal name. Initialization is
excluded from `model_config_hash`.

### 4.8 Materialized slice store (on-disk layout)

`prepare` writes a reusable product under `prepare_outputs.dir` (default
`./.cache/prepared_datasets/<prepared_dataset_id>`):

```
<prepared_product>/
  config/               composed.yaml, resolved.yaml, resolved.json, cli.txt, overrides.txt
  prepare_manifest.json
  manifests/            fullframes.parquet, annotations.parquet,
                        slices.parquet, slice_annotations.parquet, _metadata.json
  yolo/
    data.yaml           generated; consumed verbatim by ultralytics
    images/{training,validation}/   <fullframe_id>_L<level>_S<slice_id>.png
    labels/{training,validation}/   matching .txt, YOLO cxcywh in emitted-slice pixels
```

**Pixel-source seam (settled 2026-07-03):** inside `sati.materialize`, pixels flow only
through `build_level_pyramid(frame) → levels` and `emit_slice(level_pixels, slice_spec)`.
Emission, labels, and manifests consume those two functions and nothing else, so v2 can
swap the in-memory pyramid for a zarr-backed one without touching them.

The YOLO tree is a **derived artifact** of the manifests (regenerable), but v1 writes it
during `prepare` so `train` is a pure stock-ultralytics call. Holdout frames get manifests
and slices but are excluded from `data.yaml`. Zero-target slices are written with empty
label files (ultralytics treats them as negatives); v1 keeps **all** of them.

---

## 5. Pipeline

```
prepare  →  train  →  eval  →  export
prepare  →  infer  →  (results)
```

- **prepare** — ingest COCO/YOLO, resolve splits, build levels in memory per frame, emit
  slices to disk, write manifests + the YOLO training tree (§4.8).
- **train** — stock ultralytics over the materialized YOLO tree (§8).
- **eval** — run sliced inference over a split, recombine, score vs full-frame full-res GT (§9, §11).
- **infer** — SAHI-like sliced inference on new imagery; slices held in memory only (§9).
- **export** — checkpoint + inference-config sidecar (§10).
- **convert** — standalone utility (not a stage): format translation between COCO/YOLO and
  the canonical manifests, both directions. No slicing, no level building, no YOLO training
  tree — use `prepare` to start a run, `convert` for interchange.
- **inspect** — standalone **read-only** utility (not a stage): summarize any SATI artifact
  from its manifest/sidecar without loading a model or touching pixels (§5.1).

Memory note: `prepare` and `infer` hold **one frame plus its level stack** in memory at a
time (a 7680×4320 RGB frame with factor-2 levels ≈ 130 MB — fine). Frames are processed
sequentially; no whole-dataset buffering.

### 5.1 `inspect` (read-only introspection, `sati.inspect`)

A Typer sub-app with one subcommand per artifact, each reading only manifests, sidecars, and
file headers — never a model forward pass, never a pixel decode (except an opt-in thumbnail).
It **never writes** to the inspected store. Every subcommand prints `rich` tables to stdout
and accepts `--format text|json`.

| subcommand | reads | reports |
|---|---|---|
| `inspect config [experiment=…\|--config …\|--resolved-config …]` | authored/composed/resolved config | validation, derived values, IDs, hashes, active output, and resolved tree |
| `inspect config-compare <left> <right>` | two config artifacts | differences grouped by contributing hash source plus `other` |
| `inspect dataset <root>` | `fullframes` + `annotations` | frame & annotation counts, per-split counts, class map, per-class instance counts, image-dimension and box-size distributions, count of zero-annotation frames |
| `inspect slices <root>` | `slices` + `slice_annotations` | per-level slice & object counts, `dropped_by_size_gate` / `dropped_by_min_area_ratio` tallies, stretched-axis frequency, on-disk size of the materialized YOLO tree, and a spot-check that every `slice_uri` resolves |
| `inspect results <parquet>` | table + `_metadata.json` | `record_type` row breakdown, `run_id`s and hash families present, sidecar validation (schema_version, **each hash re-verified against its paired `*_source`**), and any scored metrics |
| `inspect model <path>` | checkpoint metadata or exported inference sidecar | portable contract and byte identities without loading tensor/model payloads |

`inspect config`'s hash preview and `inspect results`'s hash re-verification share one code
path with `sati.hashing`, so "what the hash *would* be" and "what the hash *was*" can never
drift. `inspect` is **retrospective** — it reports artifacts that already exist; contrast
`prepare --dry-run` (v2), which is **prospective** — it predicts slice counts *before*
building. Keep the two distinct so their outputs don't blur.

> [!NOTE]
> Undecided: whether `--json` emits a stable, versioned schema (so scripts/CI can depend on
> `inspect` output) or stays a best-effort dump. If artifacts are meant to be gated in CI
> ("fail if any level has zero slices", "fail if a results hash fails re-verification"),
> the JSON needs its own tiny Pydantic contract and a `--check <expr>` exit-code mode.
> Decide when the first automation consumer appears; ship the human tables first.

---

## 6. Tiling (`sati.tiling`)

SATI rolls its own slicing (SAHI is a design reference, not a dependency — its API can't
express multi-level slicing or our affine bookkeeping). SAHI's `overlap_ratio` and
`min_area_ratio` **semantics** are kept identical so migrating users get predictable
behavior. `shapely` is used for clipping/IoU, as SAHI itself does.

### 6.1 Slicing

**Level stack.** In `auto` mode, level 0 is full-res; each subsequent level divides both
dimensions by `level_policy.downsample_factor` (floor, min 1 px) until
`max(level_w, level_h) ≤ nominal_slice_size` — terminate on the *longer* side — or
`max_levels` is hit. In `start_then_geometric` mode, the stack starts at
`level_policy.start_scale` of full-res and each subsequent level divides by
`level_policy.subsequent_downsample_factor`; by default it uses the same longer-side
terminal condition, with `max_levels` still available as a hard cap. Level numbering
follows the §2 convention: index 0 ⇔ full-res, so a below-full-res stack is numbered
from 1. Both requested and realized (post-floor) scales are recorded. Downsampling uses
OpenCV `INTER_AREA`.

**Per-axis tiling with stretch.** For each level, per axis:

- axis `≥ nominal_slice_size` → overlapping tiles of nominal length. Stride is
  `nominal_slice_size · (1 − overlap_ratio)`. **Last-tile rule:** the final tile is clamped
  flush to the far edge (`origin = max(0, axis_len − nominal)`), so it keeps full nominal
  length and simply overlaps its neighbor more. No short tiles, no padding. Deterministic
  given `(axis_len, nominal, overlap_ratio)`.
- axis `< nominal_slice_size` → one span covering the axis, **stretched** to nominal length
  at emission (`cv2.resize`, per-axis). The terminal whole-frame level is the degenerate
  case: the entire level stretched to nominal×nominal.

**Sharp-source rule (settled 2026-07-03).** A stretched window is never upscaled from its
own (already-downsampled) level's pixels — that made whole-level slices visibly blurry.
Instead, emission maps the window to the smallest level whose mapped size is ≥ emitted
size on both axes (falling back to the full-res frame when none is) and resizes *down*
with `INTER_AREA`. Geometry, affines, and truth are unchanged — only pixel provenance
improves. Recorded in `preprocessing_hash` as `stretch_policy:
stretch-to-nominal-sharp-source/v2`.

Every emitted slice is therefore exactly nominal×nominal — square inputs, uniform batches,
zero letterbox logic — at the cost of anisotropic distortion on stretched axes, which is
(a) recorded in `stretched_axes` + the affines, and (b) bounded: a stretched axis is off by
at most `downsample_factor`× relative to isotropic. v2 replaces stretch with native
rectangular slices.

**Annotation projection.** Boxes map fullres → level (scale) → emitted slice (affine
inverse), clip to the slice, then:

- **Size gate:** drop as a training target any annotation whose shorter side, *evaluated at
  the level before stretch*, is `< size_gate_px` (default 16 px ≈ 2 cells of YOLO's stride-8
  P3 head; commonly raised to 32). Gated targets remain at finer levels and always count in
  final recombined metrics.
- **`min_area_ratio`:** keep as a training target iff
  `visible_area_frac = clipped_area / level_box_area ≥ min_area_ratio` (default 0.20; `0.0`
  keeps any overlap, `1.0` only fully contained objects).

Both exclusions are recorded per row in `slice_annotations.parquet` (§4.4), never silently.

### 6.2 Recombination — two stages, configurable matching + optional merging

Both stages are behind a `Strategy` interface (the cross-scale heuristic is the project's
research axis). Each stage clusters greedily by score, ties broken by input index, under a
configurable **match metric**, and then either discards the non-winning members
(`method: nms`, the default) or replaces the winner's box with the cluster envelope
(`method: greedy_nmm`). The winner's score and class are kept in both modes.

**Match metric.** `intersection_over_smaller` is the default, not IoU. A slice-clipped
fragment is a *strict subset* of its whole detection, so their IoU degenerates to the area
ratio and can never clear a usable threshold — a 24px clip of a 100px object scores IoU
0.24. IoS scores it 1.0. This is why SAHI defaults to IoS and why SATI does. Thresholds are
named `within_level_threshold` / `cross_level_threshold` rather than `*_iou` precisely
because the metric is a choice; `0.8` is tuned for IoS, and an IoU config should set `0.5`.

**Class-agnostic by default — a deliberate mAP trade.** One object must not keep two
detections that differ only in class. Because both stages are score-ordered greedy, the most
confident class wins with no extra tie-break logic. `merged_class_ids` on each merge row
records every absorbed class, winner's first — without it a collapse makes the losing class
unrecoverable, and near-ties do occur.

Measured cost: **-14.3% mAP50_95**, concentrated as ~-36% AP in the lowest-confidence
classes, because the absorbed cross-class boxes are true positives extending the per-class
recall tail rather than duplicates. This is accepted on purpose: the primary objective is
clean output for a human reviewer, for whom a duplicate box with a disagreeing class is
noise regardless of COCO AP. Consequently **mAP is not comparable across the
`class_agnostic` setting** — under `true` it measures output policy as much as model quality.
Set `class_agnostic: false` for an honest benchmark (`v1/combo1/default` is pinned there);
the review-objective metric is the collapsed-duplicate count reported by
`sati inspect results`.

**Stage 1 — within-level.** Per `(fullframe_id, level)`: apply `confidence_threshold`; map
raw boxes through `affine_slice_to_level` into shared level coordinates; cluster and combine
at `within_level_threshold`. Winners become `level_merge` rows (`bbox_level_xyxy`) recording
contributing `slice_id`s, `merged_class_ids`, `merge_strategy_name`, and a
`recombination_hash` over within-level params only.

**Stage 2 — cross-level.** Per `fullframe_id`: lift each `level_merge` box to full-res by
the level's realized scale; cluster and combine at `cross_level_threshold`, **ranked by raw
confidence** (no favorability weighting in v1). Winners become `final_merge` rows
(`bbox_fullres_xyxy`) recording `source_levels`, `merged_class_ids`, and a
`recombination_hash` over within-level + cross-level params.

**Bounded envelopes — a deliberate departure from SAHI**, which unions unconditionally. A
merge takes the envelope only when its area is at most `max_merge_growth` (default `8.0`)
times the winner's own area; otherwise the winner keeps its box. Matched members are
consumed either way, so a blocked merge degrades that one cluster to `nms` behavior and the
detection count is never affected. The motivation is measured: a large false-positive blob
nests around genuine small detections (IoS 1.000, IoU 0.002), and because a small box
outscores the blob, an unbounded union replaced a 137×102 box with a 2658×1206 one — 248×
inflation. Observed legitimate merges needed ≤3.2× and a slice-clipped fragment needs
`1 / area_ratio` (4.17× for a 24px clip of a 100px object), so `8.0` sits mid-gap: it admits
every observed legitimate merge and still blocks the blob by 31×. The bound's accepted cost
is that a fragment smaller than `1 / max_merge_growth` — 12.5% at the default — is suppressed
rather than restored. The bound is applied per merge, so end-to-end inflation across both
stages is bounded by `max_merge_growth ** 2`. Cap activations are counted and reported so that is observable, not invisible.

**Not idempotent.** Matching evaluates against the *original* winner box in a single pass
(SAHI's behavior, and what keeps the stage deterministic), so feeding `final_merge` boxes
back through a stage can merge further. Nothing in the pipeline does this; no test asserts
round-trip stability.

Superseded v1 note: the previous mitigation for edge fragments — configure
`overlap_ratio ≥ largest_expected_object / nominal_slice_size` — was inadequate on its own,
because overlap changes *whether* a whole detection exists, not whether the metric can match
a fragment to it. Overlap is still required for a whole detection to exist anywhere; IoS is
what then deduplicates against it. Stage-2 confidence ranking remains unweighted until v2's
favorability/edge machinery.

---

## 7. Backend (`sati.backends`)

A minimal `Backend` protocol — `train(data_yaml, cfg)`, `predict(batch, cfg)`,
`export(ckpt, cfg)` — with one implementation: **ultralytics, used stock**. v1 never
subclasses `DetectionTrainer`, never injects a dataset, never touches the loss. The
protocol exists so v2's custom-dataset backend and any later framework slot in without
touching callers.

`predict` receives nominal×nominal uint8 arrays. Because slice size equals the model
`imgsz`, ultralytics applies no letterbox and returned boxes are already in emitted-slice
pixels — no preprocessing inversion needed in v1.

**Implemented (2026-07-03):** ultralytics pinned `>=8.3,<9` in `pyproject.toml`; the
tripwire test (`test_ultralytics_coordinate_tripwire`, run with `SATI_RUN_TRIPWIRE=1`)
asserts predictions on a nominal-size slice come back in-bounds of the slice's own pixel
grid with no letterbox. Verified against ultralytics 8.4.86 + yolo26n.pt.

---

## 8. Training

`sati train` verifies `data.prepared.uri` identities and runs stock Ultralytics over its
generated `yolo/data.yaml`. Native output stays under `backend/ultralytics/`. SATI embeds
the portable inference contract before hashing best/last checkpoints and exposes:

```
<training_run>/
  config/                       # resolved.yaml is authoritative
  run_manifest.json
  checkpoints/{best,last}.<hash6>.pt
  checkpoints/checkpoint_metadata.json
  metrics/metrics.csv
  figures/ultralytics/          # native filenames, when enabled
  backend/ultralytics/          # non-contract native output
```

Deliberately absent in v1: per-level sampling weights and `negative_sample_ratio`. Coarse
levels contribute few slices, so large-object scales may be under-represented — watch the
per-level counts `prepare` logs and the per-level eval breakdown (§11) to see whether v2's
sampling knobs are needed for your data.

Training-time augmentation is **deactivated by default** so a run is deterministic unless
augmentation is explicitly requested.

**Decided (2026-08-03):** SATI does not inherit Ultralytics' stock augmentation. Every
`training.yolo_augment.*` knob defaults (in the schema) to its off-value, so omitting the
block yields no augmentation — the opposite of Ultralytics' defaults. Two packaged config
groups make the choice explicit and swappable: `augmentation/default` (all off; also the
`sati init` template) and `augmentation/ultralytics` (stock 8.4.x values, mosaic included),
opted into from an experiment's defaults list as
`- /augmentation@training.yolo_augment: ultralytics`. A single knob set to `null` defers
just that knob to Ultralytics' own default. `auto_augment` is classification-only in
Ultralytics (a detection no-op), so it has no off-value and stays unset.

*(Superseded 2026-07-03's "pass augmentation through untouched" decision — mosaic mixing
slices across levels is now off unless explicitly enabled; revisit with v2's per-level
sampling if eval shows scale confusion.)*

---

## 9. Inference (`sati.infer`)

Input: explicit `--checkpoint` and `--input`. Model, class, target, and preprocessing
defaults reconstruct from the embedded contract; config may change only recombination and
result policy.

Per frame: build the level stack in memory → emit nominal×nominal slices (same code path as
`prepare`, §6.1, minus disk writes) → `predict` per slice → recombine (§6.2) → append
`final_merge` (+ `frame_metadata`, + optional cached `slice_prediction`) rows to
`results/predictions.parquet` → optionally export final boxes as COCO JSON and/or a YOLO
prediction tree. An optional debug flag materializes slices + per-slice overlays for
inspection.

`eval` (§11) is `infer` over a prepared split plus scoring — one engine, two commands.

---

## 10. Export

`sati export --checkpoint ...` wraps Ultralytics export and writes the model plus a complete
**`inference_config.yaml`** under `export_outputs.models.dir`. It retains the source
`checkpoint_hash` and adds `model_hash` over exported bytes.

v1's all-square-slices rule has a quiet payoff here: exported engines get one fixed input
shape (`nominal × nominal`), so ONNX/TensorRT static shapes just work. (v2's rectangular
slices reopen this — see v2 doc.)

---

## 11. Evaluation (`sati.evaluation`)

Metrics are computed on **`final_merge` predictions vs full-frame full-res ground truth** —
never per-slice. pycocotools-compatible mAP/PR, overall and per-class, is written to
`metrics/metrics.json` alongside `results/validation_results.parquet`; every score traces
to exact slices/levels via provenance
columns.

**Per-level breakdown (v1 = diagnostic, not mAP):** for each level, the count and
confidence distribution of `final_merge` predictions whose `source_levels` include it, plus
recall of GT boxes binned by size. This detects a starved scale without inventing new
metric semantics.

> [!NOTE]
> Full per-scale mAP needs an attribution rule — a `final_merge` prediction has
> `source_levels`, but a GT box has no level, so "per-level mAP" is ill-posed until you
> choose: (a) score predictions-from-level-L against all GT, (b) bin GT by the levels whose
> size gate admits them, or (c) both as separate views. Deferred to v2; v1 ships the
> diagnostic above.

Expected and documented: per-slice training loss and recombined-frame metrics will not
track one-to-one. That is intrinsic to SATI, not a bug.

---

## 12. Module Layout

| module | responsibility |
|---|---|
| `sati.config_schemas` | Pydantic contract for all options & I/O typing |
| `sati.config_loader` | compose→validate→resolve via Hydra Compose API |
| `sati/config_defaults/` | packaged YAML fragment tree |
| `sati.io` | image read/decode, COCO/YOLO adapters, path handling |
| `sati.manifests` | Parquet schemas + column-by-column writers + sidecar |
| `sati.hashing` | canonical JSON + the hash family (§4.7) |
| `sati.dataset_identity` | logical/prepared dataset projections |
| `sati.lifecycle` | config provenance, collision safety, lifecycle manifests |
| `sati.checkpoints` | embedded contracts and checkpoint finalization |
| `sati.tiling` | level stack, per-axis tiling + stretch, affines, annotation projection, recombination |
| `sati.materialize` | slice/label emission to disk + `data.yaml` generation |
| `sati.backends` | `Backend` protocol + stock-ultralytics impl |
| `sati.train` / `sati.infer` / `sati.evaluation` | orchestration per command |
| `sati.inspect` | read-only artifact summaries (§5.1); reuses `sati.hashing` for hash preview/verify |
| `sati.cli` | Typer entrypoints |

Testing spine (worth naming in the design): `sati.tiling` gets property-based tests —
random frame sizes/factors/overlaps → slice → project annotation → invert affine → assert
round-trip within float tolerance, and assert full coverage of every level by its tiles.
This module is pure geometry with no I/O, deliberately, so these tests are cheap.

---

## 13. CLI & Packaging

Typer, thin: each command parses args, calls `compose_and_resolve`, hands the typed
`RootConfig` to a CLI-free work function (`execute_train`, …). No logic in the CLI.

**The dash rule:** `--dashed` args are the limited Typer surface (`--config`,
`--config-dir`, `--resolved-config`, command I/O, format, resume, clobber); dash-free
Hydra grammar reaches composition. `--resolved-config` is exact replay and rejects
behavioral overrides; branch from `config/composed.yaml` with `--config`.

**Console script:** `sati` → `prepare` · `convert` · `train` · `eval` · `infer` · `export`
· `inspect` (a sub-app: `inspect config|config-compare|dataset|slices|results|model`, §5.1).

**Core deps:** `pydantic`, `hydra-core`, `omegaconf`, `typer`, `pyarrow`, `numpy`,
`shapely`, `opencv-python`, `rich`. **Extras:** `[ultralytics]`, `[dev]`. (No `zarr` in v1.)

---

## 14. Deferred to v2 (see `DESIGN-DOC-v2.md`)

Zarr pyramid store · on-the-fly slicing datasets · custom ultralytics dataset/trainer ·
native rectangular & whole-frame slices (stretch removed) + backend preprocessing capture ·
explicit scale lists + short-side floor · non-greedy NMM/LSNMS, favorability scoring, edge
proximity/penalty · post-hoc recombination sweeps · per-level sampling weights ·
`negative_sample_ratio` · parameter-sweep config generation · per-scale mAP semantics ·
resolution-aware loss and beyond (v2 §11).

## 15. v1 Pitfalls & Mitigations

- **Stretch distortion at coarse levels.** Bounded by the configured level policy; recorded
  in affines; objects there are large in slice pixels, which tolerates distortion best.
  Still a known accuracy tax v2 removes.
- **Storage blow-up.** Materializing every slice of every level roughly doubles-plus the
  dataset on disk (overlap × levels). Accepted for v1 simplicity; PNG default keeps it
  lossless. v2's zarr + on-the-fly removes it.
- **Confidence-only cross-level ranking.** Edge fragments can still outrank whole coarse
  detections. Since `recombination-match-and-merge` this costs the *box* rather than the
  detection: IoS matching deduplicates the pair, and `greedy_nmm` restores the whole extent
  when the fragment wins — up to `max_merge_growth`. Unweighted ranking itself waits for v2
  favorability.
- **Un-hashed per-slice NMS (open).** `backends.py` calls Ultralytics `predict()` without
  `iou` or `agnostic_nms`, so per-slice NMS silently runs at the library's defaults
  (`iou=0.7`, class-aware) — values absent from every config, every resolved artifact, and
  every hash. Two consequences: an Ultralytics default change moves detections with every
  digest unchanged, and the class-aware per-slice pass is what *emits* the cross-class
  duplicates §6.2 then collapses. Tracked as the `expose-backend-nms-params` change; named
  here so the gap is documented rather than merely unnoticed.
- **Scale starvation.** Few coarse slices per frame; no sampling knobs yet — monitor §11
  diagnostics.
- **Stale YOLO tree.** The materialized tree is derived from manifests; re-running `prepare`
  with changed slicing config must fully rebuild it. `data.yaml` embeds
  `preprocessing_hash`; `train` refuses on mismatch with the composed config.
- **Ultralytics API drift.** Version-pin + the §7 tripwire test.
