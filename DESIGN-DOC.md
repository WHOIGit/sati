# SATI — Scale-Aware Training and Inference

**Status:** Design draft · **Last updated:** 2026-07-01

SATI is a package for object detection training and inference over **large, medium, and
small pixel-size objects inside very large images**. It streamlines a multi-resolution
slicing workflow so that a single nominal sampling window can cover many object scales.

---

## 1. Overview & Motivation

Detecting objects whose sizes span orders of magnitude inside very large frames (e.g.
7680×4320 and up) is hard: downsampling the whole frame to the model's input size destroys
small objects, while running the model at native resolution misses the global context that
large objects need, and is often infeasible memory-wise. This project is inspired by [SAHI](https://github.com/obss/sahi).

**SAHI** (Slicing Aided Hyper Inference) addresses part of this by tiling the full-resolution
image into fixed-size overlapping slices, running inference per slice, and merging the
results — with an optional single full-image pass for large objects. SAHI uses **one** slice
size against **one** input resolution (+ optional full frame), and is only intended for inference.

**SATI generalizes this idea.** It applies a nominal slice size (e.g. 1024×1024) against the
full-resolution image **and against a configurable stack of downsampled full-frames**. A
single nominal slice size therefore samples multiple scales: small objects are caught in full-res
slices, large objects are caught in slices taken from coarse downsampled frames. This applies
to both **inference** (SAHI-like) and **training** (slices + scale-aware metrics/loss).

### Goals
- One nominal slice size covering many detection scales via a multi-resolution full-frame stack.
- Configurable resolution stacks (explicit scales or automatic downsampling by a factor).
- Fully **backtraceable** slicing/resizing/annotation via manifest tables.
- Training runs over slices; validation is scored against **fully recombined predictions vs
  full-frame, full-resolution annotations**.
- A SAHI-like inference command that takes a model + a nominal-slice/resolution config and emits
  final boxes in YOLO and/or COCO.
- Exported checkpoints/pt files ship **with an inference configuration file**.
- Backend agnostic - implements a clean abstraction that could later host different backends. 
  Initially only supports ultralytics.

### Non-goals (initially)
- Segmentation / keypoints (detection boxes only at first).
- Custom "scale-aware" training loss (use stock ultralytics at first — see §8).
- A production mmdetection backend (documented stub only — see §7).

---

## 2. Core Concepts & Glossary

- **Full-frame / full-res:** the original source image at native resolution; the unit that
  input annotations describe and that final predictions/metrics are reported against.
- **Resolution level:** one entry in the downsample stack (e.g. level 0 = full-res, level 1 =
  ½, …). Each level is a downsampled full-frame.
- **Nominal slice size:** the configured sampling window size (e.g. 1024×1024). In Phase 1 the
  nominal slice is **square** — a single scalar `nominal_slice_size` — so all termination and
  per-axis tiling comparisons use one value. Rectangular nominal slices are out of scope for Phase 1.
- **Actual slice size:** the real emitted slice/window size. It usually matches the nominal
  size, but may be smaller on axes where the resolution level is smaller than the nominal
  slice size.
- **Backend input size:** the tensor shape passed to the detector after backend preprocessing
  such as resize or letterbox.
- **Overlap ratio:** fractional slice-to-slice overlap (e.g. 0.25 = 25%), as in SAHI.
- **Manifest tables:** the backtraceable ledger linking every slice/resize/annotation back to
  the full-frame full-res source via invertible transforms.
- **Bounding box convention:** internal boxes use top-left-origin, half-open `xyxy` coordinates:
  `[x1, y1, x2, y2)`, with width `x2 - x1` and height `y2 - y1`.
- **Recombination:** merging per-slice, per-level predictions into a single set of full-frame
  full-res predictions.
- **Inference config:** the YAML emitted alongside an exported model describing the nominal
  slice size, resolution stack, backend preprocessing, and recombination settings to use at
  inference time.
- **Authored vs. resolved config:** *authored* is what a human writes (templates, relative
  paths, `auto` sentinels); *resolved* is the fully rendered, path-absolute, derived-value
  artifact that actually runs and is written to the run directory as `resolved.yaml` (§3.1).
- **Compatibility hash:** a SHA-256 over canonical JSON of a chosen slice of the resolved
  config / dataset (`config_hash`, `dataset_hash`, `recombination_hash`, …) that gives every
  result row content-addressed provenance (§4.9).

---

## 3. Configuration (Hydra) ↔ Schemas (Pydantic)

**Pydantic schemas in `sati.config_schemas` are the single source of truth** for which
options are valid and for function input/output typing. **Hydra** YAML fragments packaged in
`sati/config_defaults/` compose runtime configuration, which is validated against the schemas
before use. If the two ever disagree, the schema wins.

### 3.1 Config principles

- **Schemas are the contract; YAML only fills them in.** Every option a user can set exists
  because a schema field exists. Validation happens **once, at the boundary**; everything
  downstream takes a typed model, never a raw dict / `DictConfig`.
- **Strict by default.** Schemas forbid unknown keys (`extra="forbid"`): a typo in a YAML
  key is an error, not a silently ignored field. **Absent = unsupported** — unimplemented
  features have no schema field, so configuring them fails validation rather than hitting a
  stub.
- **Compose small fragments.** A run is assembled from single-responsibility fragments
  (`data/`, `model/`, `slicing/`, `recombination/`, …); users select and override fragments
  instead of hand-maintaining a monolith.
- **Authored vs. resolved.** *Authored* config is what a human writes (templates like
  `{run_id}`, relative paths, `auto` sentinels). *Resolved* config is what actually runs
  (templates rendered, paths absolute, derived values computed). The resolver turns the
  first into the second and writes `resolved.yaml` into the run directory so a run is
  exactly re-runnable without re-composition.
- **One entry point.** CLI, tests, and any orchestrator all call the same
  `compose_and_resolve(...)`, so composition/validation order can't drift.
- **Deterministic identity.** The resolved config hashes to a stable `config_hash` (§4.9).

### 3.2 Config modules

- `sati.config_schemas` — Pydantic models (the contract): `RootConfig` + sub-models,
  `Literal[...]` enums, `Field(ge=/le=)` bounds, discriminated unions, and `model_validator`s
  for cross-field invariants. Importable with zero Hydra dependency.
- `sati/config_defaults/` — the packaged Hydra YAML fragment tree (composable defaults): one
  config group per subdirectory, one option per file, plus an `experiment/` entry per runnable
  job. Loaded via `importlib.resources`; a local `./config_defaults` (or `--config-dir`)
  shadows it, so users override only what differs without copying the tree.
- `sati.config_loader` — compose → validate → resolve, exposing one public
  `compose_and_resolve`. Drives Hydra through the **Compose API** (`initialize_config_dir` +
  `compose`), **not** `@hydra.main` or `-m`/multirun, so SATI owns the process, the CLI, and
  any sweep expansion.

### Resolution stack configuration modes

1. **Explicit scales** — `[1.0, 0.5, 0.25, 0.125]`. Each scale is applied isotropically to the
   full-frame.
2. **Automatic factor** — start from level 0/full-res and repeatedly divide dimensions by a
   configured downsample factor (e.g. `2` or `4`) until the **whole full-frame level fits in a
   single nominal slice** — i.e. `max(level_dim) ≤ nominal_slice_size` (terminate by the
   *longer* side, not the shorter; see "Slice-dimension reduction" below for why). Optional
   `max_levels` and an optional **short-side floor** bound the stack for extreme aspect ratios.
   The short-side floor takes a value plus a configurable **mode**:
   - `stop` (default) — do not add a level whose downscaled short side would fall below the
     floor; the pyramid terminates at the previous level even if the long side hasn't reached
     the nominal slice.
   - `clamp` — still add the level, but clamp its short-side dimension at the floor (the long
     side keeps scaling by the factor). This trades a small aspect-ratio distortion on that one
     axis for keeping the coarsest level; the realized per-axis scale is recorded in pyramid
     metadata (§4.6) like any other level.

Unlike the automatic mode, an **explicit** scale list is used exactly as given: it is **not**
required to reach a whole-frame level. A list whose coarsest scale still leaves the frame larger
than the nominal slice is allowed — this intentionally lets the coarsest level keep tiling so the
"forest is not missed for the trees." Validation only rejects degenerate lists (empty, non-positive
scales, or scales `> 1.0`), not lists that stop short of the whole frame.

Explicit target resolutions are intentionally not supported in Phase 1. Levels are derived
from source dimensions plus scale, which preserves isotropic scale and avoids aspect-ratio
ambiguity. Level dimensions are deterministic: `level_w = floor(fullres_w * scale)` and
`level_h = floor(fullres_h * scale)`, clamped to a minimum of 1 pixel per axis. Because
flooring can make the realized scale differ slightly from the requested scale, both are
recorded in pyramid metadata.

> A dry-run tool can output the expected resolutions for a given input image(s).

### Slice configuration
- **Nominal slice size** (e.g. `1024×1024`).
- **Overlap ratio** (e.g. `0.10`), applied in height and width.
- **Training `min_area_ratio`** (default `0.20`): how much of an object must be visible inside
  a slice before it is included as a training target.
- **`negative_sample_ratio`** — controls how zero-target (background) slices are sampled during
  training. By default **all** zero-target slices are emitted and used as negatives; this knob
  sub- or over-samples them relative to positive slices (see §8). Independent of the per-level
  sampling weights, which govern scale coverage rather than positive/negative balance.
- **Recombination edge policy** (see §6.2 and §14): how near-edge predictions are weighted
  during cross-level recombination.

> a dry-run tool can show how many patches per level and in total would be created for given input image(s).

### Slice-dimension reduction
SATI does **not** pad (letterbox) or distort (stretch) frames to force a square slice. Instead,
tiling is applied **per-axis**:
- an axis `≥ nominal_slice_size` → overlapping tiles of nominal slice length;
- an axis `< nominal_slice_size` → a **single native, stride-aligned span** along that axis (the slice
  dimension is *reduced* to the frame's extent, not padded).

**Last-tile rule (tiled axes).** Stride is `nominal_slice_size * (1 - overlap_ratio)`. When the
axis length is not an exact multiple of the stride, the final tile is **clamped flush to the far
edge** — its origin is `max(0, axis_len - nominal_slice_size)` — so the trailing tile keeps full
nominal length and simply overlaps its neighbor more than `overlap_ratio`. No short/partial tiles
and no padding are ever emitted. This is deterministic given `(axis_len, nominal_slice_size,
overlap_ratio)`.

So slices become rectangular at coarse levels. In **automatic** mode the pyramid terminates at
the level where the **whole full-frame** is fed as one input rather than a slice; in **explicit**
mode the stack is exactly the listed scales, which may or may not include such a whole-frame level
(see "Resolution stack configuration modes"). This keeps scale **isotropic** (no aspect distortion), stores
no padded pixels, and keeps the slice affine a clean per-axis scale+translate (sole exception:
short-side-floor `clamp` mode above). Non-standard
(rectangular / whole-frame) slices are native to SATI. Backend adapters normalize them at the
framework boundary when needed (see §6.1) and record the preprocessing transform so model
outputs can be mapped back to slice space before recombination. Extreme aspect ratios (e.g.
11260×2280) are handled by the same rule — long axis tiled, short axis reduced — bounded by
`max_levels` / short-side floor.

### Annotation size gating (per-resolution minimum)
As a frame is downsampled, its objects shrink; below some size an annotation is an unlearnable,
sub-pixel-noisy target. A configurable **minimum object size**, evaluated on the box **after
downscaling to that level**, drops such annotations as training targets and from per-level
metrics at that level (they remain in coarser-relative/finer levels where they are large
enough, and always count in the final full-res recombined metrics).
- **Default:** drop any annotation whose **shorter side < 16 px** at the level (for a roughly
  square object that is ≈ 256 px²; the gate is on the shorter side, not area, unless area is
  explicitly configured).
- **Rationale:** YOLO's finest (highest-resolution) detection head is P3 at stride 8 px (P4-based
  models push the finest head to stride 16 px), so a 16 px shorter side ≈ 2 feature cells — the
  practical floor for a learnable target and comfortably above downscaling box-rounding noise.
- **Configurable:** the cutoff is a config value (`sati.config_schemas`), commonly raised to **32 px
  or higher** for cleaner targets (32 px aligns with COCO's "small" boundary). Expressed as
  min shorter-side **or** area.

---

## 4. Data Model

The project's source of truth is **Parquet tables + zarr image pyramids**. COCO, YOLO, and
precomputed tiles are import/export adapters only. Canonical full-frame image records,
full-frame annotations, slice geometry, derived slice annotations, and validation results are
separate tables so `level = 0` always means the full-resolution pyramid level, never a special
canonical row.

### Coordinate conventions

SATI uses one internal geometry convention everywhere:

- origin is the top-left pixel corner;
- `x` increases right, `y` increases down;
- image coordinate domains are continuous and half-open: `[0, width) × [0, height)`;
- boxes are half-open `xyxy`: `[x1, y1, x2, y2)`;
- box width is `x2 - x1`, height is `y2 - y1`, and area is `width * height`;
- transformed boxes are stored as floats; image sizes, slice origins, and slice dimensions are
  stored as integers.

COCO `xywh` and YOLO normalized `cxcywh` are adapter/export conventions. They may be stored as
convenience columns on canonical annotations and final outputs, but `xyxy` is the authoritative
internal box form for clipping, affine transforms, IoU, and recombination.

### 4.1 `fullframes.parquet`

Canonical image-level source records. This table is the source of truth for source image
membership, dimensions, and dataset split assignment, including frames with zero annotations.

| column | type | notes |
|---|---|---|
| `fullframe_id` | str | source full-frame id (canonicalized to string at ingest) |
| `source_uri` | str | path/URI of the source image |
| `dataset_split` | str | `training`, `validation`, or `holdout` |
| `fullres_wh` | int×2 | full-resolution image width and height |
| `color_mode` | str | `RGB` or grayscale; stored channel count is implied (RGB=3, grayscale=1) |

### 4.2 `annotations.parquet`

Canonical **full-frame, full-resolution** ground truth. This table has no slice semantics and
contains only annotation rows. Dataset split assignment lives in `fullframes.parquet`.

| column | type | notes |
|---|---|---|
| `fullframe_id` | str | source full-frame id |
| `ann_id` | int | annotation id |
| `class_id` | int | |
| `class_name` | str | |
| `bbox_xyxy` | float×4 | authoritative full-res `[x1, y1, x2, y2)` |
| `bbox_w` | float | `bbox_xyxy[2] - bbox_xyxy[0]` |
| `bbox_h` | float | `bbox_xyxy[3] - bbox_xyxy[1]` |
| `bbox_rel_cxcywh` | float×4 | YOLO-style convenience column normalized to full-frame |

### 4.3 `slices.parquet`

One row per generated slice/window, partitioned by `fullframe_id` / `level` / `slice_id`.
`level = 0` is reserved for the full-resolution pyramid level.

- `fullframe_id`, `level`, `slice_id`. **`slice_id` is unique across all slices of all levels
  for a given `fullframe_id`** (not reused per level), so a slice is identified by
  `(fullframe_id, slice_id)` alone; `level` is redundant-but-stored for partitioning/queries.
- `dataset_split` — copied from `fullframes.parquet`.
- `scale`, `level_wh` — which resolution level and its dimensions.
- `slice_xy`, `slice_wh` — actual slice origin and size in level coordinates. Typically
  `slice_wh` is the same as `nominal_slice_size` on each axis (smaller on a reduced axis).
- `slice_fullres_xyxy` — the half-open slice window mapped to full-res coordinates.
- `is_whole_level` — true when the entire resolution level is represented by one slice.
- **`affine_slice_to_level`** — `[tx, ty]`, the translate-only slice-local → **resolution-level**
  mapping: `x_level = x_slice + slice_x`, `y_level = y_slice + slice_y` (scale is 1 within a
  level). Basis for within-level merge (§6.2 Stage 1).
- **`affine_slice_to_fullres`** — `[sx, sy, tx, ty]`, an invertible per-axis scale+translate
  transform mapping slice-local coordinates to full-frame full-res coordinates:
  `x_fullres = sx * x_slice + tx`, `y_fullres = sy * y_slice + ty`, derived from the actual level
  dimensions: `sx = fullres_w / level_w`, `sy = fullres_h / level_h`, `tx = slice_x * sx`,
  `ty = slice_y * sy` (equivalently, slice→level then a scale-about-origin by `sx, sy`).

### 4.4 `slice_annotations.parquet`

Derived annotation projections, partitioned by `fullframe_id` / `level` / `slice_id`.
Rows link back to `annotations.parquet` via `parent_ann_id`.

- `slice_ann_id`, `parent_ann_id`, `fullframe_id`, `level`, `slice_id`.
- `bbox_fullres_xyxy` — canonical full-res annotation box, copied for traceability.
- `bbox_level_xyxy` — full object box after mapping to the resolution level.
- `bbox_slice_xyxy` — clipped object box in slice-local coordinates.
- `visible_area_frac` — clipped slice-visible area divided by the annotation area at that
  resolution level.
- `truncated_edges` — zero or more of `top`, `bottom`, `left`, `right`.
- `dropped_by_size_gate`, `dropped_by_min_area_ratio` — explicit bookkeeping for why a
  possible target was excluded from training.

Because every derived artifact links through `slices.parquet`, **any slice, resize, converted
annotation, or prediction is always backtraceable** to the source frame.

### 4.5 `validation_results.parquet` (results table)

Offline validation and inference artifacts live outside the direct model training loop, in
**one tall table discriminated by `record_type`**, with an **explicit, hand-written PyArrow
schema** (never Arrow type inference — it breaks on null-leading and list/vector columns).
Rows are partitioned by a configurable key set (default `record_type`, commonly plus run /
`fullframe_id` / `level`).

`record_type` mirrors the pipeline stages:
- `slice_prediction` — one row per raw detection in a slice; carries `level` and `slice_id`;
  box stored in **slice-local** coords (`bbox_slice_xyxy`). Persisted **only when the
  slice-prediction cache flag is enabled** (off by default); required for post-hoc
  recombination sweeps (§4.9).
- `level_merge` — one detection after within-level merge; has `level`, no specific `slice_id`;
  box stored in **resolution-level** coords (`bbox_level_xyxy`), not full-res.
- `final_merge` — one detection in full-frame full-res coords; no specific level/slice;
  scored against ground truth.
- `frame_metadata` — one row per source full-frame (dimensions, split, `source_uri`); static
  context predictions join by `fullframe_id`.

Columns not applicable to a row are typed nulls. Every row carries:
- **Provenance / identity keys** — `fullframe_id`, `dataset_split`, `run_id`, and the hash
  family (§4.9), so a single row is self-describing and rows from many runs pool into one frame
  and `group by` any hash without a join.
- **Recombination provenance** (merge rows) — `merge_strategy_name` (human-readable label) and
  `recombination_hash` (§4.9), the latter scoped to what produced the row: within-level params
  on `level_merge`, within-level + cross-level params on `final_merge` (§6.2).
- **Box columns per stage** — each row stores its box in its own coordinate space:
  `bbox_slice_xyxy` on `slice_prediction`, `bbox_level_xyxy` on `level_merge`, and
  `bbox_fullres_xyxy` on `final_merge`. `bbox_fullres_xyxy` is the authoritative form and is
  required for every kind that participates in metrics (only `final_merge` does). `final_merge`
  rows track contributing `source_levels` (and, where useful, the slice provenance that fed their
  within-level merge). Final outputs may also store convenience `w`, `h`, and `bbox_rel_cxcywh`
  columns for export.
- **`schema_version`** — bumped together with the sidecar (below).

Build the Arrow table **column-by-column against the declared schema** (a dict row that omits
a column yields a typed null; an unknown key is dropped) — a row's key set never defines the
columns.

A **`_metadata.json` sidecar** next to the Parquet holds constant-for-the-table semantic
metadata, validated by its own strict Pydantic model: the class-index→name map, column
descriptions, `schema_version`, and each compatibility hash **paired with the canonical
`*_source` object it was computed from** so the hash is independently verifiable. Do not
`sort_keys` when dumping the class map — numeric string keys (`"0"`, `"1"`, `"10"`) must keep
insertion order.

### 4.6 Zarr image pyramid

- One zarr group **per source image**; arrays named `0, 1, 2, …`, one per resolution level.
- Arrays are stored as `HWC`: `(height, width, channels)`.
- Phase 1 canonical imagery is `uint8`. Color images are stored as `RGB`; grayscale images are
  stored as one-channel arrays. Non-`uint8` sources are rejected or explicitly converted during
  ingest, with the conversion recorded in metadata.
- Level dimensions are derived with deterministic floor sizing from the requested scale. Metadata
  records both the requested scale and realized per-axis scale for each level.
- Downsampling method is configurable and recorded; the Phase-1 default is area/box-style
  resampling for downsampling natural imagery.
- Default chunk shape for `HWC` arrays is:
  `(min(level_h, 2 * nominal_slice_h), min(level_w, 2 * nominal_slice_w), channels)`.
  This is configurable and keeps on-the-fly slice reads local without creating oversized chunks
  at small levels.
- Group **attrs** record: `fullframe_id`, `source_uri`, source dimensions, aspect ratio,
  `dtype`, `channel_order`, requested scales, realized scales, level dimensions, chunk shape,
  downsample method, and pyramid config hash. Source content hashing is out of scope for
  Phase 1.
- Minimal, purpose-built layout. OME/NGFF interop is a possible later add (see §15).

### 4.7 Conversions
Canonical (parquet + zarr) ↔ **COCO**, **YOLO**, and **precomputed tiles**, round-tripping box
coordinates through the slice affine. These adapters are exposed as the `sati convert` CLI
command (§5, §13) — format translation only, distinct from the full `prepare` ingest.

### 4.8 Dataset splits

Phase 1 supports user-provided splits only. Splits are resolved at `prepare` time and stored in
`fullframes.parquet`, then copied into `slices.parquet` for direct training/evaluation queries.

Supported split sources:
- COCO inputs may provide separate files for train / validation / holdout.
- YOLO datasets may provide their existing train / validation layout.
- A TOML split file may disambiguate or override source-provided split membership, with
  `training = [...]`, `validation = [...]`, and optional `holdout = [...]` lists of
  `fullframe_id`s.

If a TOML split file is supplied, a `fullframe_id` listed in TOML but missing from the source
dataset is an error. A source dataset frame that is not listed in TOML is ignored with a warning
and omitted from `fullframes.parquet`, the zarr pyramid, slices, and annotations. SATI does not
generate random or grouped splits in Phase 1.

### 4.9 Compatibility hashes

Every result row and the sidecar (§4.5) carry a family of **content hashes** so any
prediction is traceable to the exact config, dataset, model, and preprocessing that produced
it **by value, not by remembering its directory**. All are **SHA-256 over canonical JSON** of
a deliberately chosen source object — sorted keys, no whitespace, floats quantized (`%.12g`),
non-finite floats tokenized, an `sha256:` prefix, and a versioned source recipe — so two
logically-equal inputs on two machines hash identically. What each hash *excludes* matters as
much as what it includes.

The hashes play two distinct roles; keep them straight.

**Compatibility gates** — a *hard precondition* for an operation across runs; a mismatch is a
mechanical error you refuse to proceed on, no model load needed.

| hash | digests | gates |
|---|---|---|
| `class_map_hash` | ordered class-index→name mapping | comparing **or** ensembling — meaningless across different label spaces |
| `dataset_hash` | manifest file identities (relative path + size) + backend + column bindings; **no pixels, no mtime** | comparing metrics — only valid on the same evaluation set by structure |
| `preprocessing_hash` | nominal slice, resolution stack, overlap, backend-input/letterbox transform | ensembling raw predictions — requires the same inputs were fed to the model |

**Provenance / group-by axes** — *identifiers* that keep result sets distinct so rows pool and
`group by` cleanly. Not gates; these are the experimental variables you compare across.

| hash | digests | distinguishes |
|---|---|---|
| `config_hash` | whole resolved `RootConfig` minus output paths / `run_id` / derived-only fields | run recipe |
| `model_config_hash` | resolved `model` sub-config (backend, architecture, weights source, `num_classes`) | model spec |
| `checkpoint_hash` | **trained checkpoint file bytes** (the ultralytics `.pt` produced by training, hashed after training completes and before eval) | trained parameters |
| `recombination_hash` | strategy name + version + the merge params that produced the row (scoped per record_type — see below) | predictions-assembly |

`recombination_hash` replaces the older `merge_strategy_config_hash`; the strategy's
human-readable label is carried separately as `merge_strategy_name`. Its source is **scoped to
the record it stamps** (option A): on `level_merge` rows it digests the **within-level** merge
config only, so a byte-identical level_merge row keeps a stable hash when only cross-level
params change; on `final_merge` rows it digests **within-level + cross-level** config, since
both stages produced the row (§6.2). Because recombination config is already part of
`RootConfig`, `recombination_hash` overlaps `config_hash` within a single run; it earns its own
column for (1) **post-hoc recombination sweeps** over cached `slice_prediction` rows under a
fixed config, where `config_hash` is constant, and (2) isolating recombination as a **group-by
axis** across runs whose other settings differ. Note the sweep path in (1) requires
`slice_prediction` rows to have been persisted: they are written to `validation_results.parquet`
**only when the slice-prediction cache flag is enabled** (off by default); without it, only
`level_merge` / `final_merge` rows are retained and a re-run of Stage 1 is needed to sweep.

Config-derived hashes are pure functions of the resolved `RootConfig`, computed once per run;
`dataset_hash` is computed when the dataset is built; `checkpoint_hash` when training writes the
trained checkpoint (so eval rows, produced before `export`, already carry it). Export produces a
separate deployment artifact (ONNX/TensorRT, §10) and does not change `checkpoint_hash`.

---

## 5. Pipeline

```
prepare  ->  train  ->  eval  ->  export
prepare -> infer -> (results)
```

- **prepare** — ingest a COCO- or YOLO-based dataset, resolve training / validation / holdout
  membership, then write canonical Parquet + zarr pyramid + manifest tables.
- **train** — stock ultralytics over slices generated from the pyramid.
- **infer** — SAHI-like multi-resolution sliced inference (§9).
- **eval** — metrics on recombined predictions vs full-frame full-res ground truths (§11).
- **export** — checkpoint/pt-file + inference config (§10).
- **results** — not a command but the **output of `infer`**: the Parquet final output
  (`validation_results.parquet`), optionally exported as COCO or YOLO predictions.

**`convert`** is a standalone **utility command** (not a stage — hence absent from the diagram):
the CLI surface for the §4.7 conversions, translating external datasets ↔ SATI's canonical store
in both directions. Unlike `prepare`, it does **format translation only** — no zarr pyramid, no
manifest build. Use `prepare` to start a run; `convert` for one-off interchange with external
tooling.

Slices are generated **on-the-fly** from the zarr pyramid by default (seeded for reproducible
validation), with a config option to **materialize** slices to disk.

---

## 6. Tiling — slicing & recombination (`sati.tiling`)

### 6.0 Slicing: build vs. SAHI dependency
We **roll our own slicing** and treat [SAHI](https://github.com/obss/sahi/blob/main/docs/slicing.md)
as a design reference, not a runtime dependency. SAHI's `slice_image` / `slice_coco` operate on
a **single in-memory/file image at one resolution** and return COCO-shifted annotations; the
genuinely reusable logic is just tile-window-offset math, box shift/clip, and `min_area_ratio`
filtering (a few dozen well-tested lines). SATI's slicer instead must read **lazily from the
zarr pyramid across multiple resolution levels**, emit the **slice affine** per slice, do
**per-axis slice-dimension reduction** (§3), and produce our dual box convention (§4) — none of which SAHI's API can
express, so we write the slicer regardless and would only borrow trivial geometry while taking on
coupling to SAHI's data model (`SliceImageResult` / `ObjectPrediction`) and release cadence.

What we **borrow without depending on the package**:
- **Semantics, not code:** keep SAHI's `overlap_*_ratio` and `min_area_ratio` meanings identical
  so users migrating from SAHI get predictable behavior (see §3).
- **`shapely`** (a direct core dependency — what SAHI itself uses) for box/polygon clipping and IoU.
- For recombination dedup (§6.2), implement **NMM / GREEDYNMM** ourselves (or a small standalone
  such as torchvision NMS / `lsnms`) rather than `sahi.postprocess`, since our merge is
  cross-scale and provenance-aware.

> If bit-for-bit parity with SAHI inference outputs ever becomes a product requirement, depending
> on SAHI would buy that — but SATI is a *generalization* of SAHI (multi-resolution + training), so
> parity is not a goal and not worth the coupling.

### 6.1 Slicing
For each configured resolution level, tile the (downsampled) full-frame **per-axis** (§3): an
axis `≥ nominal_slice_size` is cut into overlapping tiles of nominal slice length; an axis
`< nominal_slice_size` becomes a single native, stride-aligned span (slice-dimension reduction
— no padding/stretch). The pyramid terminates at the level where the whole frame is fed as one
input. For each slice, write its per-axis affine to `slices.parquet`, the dropped-annotation set
from the per-resolution **size gate** (§3), and per-object `visible_area_frac` /
`truncated_edges` flags to `slice_annotations.parquet`.

Training target truncation uses SAHI-compatible `min_area_ratio` semantics:
`visible_area_frac = clipped_box_area / level_box_area`. A slice annotation is kept as a
training target iff `visible_area_frac >= min_area_ratio`. The default is `0.20`; `0.0` keeps
any overlap and `1.0` keeps only fully contained objects. This policy applies to training
targets only. Phase-1 per-slice validation has no special truncation treatment; final metrics
are computed from recombined full-frame predictions.

Rectangular / whole-frame slices are native to SATI. Backend adapters define how off-`imgsz`
slices are transformed into model tensors and must record the preprocessing transform
(`actual_slice_wh`, `backend_input_wh`, resize scale, pad offset, letterbox color, stride
alignment) so model outputs are inverted back to slice space before SATI recombination.

### 6.2 Recombination (pluggable `Strategy`)
Recombination has two stages. Both are pluggable because the best cross-scale heuristic is an
open research question, but Phase 1 keeps the default simple and debuggable.

#### Stage 1: within-level merge

Within each `fullframe_id` and resolution `level`, raw per-slice predictions are mapped through
their **slice→level** transform (`affine_slice_to_level`, §4.3) into shared level coordinates and
merged into `level_merge` predictions (`bbox_level_xyxy`). This
stage removes duplicates caused by overlapping slices at the same scale and exposes SAHI-like
postprocess options: `NMS`, `NMM`, `GREEDYNMM`, optional `LSNMS`, IoU threshold, confidence
threshold, and class-aware vs class-agnostic matching. Only Stage 2 lifts boxes to full-res.

Every `level_merge` prediction records provenance, including contributing `slice_id`s and the
**`near_edge_count`** of the single contributing raw detection that won the merge: the integer
number (0–4) of that detection's own slice edges — `top`/`bottom`/`left`/`right` — that lie
within the configured `edge_proximity` of the box in that slice. It is a per-slice count of one
detection, not a sum across contributing slices. Each row is stamped with `merge_strategy_name`
and a `recombination_hash` digesting the **within-level** merge config only (§4.9).

#### Stage 2: cross-level merge

Across all levels for one `fullframe_id`, `level_merge` predictions are lifted from level to
full-res by the level's scale `(sx, sy)` (§4.3), then clustered by IoU (class-aware by default)
and reduced to `final_merge` predictions in full-frame full-res coordinates. The Phase-1 default
selects one candidate per cross-level cluster using:

```text
favorability =
  confidence
  * scale_multiplier
  * edge_multiplier

scale_multiplier =
  (sqrt(box_area_in_slice_pixels) / sqrt(slice_area_pixels))

edge_multiplier =
  (1 - edge_favorability_penalty) ** near_edge_count
```

`edge_favorability_penalty = 0.0` disables the edge penalty. `near_edge_count` is the number of
slice edges that meet the configured edge-proximity rule. `edge_proximity = 0.0` means only
touching the edge counts; `0.0 < edge_proximity < 1.0` is interpreted as a fraction of the
relevant slice dimension; `edge_proximity >= 1.0` is interpreted as pixels. Multiple near edges
compound multiplicatively. Edge proximity is calculated during the within-level merge so the
cross-level strategy can use stable level-merge metadata.

Final candidates record which level they came from, and each `final_merge` row is stamped with
`merge_strategy_name` and a `recombination_hash` digesting the **within-level + cross-level**
config, since both stages produced it (§4.9). Weighted box fusion / cross-level NMM is a
later-phase strategy option, not the Phase-1 default.

Validation/inference metrics are always computed on `final_merge` predictions against
full-frame full-res ground truth.

---

## 7. Backend abstraction (`sati.backends`)

A `Backend` protocol exposes `train / val / predict / export`, decoupling SATI's novel parts
(pyramid, manifest tables, tiling, recombination, evaluation, inference command) from the detector.

- **Ultralytics** — the Phase-1 implementation, used **stock** (no loss/trainer surgery).
- **mmdetection** — a documented **stub only**. mmdet's `DetDataSample` / registry design would
  make per-object metadata threading and custom losses cleaner *if adopted*, but mmdet is
  effectively unmaintained as of 2025–2026 (development stalled, staff reassigned, no clear
  PyTorch 2.x roadmap), so it is not a planned migration. **Recombined-full-frame validation is
  equally custom in any framework** — it is intrinsic to SATI, not a gap in any one backend.

> **Expected, not a bug:** when a resolution-aware loss is computed per-slice (a later phase)
> while validation metrics are computed on recombined full-frames, the two numbers will not
> track each other one-to-one. This is documented behavior.

---

## 8. Training

### Phase 1 — stock ultralytics
Train ultralytics over slices produced from the pyramid, with no custom loss or trainer.
Recombined-full-frame metrics are produced by an **offline evaluator** (§11), not wired into
the training loss. This yields a complete, exportable, deployable system quickly.

**Scale coverage / sampling.** Coarse-level (whole-frame / rectangular) slices are rare but hold
the **largest objects**, so a naive uniform sampler under-represents large-object scales. The
training sampler supports **per-level sampling weights** (oversample coarse-level slices) so each
scale is adequately represented; watch the per-scale eval breakdown (§11) to detect a starved
level. This is a data-distribution knob, independent of slice shape, and a natural companion to
the resolution-aware loss below.

**Negative (background) slices.** Zero-target slices — including those from zero-annotation
frames — are emitted and used as negatives, sampled per `negative_sample_ratio` (§3). Orthogonal
to the per-level weights above: one balances positive/negative, the other balances scale.

### Later phases — resolution-aware loss
Implement a custom loss via overriding `init_criterion` on a custom model plus subclassing
`DetectionTrainer`, threading per-object metadata (source scale, truncation flags) through a
custom dataset / collate. The reweighting is a **configurable research knob**:

- reweight by **resolution level**,
- down-weight **slice-edge-truncated** objects,
- **both**, or
- **neither**.

The config surface for this knob is designed now (in `sati.config_schemas`) even though the
implementation lands later.

---

## 9. Inference (SAHI-like, `sati.infer`)

The inference command takes:
- a model (exported checkpoint), and
- a **nominal-slice + resolution-sampling YAML config** (the one shipped with the model, or an override).

It internally builds the resolution stack and slices, runs the backend per slice, recombines
(§6.2), and outputs final boxes + class labels in **YOLO and/or COCO**. Intermediate image
files and annotations may be **optionally cached** for inspection/traceability.

---

## 10. Export

A finalized checkpoint is exported **together with an inference configuration file** that
specifies the nominal slice size, downsample steps, backend preprocessing, and recombination
settings to use at inference. Ultralytics' native ONNX /
TensorRT export is leveraged for the model artifact; SATI adds the inference-config sidecar so
the deployment exactly reproduces the intended multi-resolution sampling.

---

## 11. Evaluation (`sati.evaluation`)

Metrics (mAP / PR, pycocotools-compatible) are computed on **recombined predictions vs
full-frame, full-resolution ground truth**, with **per-scale breakdowns** so we can see how
each resolution level contributes. Every prediction remains provenance-backed, so any score can
be traced to the exact slices/levels that produced it. In Phase 1 this runs as a standalone
evaluator over training/validation outputs and writes `validation_results.parquet`.

---

## 12. Module Layout

| module | responsibility |
|---|---|
| `sati.config_schemas` | Pydantic — source of truth for options & I/O typing |
| `sati.config_loader` | compose→validate→resolve via Hydra Compose API; exposes `compose_and_resolve` |
| `sati/config_defaults/` | packaged Hydra YAML fragment tree (composable defaults + experiments) |
| `sati.io` | Data sources & sinks: zarr, COCO, YOLO over local fs / S3 / web |
| `sati.manifests` | Parquet-backed fullframe, annotation, slice, slice-annotation, and validation-result tables |
| `sati.datasets` | Framework-facing dataset producing slices (on-the-fly or materialized) |
| `sati.tiling` | Slicing and recombination |
| `sati.backends` | `Backend` protocol + ultralytics impl + mmdet stub |
| `sati.train` | Training orchestration |
| `sati.infer` | SAHI-like multi-resolution inference command |
| `sati.evaluation` | Recombined-full-frame metrics |
| `sati.cli` | Command-line entrypoints |

---

## 13. CLI & Packaging

The CLI is **Typer**, thin by design: each command only parses arguments, calls
`compose_and_resolve` (§3), and hands the typed `RootConfig` to a CLI-free work function
(`execute_train`, `execute_infer`, …) so tests and orchestrators reuse the same seam. No
training/inference logic lives in the CLI.

**The dash rule** splits arguments cleanly:
- `--dashed` arguments are Typer options (`--config`, `--config-dir`, `--resolved-config`);
- dash-free `a.b.c=value` arguments are Hydra overrides forwarded straight to the compositor
  (`training.epochs=10`, `slicing.overlap_ratio=0.25`).

`--resolved-config` loads a previously resolved artifact, skips composition, applies only the
passed overrides, and runs — the "re-run this precise thing" path (§3.1).

`pyproject.toml` defines core dependencies, optional extras, and a console script.

- **Core deps:** `pydantic`, `hydra-core`, `omegaconf`, `typer`, `zarr`, `pyarrow` (Parquet),
  `numpy`, **`shapely`** (box/polygon clipping & IoU for slicing/recombination — §6.0), plus
  an image codec (e.g. `opencv-python` / `pillow`). `rich` for CLI output.
- **Extras:** `[ultralytics]` (Phase-1 backend), `[dev]` (lint/test/docs).
- **Console script:** `sati` →
  `prepare` · `convert` · `train` · `eval` · `infer` · `export`.

---

## 14. Pitfalls & Mitigations

- **Ultralytics loss/val are invasive to customize.** `v8DetectionLoss` is monolithic with no
  plugin hook; the validator scores per-fed-image. → Phase 1 uses stock training + an offline
  recombined evaluator; custom loss is deferred and isolated behind a custom trainer.
- **Train-vs-val metric mismatch.** Per-slice loss vs recombined-frame metrics won't track
  one-to-one. → Documented as expected (§7).
- **Cross-scale recombination is the core research problem.** Objects whole at one scale and
  cut at another, double counting, class disagreement. → Two-stage recombination with
  SAHI-like within-level merge, configurable cross-level strategy, and per-record-type
  `recombination_hash` provenance in `validation_results.parquet` (§4.9, §6.2).
- **Edge / cut-off training targets.** → Training uses SAHI-compatible `min_area_ratio`
  (default `0.20`) so clipped targets are kept only when enough object area is visible.
- **Near-edge predictions.** → Cross-level merge has configurable `edge_proximity` and
  `edge_favorability_penalty`; set the penalty to `0.0` to ignore near-edge weighting. Recommend
  **overlap ratio ≥ (largest expected object / nominal_slice_size)** so big objects survive
  whole in at least one slice.
- **Coarse-level / extreme-aspect bookkeeping.** Per-axis slice-dimension reduction (§3) avoids
  pad/stretch entirely, but the per-axis scale+translate must stay invertible or recombined boxes
  misplace. → Every transform is stored as an invertible affine in `slices.parquet` (§4.3);
  extreme aspect ratios bounded by `max_levels` / short-side floor.
- **Scale coverage starvation.** Rare coarse-level slices under-represent large objects. →
  Per-level sampling weights + per-scale eval breakdown (§8, §11).
- **Over-aggressive annotation size gate.** Too high a per-resolution minimum silently drops
  real targets. → Conservative default (≈16 px short side); monitor dropped-annotation counts in
  `slice_annotations.parquet` (§3).
- **Storage blow-up vs determinism.** A full pyramid + materialized slices for huge imagery is
  large; on-the-fly slicing is lean but must be reproducible. → On-the-fly default, **seeded**;
  materialization is opt-in.
- **mmdet maintenance risk.** → Stub only; never on the Phase-1 critical path.

---

## 15. Roadmap / Phasing

- **Phase 1 (foundation):** `sati.config_schemas` / `sati.config_loader` / `sati/config_defaults/`, `sati.io`, zarr pyramid +
  `sati.manifests` tables, `sati.tiling` (slice + recombine), `sati.datasets`,
  **stock-ultralytics** training, offline recombined-full-frame evaluation with
  `validation_results.parquet`, the SAHI-like inference command + exported inference config,
  `sati.cli`, `pyproject.toml`.
- **Phase 2:** resolution-aware loss + truncation weighting (custom ultralytics model/trainer),
  threading per-object metadata.
- **Phase 3:** Hydra **sweeps** for hyperparameters, including resolution steps.
- **Later / optional:** an mmdetection backend; NGFF interop for the image pyramid.

---

## Appendix — References

- SAHI: sliced inference, overlap ratios, `GREEDYNMM` / `NMM` / `NMS` / `LSNMS` merge,
  optional standard full-image pass.
- Ultralytics: custom loss via `init_criterion` override + `DetectionTrainer` subclassing;
  native ONNX / TensorRT export.
- yolo-tiling: tile slicing with YOLO label re-derivation and boundary clipping (inspiration
  for the edge/truncation policy).
