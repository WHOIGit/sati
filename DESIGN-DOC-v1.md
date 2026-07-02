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
- NMM / GREEDYNMM / favorability scoring / edge penalties in recombination.
- Parameter sweeps. Segmentation/keypoints. mmdetection.

---

## 2. Core Concepts & Glossary

- **Full-frame / full-res:** the original source image at native resolution; the unit input
  annotations describe and that final predictions/metrics are reported against.
- **Resolution level:** one entry in the downsample stack. Level 0 = full-res; level *k* has
  dimensions `floor(fullres / factor^k)` per axis (min 1 px).
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
  sentinels); *resolved* is the fully rendered artifact that actually runs, written to the run
  directory as `resolved.yaml`.
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
  entry per runnable job.
- **Authored vs. resolved.** The resolver renders templates, absolutizes paths, computes
  derived values, and writes `resolved.yaml` into the run directory so a run is exactly
  re-runnable without re-composition.
- **One entry point.** CLI, tests, and orchestrators all call the same
  `compose_and_resolve(...)`.
- **Deterministic identity.** The resolved config hashes to a stable `config_hash` (§4.7).

### 3.2 Modules

- `sati.config_schemas` — Pydantic models: `RootConfig` + sub-models, `Literal[...]` enums,
  `Field(ge=/le=)` bounds, `model_validator`s for cross-field invariants. Zero Hydra imports.
- `sati/config_defaults/` — packaged Hydra YAML tree, loaded via `importlib.resources`; a
  local `./config_defaults` (or `--config-dir`) shadows it.
- `sati.config_loader` — compose → validate → resolve; exposes one public
  `compose_and_resolve`. Drives Hydra through the **Compose API**
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
| `data.source_format` | `coco` \| `yolo` | input dataset adapter |
| `data.source_path` | path | dataset root / annotation file(s) |
| `data.split_file` | path, optional | TOML split override (§4.6) |
| `slicing.nominal_slice_size` | int, `1024` | square sampling window |
| `slicing.overlap_ratio` | float `[0,1)`, `0.10` | applied on both axes |
| `slicing.downsample_factor` | number > 1, `2` | automatic pyramid factor |
| `slicing.max_levels` | int, optional | hard cap on stack depth |
| `slicing.min_area_ratio` | float `[0,1]`, `0.20` | training-target visibility gate |
| `slicing.size_gate_px` | int, `16` | min shorter side at level, post-downscale (§6.1) |
| `slicing.image_format` | `png` \| `jpeg`, `png` | materialized slice codec |
| `model.*` | ultralytics arch/weights/`num_classes`/`imgsz` | passed through to backend |
| `training.*` | epochs, batch, lr, seed, device | passed through to backend |
| `recombination.within_level_iou` | float, `0.5` | Stage-1 NMS IoU threshold |
| `recombination.cross_level_iou` | float, `0.5` | Stage-2 NMS IoU threshold |
| `recombination.confidence_threshold` | float, `0.05` | applied before Stage 1 |
| `recombination.class_agnostic` | bool, `false` | matching mode for both stages |
| `results.cache_slice_predictions` | bool, `false` | persist `slice_prediction` rows (§4.5) |
| `output.run_dir` | path template | run directory root |

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

All hashes are **SHA-256 over canonical JSON** of a deliberately chosen source object:
sorted keys, no whitespace, floats `%.12g`, non-finite floats tokenized, `sha256:` prefix,
versioned source recipe. What each hash *excludes* matters as much as what it includes.

**Gates** — hard preconditions; a mismatch refuses the operation, no model load needed:

| hash | digests | gates |
|---|---|---|
| `class_map_hash` | ordered class-index→name map | comparing **or** ensembling |
| `dataset_hash` | manifest file identities (relative path + size), no pixels/mtime | metric comparison |
| `preprocessing_hash` | nominal slice, stack, overlap, stretch policy | ensembling raw predictions |

**Provenance / group-by axes** — identifiers, not gates:

| hash | digests |
|---|---|
| `config_hash` | resolved `RootConfig` minus output paths / `run_id` / derived-only fields |
| `model_config_hash` | resolved `model` sub-config |
| `checkpoint_hash` | trained `.pt` file bytes, hashed after training / before eval |
| `recombination_hash` | strategy name + version + merge params, scoped per record_type: within-level config on `level_merge`; within-level **+** cross-level on `final_merge` |

**Decided (2026-07-02):** path exclusion is type-driven, never a hand-maintained list.
Every path-typed schema field is declared as `SatiPath` — an `Annotated[Path, ...]` alias
defined in `sati.config_schemas` — and the hasher walks the Pydantic model tree, replacing
any `SatiPath`-annotated value with the sentinel `"<path>"` before canonical-JSON
serialization. A plain `Path`/`str` holding a path in a schema is a bug by definition.
Guard tests: (1) hash one authored config resolved under two different fake roots and
assert equality; (2) walk `RootConfig` and assert no field annotates bare `pathlib.Path`.

### 4.8 Materialized slice store (on-disk layout)

`prepare` writes, under the dataset root:

```
<dataset_root>/
  manifests/            fullframes.parquet, annotations.parquet,
                        slices.parquet, slice_annotations.parquet, _metadata.json
  yolo/
    data.yaml           generated; consumed verbatim by ultralytics
    images/{training,validation}/   <fullframe_id>_L<level>_S<slice_id>.png
    labels/{training,validation}/   matching .txt, YOLO cxcywh in emitted-slice pixels
```

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
and accepts `--json` for a machine-readable dump.

| subcommand | reads | reports |
|---|---|---|
| `inspect config <path\|run_dir>` | authored YAML or `resolved.yaml` | validation status, authored→resolved diff, derived values, and the **hash family** the config would produce (`config`/`model`/`dataset`/`preprocessing`); `--diff <other>` compares two resolved configs field-by-field |
| `inspect dataset <root>` | `fullframes` + `annotations` | frame & annotation counts, per-split counts, class map, per-class instance counts, image-dimension and box-size distributions, count of zero-annotation frames |
| `inspect slices <root>` | `slices` + `slice_annotations` | per-level slice & object counts, `dropped_by_size_gate` / `dropped_by_min_area_ratio` tallies, stretched-axis frequency, on-disk size of the materialized YOLO tree, and a spot-check that every `slice_uri` resolves |
| `inspect results <parquet>` | table + `_metadata.json` | `record_type` row breakdown, `run_id`s and hash families present, sidecar validation (schema_version, **each hash re-verified against its paired `*_source`**), and any scored metrics |
| `inspect model <ckpt>` | `.pt` bytes (+ sibling sidecar) | `checkpoint_hash`, ultralytics metadata (arch, `num_classes`, `imgsz`, epoch), and — if an exported `inference_config.yaml` sits alongside — its slice/stack/recombination summary |

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

**Level stack (automatic factor mode — the only v1 mode).** Level 0 is full-res; each
subsequent level divides both dimensions by `downsample_factor` (floor, min 1 px) until
`max(level_w, level_h) ≤ nominal_slice_size` — terminate on the *longer* side — or
`max_levels` is hit. Both requested and realized (post-floor) scales are recorded.
Downsampling uses OpenCV `INTER_AREA`.

**Per-axis tiling with stretch.** For each level, per axis:

- axis `≥ nominal_slice_size` → overlapping tiles of nominal length. Stride is
  `nominal_slice_size · (1 − overlap_ratio)`. **Last-tile rule:** the final tile is clamped
  flush to the far edge (`origin = max(0, axis_len − nominal)`), so it keeps full nominal
  length and simply overlaps its neighbor more. No short tiles, no padding. Deterministic
  given `(axis_len, nominal, overlap_ratio)`.
- axis `< nominal_slice_size` → one span covering the axis, **stretched** to nominal length
  at emission (`cv2.resize`, per-axis). The terminal whole-frame level is the degenerate
  case: the entire level stretched to nominal×nominal.

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

### 6.2 Recombination — two stages, NMS only

Both stages are behind a `Strategy` interface (the cross-scale heuristic is the project's
research axis), but v1 ships exactly one strategy per stage and both are plain NMS.

**Stage 1 — within-level merge.** Per `(fullframe_id, level)`: apply
`confidence_threshold`; map raw boxes through `affine_slice_to_level` into shared level
coordinates; class-aware NMS (or agnostic per config) at `within_level_iou`. Winners become
`level_merge` rows (`bbox_level_xyxy`) recording contributing `slice_id`s,
`merge_strategy_name`, and a `recombination_hash` over within-level params only.

**Stage 2 — cross-level merge.** Per `fullframe_id`: lift each `level_merge` box to
full-res by the level's realized scale; class-aware NMS at `cross_level_iou`, **ranked by
raw confidence** (no favorability weighting in v1). Winners become `final_merge` rows
(`bbox_fullres_xyxy`) recording `source_levels` and a `recombination_hash` over
within-level + cross-level params.

Known, accepted v1 bias: plain confidence ranking lets a near-slice-edge fragment outrank a
whole detection from a coarser level, and duplicates split across levels dedup purely by
IoU. Mitigation until v2's favorability/edge machinery: configure
`overlap_ratio ≥ largest_expected_object / nominal_slice_size` so big objects survive whole
in at least one slice of some level.

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

> [!NOTE]
> Pin the ultralytics version range in `pyproject.toml` and add one integration test
> asserting predictions on a synthetic slice land where expected — the returned-coordinate
> contract has shifted across ultralytics releases, and this test is the tripwire.

---

## 8. Training

`sati train` = `compose_and_resolve` → hash bookkeeping → `YOLO(model).train(data=<generated
data.yaml>, imgsz=nominal_slice_size, epochs=…, seed=…)` → hash the produced checkpoint.
That's the whole command. Slices are square and pre-materialized, labels are pre-projected,
negatives are empty-label files: everything ultralytics natively understands.

Deliberately absent in v1: per-level sampling weights and `negative_sample_ratio`. Coarse
levels contribute few slices, so large-object scales may be under-represented — watch the
per-level counts `prepare` logs and the per-level eval breakdown (§11) to see whether v2's
sampling knobs are needed for your data.

Training-time augmentation is whatever stock ultralytics does at defaults.

> [!NOTE]
> Ultralytics' mosaic/scale augmentations operate on slices that SATI has already scale-
> stratified — mosaic mixing slices from different levels is probably harmless (it's just
> more scale augmentation) but is unstudied. Decide at implementation whether v1 passes
> augmentation controls through `training.*` untouched (recommended) or pins mosaic off;
> record the choice in the config defaults.

---

## 9. Inference (`sati.infer`)

Input: a model checkpoint + an inference config (the exported sidecar, or any config
providing `slicing.*` and `recombination.*`).

Per frame: build the level stack in memory → emit nominal×nominal slices (same code path as
`prepare`, §6.1, minus disk writes) → `predict` per slice → recombine (§6.2) → append
`final_merge` (+ `frame_metadata`, + optional cached `slice_prediction`) rows to
`validation_results.parquet` → optionally export final boxes as COCO JSON and/or a YOLO
prediction tree. An optional debug flag materializes slices + per-slice overlays for
inspection.

`eval` (§11) is `infer` over a prepared split plus scoring — one engine, two commands.

---

## 10. Export

`sati export` wraps ultralytics' native ONNX/TensorRT export and writes an
**`inference_config.yaml` sidecar**: nominal slice size, downsample factor / realized stack
recipe, stretch policy, recombination params, class map, and the hash family
(`checkpoint_hash` is of the source `.pt`; export artifacts don't change it). Deployment
reproduces the exact multi-resolution sampling from the sidecar alone.

v1's all-square-slices rule has a quiet payoff here: exported engines get one fixed input
shape (`nominal × nominal`), so ONNX/TensorRT static shapes just work. (v2's rectangular
slices reopen this — see v2 doc.)

---

## 11. Evaluation (`sati.evaluation`)

Metrics are computed on **`final_merge` predictions vs full-frame full-res ground truth** —
never per-slice. pycocotools-compatible mAP/PR, overall and per-class, written alongside
`validation_results.parquet`; every score traces to exact slices/levels via provenance
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

**The dash rule:** `--dashed` args are Typer options (`--config`, `--config-dir`,
`--resolved-config`); dash-free `a.b=c` args are Hydra overrides forwarded to the
compositor (`training.epochs=10`, `slicing.overlap_ratio=0.25`). `--resolved-config` loads
a previously resolved artifact, skips composition, applies only the passed overrides — the
"re-run this precise thing" path.

**Console script:** `sati` → `prepare` · `convert` · `train` · `eval` · `infer` · `export`
· `inspect` (a sub-app: `inspect config|dataset|slices|results|model`, §5.1).

**Core deps:** `pydantic`, `hydra-core`, `omegaconf`, `typer`, `pyarrow`, `numpy`,
`shapely`, `opencv-python`, `rich`. **Extras:** `[ultralytics]`, `[dev]`. (No `zarr` in v1.)

---

## 14. Deferred to v2 (see `DESIGN-DOC-v2.md`)

Zarr pyramid store · on-the-fly slicing datasets · custom ultralytics dataset/trainer ·
native rectangular & whole-frame slices (stretch removed) + backend preprocessing capture ·
explicit scale lists + short-side floor · NMM/GREEDYNMM/LSNMS, favorability scoring, edge
proximity/penalty · post-hoc recombination sweeps · per-level sampling weights ·
`negative_sample_ratio` · parameter-sweep config generation · per-scale mAP semantics ·
resolution-aware loss and beyond (v2 §11).

## 15. v1 Pitfalls & Mitigations

- **Stretch distortion at coarse levels.** Bounded by `downsample_factor`; recorded in
  affines; objects there are large in slice pixels, which tolerates distortion best. Still a
  known accuracy tax v2 removes.
- **Storage blow-up.** Materializing every slice of every level roughly doubles-plus the
  dataset on disk (overlap × levels). Accepted for v1 simplicity; PNG default keeps it
  lossless. v2's zarr + on-the-fly removes it.
- **Confidence-only cross-level NMS.** Edge fragments can beat whole coarse detections;
  mitigate via the overlap-ratio rule of thumb (§6.2) until v2 favorability.
- **Scale starvation.** Few coarse slices per frame; no sampling knobs yet — monitor §11
  diagnostics.
- **Stale YOLO tree.** The materialized tree is derived from manifests; re-running `prepare`
  with changed slicing config must fully rebuild it. `data.yaml` embeds
  `preprocessing_hash`; `train` refuses on mismatch with the composed config.
- **Ultralytics API drift.** Version-pin + the §7 tripwire test.
