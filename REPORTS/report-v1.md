# SATI v1 Implementation Report

## Purpose

Continuous record for implementing DESIGN-DOC-v1 (and, later, the
synth-dataset-builder Stage 1) end to end. Entries earlier than this report's
creation (23:15) are reconstructed from the session log of 2026-07-02.

## 2026-07-02 ~18:30 EDT — Bootstrap: pyproject + day-one decisions

- Created shared root `pyproject.toml` (hatchling, src layout): core deps per
  v1 §13, extras `[ultralytics]` (pinned `>=8.3,<9`), `[synth]`, `[dev]`;
  console scripts `sati` and `sati-synth`.
- Editable-installed into the mandatory repo `.venv` (Python 3.14.0); all deps
  resolved with 3.14 wheels (hydra-core 1.3.3, omegaconf 2.3.1, opencv 5.0.0,
  shapely 2.1.2, pyarrow 24).
- Settled v1 day-one decisions into spec prose: (a) packaged Hydra defaults via
  `importlib.resources` → real dir, hard error on zip imports; (b) type-driven
  `config_hash` path stripping via the `SatiPath` annotated alias.
- DOX pass: built root AGENTS.md Child DOX Index; child docs for `src/sati/`
  and `src/synth_dataset_builder/`; started `QUESTIONS-FOR-SIDNEY.md` (items 1–5).

## 2026-07-02 ~21:45 EDT — config_schemas + config_loader (dojo-structured)

- User direction: mirror dojo's structure where concepts align.
- `sati/config_schemas/` package: full v1 §3.3 surface as strict Pydantic
  models; `SatiPath`/`HashDerived` markers + annotation walkers.
- `sati/config_loader/` package split dojo-style: `compositor.py` (all Hydra,
  Compose API, `hydra.searchpath` fallthrough with owning-dir-first),
  `resolver.py` (run_id, path absolutization, `imgsz` derivation,
  `resolved.yaml`), `conductor.py` (`compose_and_resolve`).
- `sati/hashing.py`: canonical JSON per §4.7 + `config_hash`/`model_config_hash`.
- Packaged `config_defaults/` tree (data/model/slicing/recombination/training
  groups; `data.source_path` and `model.num_classes` mandatory).
- Finding: Hydra loads the primary config and group defaults main-dir-first, so
  a local tree without `config.yaml` shadows nothing (QUESTIONS item 7).
- Verification: `pytest tests/` — 18 passed; ruff clean.

## 2026-07-02 ~22:40 EDT — Hybrid experiment root selection

- Investigated dojo's answer to the primary-config constraint: dojo has **no
  root config.yaml**; `experiment=name` is intercepted and used as the primary
  config name. Sidney chose the hybrid for SATI.
- Ported `_split_experiment_selector`: `experiment=name` promotes
  `experiment/name` to primary root (experiments are full roots with their own
  defaults lists); selector-less runs keep the packaged `config.yaml`.
- Rewrote packaged `experiment/example.yaml` as a full root.
- Verification: 21 passed (packaged experiment primary, local experiment
  falling through to packaged groups, file-path selector error); ruff clean.

## 2026-07-02 ~23:00 EDT — Weights hash rule + settlements

- Sidney's rule: file-backed `model.weights` contribute their byte hash
  (`checkpoint_hash`) to `config_hash`/`model_config_hash`; hub asset names
  hash verbatim. Implemented `checkpoint_hash()` (streaming SHA-256) +
  substitution in both hash sources.
- Settled: author/license metadata in pyproject (Sidney Batchelder, MIT),
  `requires-python >=3.12`, `config_defaults/` name kept, ultralytics 8.4.86 +
  torch 2.12.1 installed by Sidney (opencv-python 5.0 retained).
- Added "Communications from the SATI Agent" section to dojo's root AGENTS.md
  (Sidney authorizes two-way design-note transfer); first two entries: the
  weights-hash rule and the hybrid root-selection note.
- Verification: 23 passed; ruff clean.

## 2026-07-02 ~23:05 EDT — sati.tiling geometry

- Pure-geometry module per v1 §6.1: `Affine` (`[sx,sy,tx,ty]`,
  apply/invert/compose), `Level`, `SliceSpec`, `FramePlan`, `ProjectedBox`.
- Level stack computed from full-res directly (floors never accumulate),
  terminating on the longer side ≤ nominal or `max_levels`.
- Per-axis tiling with SAHI integer-pixel stride, last-tile-flush rule,
  sub-nominal axes as single stretched spans; frame-unique `slice_id`s.
- `project_box`: fullres → level → clip → emitted pixels with
  `visible_area_frac`, `truncated_edges`, both gate flags recorded.
- Judgment call: size gate evaluated on the clipped (visible) box
  (QUESTIONS item 10).
- Verification: §12 property-test spine (seeded random frames/factors/overlaps;
  affine round-trips, gapless coverage, stretch bookkeeping) — 33 passed in ~5 s
  after bounding random ranges (initial ranges hit multi-million-slice draws,
  48 s); ruff clean.

## 2026-07-02 23:15 EDT — Two-stage recombination

- Completed v1 §6.2 in `sati.tiling`: `SlicePrediction` → `merge_within_level`
  (confidence filter, lift through slice affine, greedy NMS in level coords,
  contributing `source_slice_ids`) → `merge_cross_level` (lift by realized
  scale, NMS ranked by raw confidence, `source_levels`) → `recombine`.
- `greedy_nms` is deterministic (score desc, index tie-break), class-aware by
  default with `class_agnostic` switch; winners record suppressed indices.
- `recombination_hash` added to `sati.hashing`, scoped per record_type:
  within-level params on `level_merge`, + cross-level on `final_merge`.
- Edge-fragment test documents the accepted §6.2 v1 bias (touching fragments
  at a slice boundary both survive Stage 1).
- Verification: `pytest tests/` — 44 passed in ~3 s; ruff clean.
- Next: `sati.manifests` (explicit PyArrow schemas + column-by-column writers
  + `_metadata.json` sidecar).

## 2026-07-03 03:13 EDT — sati.manifests + full §4.7 hash family

- `sati/manifests.py`: explicit hand-written PyArrow schemas for all five
  tables (`fullframes`, `annotations`, `slices`, `slice_annotations`,
  `validation_results` — the tall `record_type` table with nullable
  stage-specific columns); `build_table` constructs column-by-column against
  the declared schema (missing keys → typed nulls, unknown keys dropped, no
  Arrow inference anywhere); `read_manifest` asserts exact schema match.
- `_metadata.json` sidecar: strict `ManifestMetadata` Pydantic model; writer
  never sorts keys (class-map insertion order is meaning) and shares no JSON
  helper with the hasher, per §4.5; `verify_metadata_hashes` recomputes every
  hash from its paired recipe-wrapped `*_source`.
- `sati/hashing.py` completed the §4.7 family: `dataset_hash` (sorted
  relpath+size, no pixels/mtime), `class_map_hash` (order-sensitive),
  `preprocessing_hash` (nominal/stack/overlap/stretch policy;
  `image_format` deliberately excluded per spec — QUESTIONS item 11). Every
  hash now exposes a `*_full_source` builder so sidecars store the exact
  verifiable source object.
- Verification: `pytest tests/` — 56 passed in ~4 s (incl. null-leading list
  column, sidecar order round-trip, tampered-source re-verification failing as
  it should); ruff clean.
- Next: `sati.io` (image read/decode, COCO/YOLO adapters) or `sati.materialize`
  — leaning `io` since `prepare` needs ingestion first.

## 2026-07-03 03:16 EDT — sati.io: ingestion adapters + splits

- `sati/io.py`: COCO and YOLO exist only here — everything downstream speaks
  manifests and half-open `xyxy`.
- COCO adapter: single JSON (all frames → `default_split`) or a directory of
  `{training,validation,holdout}.json`; image ids canonicalized to strings;
  sparse category ids remapped to contiguous 0-based indices ordered by COCO
  id; `xywh → xyxy` at the boundary; duplicate frame ids and conflicting
  category names across split files are hard errors.
- YOLO adapter: `data.yaml` (`train`/`val`/`test` → training/validation/
  holdout), names as list or mapping, normalized `cxcywh` → absolute `xyxy`,
  header-only size reads (PIL), missing label file = negative frame,
  frame_id = image filename stem.
- Split TOML (§4.6): unknown keys, duplicate ids, and TOML-ids-missing-from-
  source are errors; source frames absent from the TOML drop with a warning.
- Pixels: `read_image` → RGB uint8 HWC; `detect_color_mode` flags grayscale.
- Decision junctions logged as QUESTIONS items 12–13 (COCO image root
  resolution; YOLO stem-as-frame-id).
- Verification: `pytest tests/` — 67 passed in ~4 s; ruff clean.
- Next: `sati.materialize` (slice/label emission + `data.yaml`) then the
  `prepare` orchestration in `sati.cli`.

## 2026-07-03 03:23 EDT — data.image_root (Sidney's call on QUESTIONS 12)

- Added `data.image_root` (optional `SatiPath`) to `DataConfig` with a
  validator rejecting it for `source_format=yolo`; wired through
  `ingest_coco(image_root=...)`; packaged `data/coco.yaml` carries
  `image_root: null`; v1 §3.3 table updated.
- Verification: `pytest tests/` — 69 passed; ruff clean.

## 2026-07-03 03:33 EDT — sati.materialize (the work of `prepare`)

- Zarr question resolved with Sidney: zarr stays v2; materialize is factored so
  the pixel source sits behind a narrow boundary — `build_level_pyramid(frame)
  → levels` (INTER_AREA, always from full-res, level 0 is the frame itself)
  feeding `emit_slice(level_pixels, slice_spec)` (crop + INTER_LINEAR stretch).
  Seam recorded in v1 §4.8.
- `materialize_frame`: decodes one frame (dims cross-checked against the
  ingest manifest — mismatch is a hard error), slices every level, writes
  `yolo/images/<split>/<id>_L<level>_S<slice>.<ext>` + matching label files
  (YOLO cxcywh normalized to emitted-slice pixels; drop-flagged projections
  excluded from labels but recorded in `slice_annotations`; zero-target slices
  get empty label files). Frames processed sequentially — one frame + stack in
  memory (§5).
- `materialize_dataset`: all four manifests + `_metadata.json` sidecar
  (class_map/dataset/preprocessing hashes with re-verifiable sources) +
  generated `data.yaml` embedding the dataset-level `preprocessing_hash`.
  Holdout frames materialize fully but stay out of `data.yaml`.
- `preprocessing_hash` gained a dataset-level form (stack *rule*, no realized
  stack) for the `data.yaml` embed; frame-level form unchanged.
- Judgment calls logged as QUESTIONS 14–15 (stretch interpolation
  INTER_LINEAR; dataset_hash source = emitted slice images).
- Verification: `pytest tests/` — 77 passed in ~5 s (incl. end-to-end
  materialize with sidecar re-verification and label-in-bounds checks);
  ruff clean.
- Next: `sati.backends` (Backend protocol + stock ultralytics) and the Typer
  CLI with `prepare`, making the pipeline runnable end to end.

## 2026-07-03 03:46 EDT — sati.backends + sati.train + CLI (prepare/train)

- `sati/backends.py`: `Backend` protocol (`train`/`load`/`predict`) +
  `UltralyticsBackend` — lazy ultralytics import (core works without the
  extra), RGB→BGR at the boundary (ultralytics treats ndarrays as BGR),
  detections parsed to `SliceDetection` in emitted-slice pixels. Model init:
  explicit `model.weights` wins, else `f"{arch}.pt"` hub-pretrained
  (QUESTIONS 17). §8 augmentation NOTE settled: `training.*` passed through
  untouched, stock defaults apply.
- `sati/train.py`: `locate_data_yaml` (prepared root or direct data.yaml —
  QUESTIONS 16), `check_preprocessing_gate` (§14 refuse-on-mismatch against
  the embedded hash), `execute_train` → `(checkpoint, checkpoint_hash)`.
- `sati/cli.py`: Typer app with `prepare` (per-level slice-count table — the
  §8 scale-starvation diagnostic) and `train`; dash rule honored (`--config-dir`,
  `--resolved-config` vs dash-free overrides). `execute_prepare` added to
  materialize (ingest → split TOML → materialize). Console script verified.
- **Tripwire ran for real** (`SATI_RUN_TRIPWIRE=1`): downloaded hub yolo26n.pt,
  predictions on a 640px noise slice came back in-bounds, no letterbox —
  coordinate contract verified against ultralytics 8.4.86. §7 NOTE folded into
  spec prose; §8 NOTE settled.
- Verification: `pytest tests/` — 83 passed, 1 skipped (tripwire, on demand);
  tripwire 1 passed with env flag; ruff clean; `sati --help` works.
- Next: `sati.infer` (sliced inference engine, shared by eval) +
  `sati.evaluation` + results writing into `validation_results.parquet`.

## 2026-07-03 09:38 EDT — sati.infer: sliced inference engine + results rows

- `sati/infer.py`: `infer_frame` (level stack in memory → `emit_slice` via the
  materialize pixel seam, no disk → batched `predict` at `training.batch` →
  §6.2 recombine); `rows_for_frame` builds `validation_results` rows —
  `frame_metadata` + `level_merge` + `final_merge` always, `slice_prediction`
  only under `results.cache_slice_predictions`; every row carries run_id + the
  full hash family; recombination_hash correctly scoped per record_type.
- `execute_infer`: `data.source_path` = image file or dir, `model.weights` =
  checkpoint; writes `validation_results.parquet` + the §4.5 `_metadata.json`
  sidecar (hashes re-verify; class_map empty for new imagery — eval will pass
  the prepared one). `sati infer` CLI command added.
- **Bug found & fixed:** parquet cannot write fixed-size lists containing
  nulls (`ArrowNotImplementedError`); `VALIDATION_RESULTS_SCHEMA`'s nullable
  geometry columns switched to variable-size lists, with a write-round-trip
  test pinning it. (Manifests tests had only built, never written, null rows.)
- Test-env fix: tripwire's hub download (`yolo26n.pt` in repo root) collided
  with the hub-name-hashes-verbatim test — test now isolates CWD (content-hash
  of a real local file matching the hub name is correct behavior: it is what
  ultralytics would load); `/*.pt` gitignored.
- Judgment call: `level_merge` rows written unconditionally (QUESTIONS 18).
- Verification: `pytest tests/` — 89 passed, 1 gated; ruff clean.
- Next: `sati.evaluation` (final_merge vs GT mAP via pycocotools, per-level
  diagnostic) + `sati eval`; then `export`, `inspect`, `convert`.

## 2026-07-03 14:34 EDT — sati.evaluation + `sati eval`

- `sati/evaluation.py`: `execute_eval` = infer over the prepared validation
  split (fixed split, QUESTIONS 20) + scoring. Reads fullframes/annotations
  manifests + the prepared sidecar; results rows carry the prepared
  `dataset_hash`/`class_map_hash` alongside the run's own hash family; writes
  `validation_results.parquet`, its sidecar (prepared class map), and
  `metrics.json`.
- Scoring: pycocotools mAP50-95/50/75 + per-class AP on **final_merge vs
  full-frame full-res GT** (in-memory COCO objects, stdout suppressed);
  empty-GT/empty-DT guarded.
- §11 diagnostics: per-level `final_merge` count + confidence mean/p50 (by
  `source_levels` membership), and GT recall@0.5 binned by long side on a
  log2 ladder (16…1024+ px) with greedy class-aware matching.
- `sati eval` CLI command prints a metric table.
- **Oracle round-trip test**: a backend primed with the true per-slice GT
  projections (via `project_box`, in plan order) scores mAP50 > 0.9 through
  the full prepare → slice → predict → recombine → score loop — every
  coordinate transform in v1 verified end to end.
- Verification: `pytest tests/` — 95 passed, 1 gated; ruff clean.
- Next: `export` (checkpoint + inference_config.yaml sidecar), `inspect`
  sub-app, `convert` — the remaining §5 surface.

## 2026-07-03 14:37 EDT — sati.export + sati.convert

- `sati/export.py`: `execute_export` wraps the backend's native export
  (`Backend.export` added to protocol + ultralytics impl) and writes the §10
  `inference_config.yaml` sidecar: slicing recipe + stretch policy,
  recombination params, model summary, class map (picked up from a prepared
  root at `data.source_path` when present, else empty — QUESTIONS 23), and the
  hash family (`checkpoint_hash` digests the source `.pt`). Requires a local
  checkpoint file. `sati export --format onnx` CLI command.
- `sati/convert.py`: `execute_convert` dispatch — `to=manifests` ingests
  COCO/YOLO into fullframes+annotations parquet + sidecar (no slice tables,
  convert never slices); `to=coco` writes one `<split>.json` per split present
  (re-ingestable by `ingest_coco`, verified by a round-trip test); `to=yolo`
  writes labels from stored `bbox_rel_cxcywh` with **symlinked** images
  (QUESTIONS 22). `sati convert --to <target>` CLI command.
- Verification: `pytest tests/` — 101 passed, 1 gated; ruff clean.
- Next: `sati.inspect` (§5.1 read-only sub-app: config|dataset|slices|results|
  model, `--json`) — the last v1 CLI surface.

## 2026-07-03 14:40 EDT — sati.inspect: v1 CLI surface complete

- `sati/inspect.py`: five read-only summary functions returning the dicts that
  `--json` dumps; the CLI renders rich tables from the same dicts so human and
  machine views can't drift. Never a forward pass, never a pixel decode, never
  a write to the inspected store.
  - `config`: validation status (reports, doesn't raise), derived values, the
    hash family the config *would* produce — via `sati.hashing` itself, so
    preview and actual can never diverge (pinned by test) — and `--diff`
    (flattened field-by-field comparison).
  - `dataset`: frame/annotation/per-split/per-class counts, class map,
    zero-annotation frames, dimension + box-long-side distributions.
  - `slices`: per-level slice & object counts, drop tallies, stretched-axis
    frequency, YOLO tree on-disk bytes, `slice_uri` spot check.
  - `results`: record_type breakdown, run_ids, hash families present, sidecar
    re-verification, metrics.json if present.
  - `model`: `checkpoint_hash`, best-effort ultralytics metadata (unreadable
    checkpoint → note, not crash), sibling `inference_config.yaml` summary.
- Registered as the `sati inspect` sub-app; `--json` is a best-effort dump per
  the §5.1 NOTE (versioned schema deferred to the first automation consumer).
- Verification: `pytest tests/` — 109 passed, 1 gated; ruff clean.
- **Milestone: every v1 §5 command and §12 module now exists** —
  `prepare · convert · train · eval · infer · export · inspect`. Remaining v1
  work is polish (open QUESTIONS items) and a real end-to-end training run,
  ideally on the synth dataset builder's Stage 1 output (DESIGN-DOC-SYNTH).
