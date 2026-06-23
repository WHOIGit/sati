# SATI — Scale-Aware Training and Inference

**Status:** Design draft · **Last updated:** 2026-06-23

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
- **Nominal slice size:** the configured sampling window size (e.g. 1024×1024).
- **Actual slice size:** the real emitted slice/window size. It usually matches the nominal
  size, but may be smaller on axes where the resolution level is smaller than the nominal
  slice size.
- **Backend input size:** the tensor shape passed to the detector after backend preprocessing
  such as resize or letterbox.
- **Overlap ratio:** fractional slice-to-slice overlap (e.g. 0.25 = 25%), as in SAHI.
- **Manifest tables:** the backtraceable ledger linking every slice/resize/annotation back to
  the full-frame full-res source via invertible transforms.
- **Recombination:** merging per-slice, per-level predictions into a single set of full-frame
  full-res predictions.
- **Inference config:** the YAML emitted alongside an exported model describing the nominal
  slice size, resolution stack, backend preprocessing, and recombination settings to use at
  inference time.

---

## 3. Configuration (Hydra) ↔ Schemas (Pydantic)

**Pydantic schemas in `sati/schemas/` are the single source of truth** for which options are
valid and for function input/output typing. **Hydra** (`configs/`) composes runtime
configuration, which is validated against the pydantic schemas before use. If the two ever
disagree, the schema wins.

### Resolution stack configuration modes

1. **Explicit scales** — `[1.0, 0.5, 0.25, 0.125]`. Each scale is applied isotropically to the
   full-frame.
2. **Automatic factor** — start from level 0/full-res and repeatedly divide dimensions by a
   configured downsample factor (e.g. `2` or `4`) until the **whole full-frame level fits in a
   single nominal slice** — i.e. `max(level_dim) ≤ nominal_slice_size` (terminate by the
   *longer* side, not the shorter; see "Slice-dimension reduction" below for why). Optional
   `max_levels` and an optional short-side floor guard bound the stack for extreme aspect
   ratios.

Explicit target resolutions are intentionally not supported in Phase 1. Levels are derived
from source dimensions plus scale, which preserves isotropic scale and avoids aspect-ratio
ambiguity.

> A dry-run tool can output the expected resolutions for a given input image(s).

### Slice configuration
- **Nominal slice size** (e.g. `1024×1024`).
- **Overlap ratio** (e.g. `0.10`), applied in height and width.
- **Training `min_area_ratio`** (default `0.25`): how much of an object must be visible inside
  a slice before it is included as a training target.
- **Recombination edge policy** (see §6.2 and §14): how near-edge predictions are weighted
  during cross-level recombination.

> a dry-run tool can show how many patches per level and in total would be created for given input image(s).

### Slice-dimension reduction
SATI does **not** pad (letterbox) or distort (stretch) frames to force a square slice. Instead,
tiling is applied **per-axis**:
- an axis `≥ nominal_slice_size` → overlapping tiles of nominal slice length;
- an axis `< nominal_slice_size` → a **single native, stride-aligned span** along that axis (the slice
  dimension is *reduced* to the frame's extent, not padded).

So slices become rectangular at coarse levels, and the pyramid terminates (per the automatic
rule above, and validated for explicit scale lists) at the level where the **whole full-frame** is fed
as one input rather than a slice. This keeps scale **isotropic** (no aspect distortion), stores
no padded pixels, and keeps the slice affine a clean per-axis scale+translate. Non-standard
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
- **Default:** drop any annotation whose **shorter side < 16 px** at the level (≈ area < 256 px²).
- **Rationale:** YOLO's finest detection head has stride 8 px (P5; 16 px-ish for P6), so 16 px
  ≈ 2 feature cells — the practical floor for a learnable target and comfortably above
  downscaling box-rounding noise.
- **Configurable:** the cutoff is a config value (`sati/schemas/`), commonly raised to **32 px
  or higher** for cleaner targets (32 px aligns with COCO's "small" boundary). Expressed as
  min shorter-side **or** area.

---

## 4. Data Model

The project's source of truth is **Parquet tables + zarr image pyramids**. COCO, YOLO, and
precomputed tiles are import/export adapters only. Canonical full-frame annotations, slice
geometry, derived slice annotations, and validation results are separate tables so `level = 0`
always means the full-resolution pyramid level, never a special canonical row.

### 4.1 `annotations.parquet`

Canonical **full-frame, full-resolution** ground truth. This table has no slice semantics and
is the training-data source of truth.

| column | type | notes |
|---|---|---|
| `fullframe_id` | str/int | source full-frame id |
| `source_uri` | str | path/URI of the source image |
| `ann_id` | int | annotation id |
| `class_id` | int | |
| `class_name` | str | |
| `bbox_abs_{x,y,w,h}` | ints | COCO-style, absolute full-res pixels |
| `bbox_rel_{cx,cy,w,h}` | floats | YOLO-style, normalized center-xy + size to full-frame |

### 4.2 `slices.parquet`

One row per generated slice/window, partitioned by `fullframe_id` / `level` / `slice_id`.
`level = 0` is reserved for the full-resolution pyramid level.

- `fullframe_id`, `level`, `slice_id`.
- `scale`, `level_wh` — which resolution level and its dimensions.
- `slice_wh`, `slice_origin_xy` — sampling window and
  actual placement within the level. Typically `slice_wh` is the same as `nominal_slice_wh`.
- `is_whole_level` — true when the entire resolution level is represented by one slice.
- **`affine_slice_to_fullres`** — an invertible per-axis transform mapping slice coordinates
  to full-frame full-res coordinates.

### 4.3 `slice_annotations.parquet`

Derived annotation projections, partitioned by `fullframe_id` / `level` / `slice_id`.
Rows link back to `annotations.parquet` via `ann_id`.

- `fullframe_id`, `level`, `slice_id`, `ann_id`, `parent_ann_id`.
- box coordinates in level space and slice space, both absolute and relative.
- `visible_area_frac` — clipped slice-visible area divided by the annotation area at that
  resolution level.
- `truncated_edges` — zero or more of `top`, `bottom`, `left`, `right`.
- `dropped_by_size_gate`, `dropped_by_min_area_ratio` — explicit bookkeeping for why a
  possible target was excluded from training.

Because every derived artifact links through `slices.parquet`, **any slice, resize, converted
annotation, or prediction is always backtraceable** to the source frame.

### 4.4 `validation_results.parquet`

Offline validation and inference artifacts live outside the direct model training loop.
Rows are partitioned by run, epoch, `fullframe_id`, level, and slice where applicable.

- `result_kind`: `slice_raw`, `level_merge`, or `final_merge`.
- `slice_raw` rows have `level` and `slice_id`.
- `level_merge` rows have `level` and no specific `slice_id`.
- `final_merge` rows represent the full-frame output and have no specific level or slice.
- `merge_strategy_name` and `merge_strategy_config_hash` identify recombination experiments.
- Final candidates track the contributing levels and, where useful, the slice provenance that
  fed their within-level merge.

### 4.5 Zarr image pyramid

- One zarr group **per source image**; arrays named `0, 1, 2, …`, one per resolution level.
- **Chunk size = 2× the nominal slice size** by default (so on-the-fly slicing reads at most
  ~4 chunks per slice). Configurable.
- Group **attrs** record: the resolutions / scale factors, the downsample method,
  source dimensions, and aspect ratio.
- Minimal, purpose-built layout. OME/NGFF interop is a possible later add (see §15).

### 4.6 Conversions
Canonical (parquet + zarr) ↔ **COCO**, **YOLO**, and **precomputed tiles**, round-tripping box
coordinates through the slice affine.

---

## 5. Pipeline

```
prepare  ->  train  ->  eval  ->  export
prepare -> infer -> results
```

- **prepare** — ingest a COCO- or YOLO-based dataset → canonical Parquet + zarr pyramid +
  manifest tables.
- **train** — stock ultralytics over slices generated from the pyramid.
- **infer** — SAHI-like multi-resolution sliced inference (§9).
- **eval** — metrics on recombined predictions vs full-frame full-res ground truths (§11).
- **export** — checkpoint/pt-file + inference config (§10).
- **results** — Parquet final output, optionally exported as COCO or YOLO predictions.

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
their slice affines and merged into `level_merge` predictions. This stage removes duplicates
caused by overlapping slices at the same scale and exposes SAHI-like postprocess options:
`NMS`, `NMM`, `GREEDYNMM`, optional `LSNMS`, IoU threshold, confidence threshold, and
class-aware vs class-agnostic matching.

Every `level_merge` prediction records provenance, including contributing `slice_id`s and
whether any contributing raw prediction touched or was near a slice edge.

#### Stage 2: cross-level merge

Across all levels for one `fullframe_id`, `level_merge` predictions are clustered by IoU
(class-aware by default) and converted into `final_merge` predictions in full-frame full-res
coordinates. The Phase-1 default selects one candidate per cross-level cluster using:

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
compound multiplicatively. This is calculated at the within-level merge stage right since. 

Final candidates record which level they came from. Weighted box fusion / cross-level NMM is a
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

### Later phases — resolution-aware loss
Implement a custom loss via overriding `init_criterion` on a custom model plus subclassing
`DetectionTrainer`, threading per-object metadata (source scale, truncation flags) through a
custom dataset / collate. The reweighting is a **configurable research knob**:

- reweight by **resolution level**,
- down-weight **slice-edge-truncated** objects,
- **both**, or
- **neither**.

The config surface for this knob is designed now (in `sati/schemas/`) even though the
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
| `sati.schemas` | Pydantic — source of truth for options & I/O typing |
| `sati.config` | Hydra glue; `configs/` composition, validated against `sati.schemas` |
| `sati.io` | Data sources & sinks: zarr, COCO, YOLO over local fs / S3 / web |
| `sati.manifests` | Parquet-backed slice, slice-annotation, and validation-result tables |
| `sati.datasets` | Framework-facing dataset producing slices (on-the-fly or materialized) |
| `sati.tiling` | Slicing and recombination |
| `sati.backends` | `Backend` protocol + ultralytics impl + mmdet stub |
| `sati.train` | Training orchestration |
| `sati.infer` | SAHI-like multi-resolution inference command |
| `sati.evaluation` | Recombined-full-frame metrics |
| `sati.cli` | Command-line entrypoints |

---

## 13. CLI & Packaging

`pyproject.toml` defines core dependencies, optional extras, and a console script.

- **Core deps:** `pydantic`, `hydra-core`, `zarr`, `pyarrow` (Parquet), `numpy`,
  **`shapely`** (box/polygon clipping & IoU for slicing/recombination — §6.0), plus an image
  codec (e.g. `opencv-python` / `pillow`).
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
  SAHI-like within-level merge, configurable cross-level strategy, and strategy hashes in
  `validation_results.parquet` (§6.2).
- **Edge / cut-off training targets.** → Training uses SAHI-compatible `min_area_ratio`
  (default `0.25`) so clipped targets are kept only when enough object area is visible.
- **Near-edge predictions.** → Cross-level merge has configurable `edge_proximity` and
  `edge_favorability_penalty`; set the penalty to `0.0` to ignore near-edge weighting. Recommend
  **overlap ratio ≥ (largest expected object / nominal_slice_size)** so big objects survive
  whole in at least one slice.
- **Coarse-level / extreme-aspect bookkeeping.** Per-axis slice-dimension reduction (§3) avoids
  pad/stretch entirely, but the per-axis scale+translate must stay invertible or recombined boxes
  misplace. → Every transform is stored as an invertible affine in `slices.parquet` (§4.2);
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

- **Phase 1 (foundation):** `sati.schemas` / `sati.config`, `sati.io`, zarr pyramid +
  `sati.manifest` tables, `sati.tiling` (slice + recombine), `sati.datasets`,
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
