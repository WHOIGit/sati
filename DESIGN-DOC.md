# SATI — Scale-Aware Training and Inference

**Status:** Design draft · **Last updated:** 2026-06-22

SATI is a package for object detection training and inference over **large, medium, and
small pixel-size objects inside very large images**. It streamlines a multi-resolution
slicing workflow so that a single, fixed model input size can cover many object scales.

---

## 1. Overview & Motivation

Detecting objects whose sizes span orders of magnitude inside very large frames (e.g.
7680×4320 and up) is hard: downsampling the whole frame to the model's input size destroys
small objects, while running the model at native resolution misses the global context that
large objects need, and is often infeasible memory-wise. This project is inspired by [SAHI](https://github.com/obss/sahi).

**SAHI** (Slicing Aided Hyper Inference) addresses part of this by tiling the full-resolution
image into fixed-size overlapping slices, running inference per slice, and merging the
results — with an optional single full-image pass for large objects. SAHI uses **one** slice
size against **one** input resolution (+ optional full frame), and only indended for inference.

**SATI generalizes this idea.** It applies a fixed slice size (e.g. 1024×1024) against the
full-resolution image **and against a configurable stack of downsampled full-frames**. A
single patch size therefore samples multiple scales: small objects are caught in full-res
slices, large objects are caught in slices taken from coarse downsampled frames. This applies
to both **inference** (SAHI-like) and **training** (slices + scale-aware metrics/loss).

### Goals
- One fixed slice size covering many detection scales via a multi-resolution full-frame stack.
- Configurable resolution stacks (explicit scales, explicit resolutions, or automatic).
- Fully **backtraceable** slicing/resizing/annotation via a manifest.
- Training and validation scored against **fully recombined predictions vs full-frame,
  full-resolution annotations**.
- A SAHI-like inference command that takes a model + a patch/resolution config and emits
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
- **Slice / patch:** a fixed-size crop (e.g. 1024×1024) taken from a resolution level.
- **Overlap ratio:** fractional slice-to-slice overlap (e.g. 0.25 = 25%), as in SAHI.
- **Manifest:** the backtraceable ledger linking every slice/resize/annotation back to the
  full-frame full-res source via an invertible affine transform.
- **Recombination:** merging per-slice, per-level predictions into a single set of full-frame
  full-res predictions.
- **Inference config:** the YAML emitted alongside an exported model describing the patch size
  and downsample steps to use at inference time.

---

## 3. Configuration (Hydra) ↔ Schemas (Pydantic)

**Pydantic schemas in `sati/schemas/` are the single source of truth** for which options are
valid and for function input/output typing. **Hydra** (`configs/`) composes runtime
configuration, which is validated against the pydantic schemas before use. If the two ever
disagree, the schema wins.

### Resolution stack configuration modes

1. **Explicit scales** — `[1.0, 0.5, 0.25, 0.125]`. Each scale is applied to the full-frame.
2. **Explicit resolutions** — `[(7680,4320), (3840,2160), (1920,1080), (960,540)]`. A
   **warning** is emitted when an entry's aspect ratio does not match the source's.
3. **Automatic** — a starting resolution (e.g. `1.0` or `(7680,4320)`) plus a downsample
   factor (e.g. `2`). Each step downsamples by the factor until the **whole full-frame fits in
   a single slice** — i.e. `max(frame_dim) ≤ slice` (terminate by the *longer* side, not the
   shorter; see "Slice-dimension reduction" below for why). Optional `max_levels` and an
   optional short-side floor guard bound the stack for extreme aspect ratios.

### Slice configuration
- **Slice size** (e.g. `1024×1024`).
- **Overlap ratio** (e.g. `0.10`), applied in height and width.
- **Edge / truncation policy** (see §6 and §14): how to treat objects cut by a slice edge.

### Slice-dimension reduction (no letterbox / stretch)
SATI does **not** pad (letterbox) or distort (stretch) frames to force a square slice. Instead,
tiling is applied **per-axis**:
- an axis `≥ slice` → overlapping tiles of slice length;
- an axis `< slice` → a **single native, stride-aligned span** along that axis (the slice
  dimension is *reduced* to the frame's extent, not padded).

So slices become rectangular at coarse levels, and the pyramid terminates (per the automatic
rule above, and validated for explicit modes) at the level where the **whole full-frame** is fed
as one input rather than a slice. This keeps scale **isotropic** (no aspect distortion), stores
no padded pixels, and keeps the manifest affine a clean per-axis scale+translate. Non-standard
(rectangular / whole-frame) slices are relatively rare versus standard squares; the backend
normalizes them at the framework boundary (see §6.1) so they remain harmless for training and
inference. Extreme aspect ratios (e.g. 11260×2280) are handled by the same rule — long axis
tiled, short axis reduced — bounded by `max_levels` / short-side floor.

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

The project's source of truth is **Parquet (annotations + manifest) + zarr (image pyramid)**.
COCO, YOLO, and precomputed tiles are import/export adapters only.

### 4.1 Canonical Parquet (annotations & predictions)

One canonical table holds **full-frame, full-resolution** localizations, storing both box
conventions so downstream code never re-derives them:

| column | type | notes |
|---|---|---|
| `fullframe_id` | str/int | source full-frame id |
| `source_uri` | str | path/URI of the source image |
| `ann_id` | int | annotation/prediction id |
| `class_id` | int | |
| `class_name` | str | |
| `bbox_abs_xyxy` | int×4 | COCO-style, absolute full-res pixels |
| `bbox_rel_cxcywh` | float×4 | YOLO-style, normalized center-xy + size to full-frame |
| `score` | float | null for ground truth; set for predictions |

### 4.2 Manifest = derived Parquet partitions

The manifest is the same columnar store extended with derived rows, **hive-partitioned by
`fullframe_id` / `level` / `slice_id`**. The canonical full-frame/full-res annotations are simply
`level = 0`, whole-frame. Derived rows add:

- `level`, `scale`, `res_wh` — which resolution level and its size.
- `slice_id`, `slice_origin_xy`, `slice_wh` — slice identity and placement within the level.
- **`affine`** — an **invertible** per-axis transform (scale + translate) mapping slice
  coordinates → full-frame full-res coordinates. No pad/stretch terms are needed since SATI
  reduces the slice dimension rather than padding/distorting (§3). This is the backbone of
  traceability and recombination.
- `visible_area_frac` (float), `truncated_edges` (one or several of top,bottom,left,right) — for edge handling
  and the recombination heuristic.
- box coordinates expressed in slice space (abs and relative).

Because every derived artifact carries its affine back to full-res, **any slice, resize, or
converted annotation is always backtraceable** to the source frame.

### 4.3 Zarr image pyramid

- One zarr group **per source image**; arrays named `0, 1, 2, …`, one per resolution level.
- **Chunk size = 2× the slice size** by default (so on-the-fly slicing reads at most ~4 chunks per slice). Configurable.
- Group **attrs** record: the resolutions / scale factors, the downsample method,
  source dimensions, and aspect ratio.
- Minimal, purpose-built layout. OME/NGFF interop is a possible later add (see §15).

### 4.4 Conversions
Canonical (parquet + zarr) ↔ **COCO**, **YOLO**, and **precomputed tiles**, round-tripping box
coordinates through the manifest affine.

---

## 5. Pipeline

```
prepare  ->  train  ->  eval  ->  export
prepare -> infer -> results
```

- **prepare** — ingest a COCO- or YOLO-based dataset → canonical Parquet + zarr pyramid +
  manifest.
- **train** — stock ultralytics over slices generated from the pyramid.
- **infer** — SAHI-like multi-resolution sliced inference (§9).
- **eval** — metrics on recombined predictions vs full-frame full-res ground truths (§11).
- **export** — checkpoint/pt-file + inference config (§10).
- **results** — parquet format final out output, optinally coco or yolo prediction output.

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
zarr pyramid across multiple resolution levels**, emit the **manifest affine** per slice, do
**per-axis slice-dimension reduction** (§3), and produce our dual box convention (§4) — none of which SAHI's API can
express, so we write the slicer regardless and would only borrow trivial geometry while taking on
coupling to SAHI's data model (`SliceImageResult` / `ObjectPrediction`) and release cadence.

What we **borrow without depending on the package**:
- **Semantics, not code:** keep SAHI's `overlap_*_ratio` and `min_area_ratio` meanings identical
  so users migrating from SAHI get predictable behavior (see §3).
- **`shapely`** (a direct core dependency — what SAHI itself uses) for box/polygon clipping and IoU.
- For recombination dedup (§6.2), implement **NMM / GREEDYNMM** ourselves (or a small standalone
  such as torchvision NMS / `lsnms`) rather than `sahi.postprocess`, since our merge is cross-scale
  and manifest-aware.

> If bit-for-bit parity with SAHI inference outputs ever becomes a product requirement, depending
> on SAHI would buy that — but SATI is a *generalization* of SAHI (multi-resolution + training), so
> parity is not a goal and not worth the coupling.

### 6.1 Slicing
For each configured resolution level, tile the (downsampled) full-frame **per-axis** (§3): an
axis `≥ slice` is cut into overlapping tiles of slice length; an axis `< slice` becomes a single
native, stride-aligned span (slice-dimension reduction — no padding/stretch). The pyramid
terminates at the level where the whole frame is fed as one input. For each slice, write its
per-axis affine, the dropped-annotation set from the per-resolution **size gate** (§3), and
per-object `visible_area_frac` / `truncated_edges` flags into the manifest. For training, input
annotations are converted to slice space **on the fly** (or at materialization time).

**Non-standard (rectangular / whole-frame) slices are harmless.** YOLO is fully convolutional
with a stride-divisible input, so any size runs correctly; inference predicts on the native
rectangular size directly. For training, heterogeneous shapes can't be stacked into one batch,
so the backend's standard letterbox-to-`imgsz` (gray-114, already used for every image)
normalizes them at the framework boundary — in-distribution and consistent with inference.
Because such slices are rare, the padded-batch compute is negligible; shape-bucketing
(ultralytics `rect=True` style) is an optional optimization. The only real risk is data
coverage, addressed in §8.

### 6.2 Recombination (pluggable `Strategy`)
Recombination merges per-slice, per-level predictions into one full-frame, full-res set. It is
a **pluggable strategy** because the optimal heuristic is an open research question. The
**default** heuristic:

1. Map all candidate detections to full-frame full-res coordinates via their manifest affine.
2. Cluster candidates by IoU.
3. Within a cluster, prefer detections from a slice where the object is **fully contained**
   (not edge-truncated); among those prefer the slice where the object's **pixel size is
   largest** (the highest-resolution slice that still contains the whole object); then prefer
   **highest confidence**. Resolve class by confidence-weighted vote.
4. Apply cross-scale NMS / NMM to dedupe.

Validation/inference metrics are always computed on these recombined predictions against
full-frame full-res ground truth.

---

## 7. Backend abstraction (`sati.backends`)

A `Backend` protocol exposes `train / val / predict / export`, decoupling SATI's novel parts
(pyramid, manifest, tiling, recombination, evaluation, inference command) from the detector.

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
- a **patch + resolution-sampling YAML config** (the one shipped with the model, or an override).

It internally builds the resolution stack and slices, runs the backend per slice, recombines
(§6.2), and outputs final boxes + class labels in **YOLO and/or COCO**. Intermediate image
files and annotations may be **optionally cached** for inspection/traceability.

---

## 10. Export

A finalized checkpoint is exported **together with an inference configuration file** that
specifies the patch size and downsample steps to use at inference. Ultralytics' native ONNX /
TensorRT export is leveraged for the model artifact; SATI adds the inference-config sidecar so
the deployment exactly reproduces the intended multi-resolution sampling.

---

## 11. Evaluation (`sati.evaluation`)

Metrics (mAP / PR, pycocotools-compatible) are computed on **recombined predictions vs
full-frame, full-resolution ground truth**, with **per-scale breakdowns** so we can see how
each resolution level contributes. Every prediction remains **manifest-backed**, so any score
can be traced to the exact slices/levels that produced it. In Phase 1 this runs as a standalone
evaluator over training/validation outputs.

---

## 12. Module Layout

| module | responsibility |
|---|---|
| `sati.schemas` | Pydantic — source of truth for options & I/O typing |
| `sati.config` | Hydra glue; `configs/` composition, validated against `sati.schemas` |
| `sati.io` | Data sources & sinks: zarr, COCO, YOLO over local fs / S3 / web |
| `sati.manifest` | Parquet-backed traceability ledger |
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
  cut at another, double counting, class disagreement. → Pluggable `Strategy` with a documented
  default (§6.2).
- **Edge / cut-off objects.** → Explicit, configurable truncation policy; recommend the
  **overlap ratio ≥ (largest expected object / slice size)** so big objects survive whole in at
  least one slice.
- **Coarse-level / extreme-aspect bookkeeping.** Per-axis slice-dimension reduction (§3) avoids
  pad/stretch entirely, but the per-axis scale+translate must stay invertible or recombined boxes
  misplace. → Every transform is stored as an invertible affine in the manifest (§4.2);
  extreme aspect ratios bounded by `max_levels` / short-side floor.
- **Scale coverage starvation.** Rare coarse-level slices under-represent large objects. →
  Per-level sampling weights + per-scale eval breakdown (§8, §11).
- **Over-aggressive annotation size gate.** Too high a per-resolution minimum silently drops
  real targets. → Conservative default (≈16 px short side); monitor dropped-annotation counts in
  the manifest (§3).
- **Storage blow-up vs determinism.** A full pyramid + materialized slices for huge imagery is
  large; on-the-fly slicing is lean but must be reproducible. → On-the-fly default, **seeded**;
  materialization is opt-in.
- **mmdet maintenance risk.** → Stub only; never on the Phase-1 critical path.

---

## 15. Roadmap / Phasing

- **Phase 1 (foundation):** `sati.schemas` / `sati.config`, `sati.io`, zarr pyramid +
  `sati.manifest`, `sati.tiling` (slice + recombine), `sati.datasets`, **stock-ultralytics**
  training, offline recombined-full-frame evaluation, the SAHI-like inference command +
  exported inference config, `sati.cli`, `pyproject.toml`.
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
