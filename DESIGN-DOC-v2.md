# SATI v2 — Expansion Design

**Status:** Design draft · **Last updated:** 2026-07-04
**Baseline:** this is a **delta document over `DESIGN-DOC-v1.md`** — v1's concepts,
conventions, manifests, hash recipes, module layout, and CLI carry forward unchanged except
where a section here explicitly amends them. Read v1 first.

v2's theme: remove v1's two deliberate accuracy/efficiency taxes — **stretch distortion**
and **materialize-everything storage** — and open the **research surface** (recombination
strategies, sampling, sweeps) that v1 intentionally pinned shut.

| # | feature | replaces / amends in v1 |
|---|---|---|
| §1 | zarr image pyramid | in-memory level building (v1 §5) |
| §2 | on-the-fly slicing dataset + custom ultralytics dataset/trainer | materialize-only training (v1 §4.8, §8) |
| §3 | native rectangular & whole-frame slices + backend preprocessing capture | stretch rule (v1 §6.1) |
| §4 | explicit scale lists, short-side floor | automatic-factor-only stacks (v1 §6.1) |
| §5 | NMM/GREEDYNMM/LSNMS, favorability, edge penalty | NMS-only recombination (v1 §6.2) |
| §6 | post-hoc recombination sweeps | — (enabled by v1's cache flag) |
| §7 | per-level sampling weights, `negative_sample_ratio` | use-everything-once sampling (v1 §8) |
| §8 | parameter-sweep config generation | single-run composition (v1 §3) |
| §9 | per-scale mAP semantics | diagnostic-only breakdown (v1 §11) |
| §10 | schema/manifest evolution & migration rules | — |
| §11 | further features (not fleshed out) | — |
| §1.1 | `inspect pyramid` + zarr-aware `inspect slices`/`results` | v1 `inspect` sub-app (v1 §5.1) |
| §1.2 | `inspect dataset` scale recommendation | informs §4 explicit scale lists |

Config growth follows v1's **absent = unsupported** rule: each feature below lands as new
schema fields in `sati.config_schemas` plus new `config_defaults/` fragments, in the same
release as its implementation — never before.

---

## 1. Zarr image pyramid (`sati.pyramids`)

`prepare` gains a persistent multi-resolution store so levels are computed once and slices
can be read lazily forever after.

- One zarr group **per source image**; arrays named `0, 1, 2, …`, one per resolution level,
  stored `HWC` (`height, width, channels`).
- Canonical imagery is `uint8`; color as `RGB`, grayscale as one-channel. Non-`uint8`
  sources are rejected or explicitly converted at ingest, conversion recorded in metadata.
- Level dims use v1's deterministic floor sizing; metadata records requested **and**
  realized per-axis scale per level.
- Downsampling method configurable and recorded; default stays area/box resampling.
- Default chunk shape:
  `(min(level_h, 2·nominal_slice_h), min(level_w, 2·nominal_slice_w), channels)` —
  keeps a slice read touching ≤ 4 chunks without oversized chunks at small levels.
  Configurable.
- Group attrs: `fullframe_id`, `source_uri`, source dims, dtype, channel order,
  requested/realized scales, level dims, chunk shape, downsample method, and the pyramid's
  `preprocessing_hash`-relevant recipe. Source content hashing stays out of scope.
- Store layout: `<dataset_root>/pyramids/<fullframe_id>.zarr`. Local filesystem first;
  the zarr store abstraction leaves S3 open (see §11).

**Interaction with v1 manifests:** `slices.parquet` is unchanged — it already describes
windows in level coordinates; only the pixel *source* moves from materialized PNGs to lazy
zarr reads. `slice_uri` becomes nullable: populated only for slices that were materialized.

**Config:** `pyramid.enabled` (v2 default `true`), `pyramid.downsample_method`,
`pyramid.chunk_shape` (`auto` sentinel → the default above). With the pyramid enabled,
materialization (v1 §4.8) becomes opt-in per consumer: `materialize.training_tree` (keep
`true` to retain the v1 stock-training path) and the existing debug/inspection flags.

> [!NOTE]
> Undecided: whether `prepare` still supports the pyramid-less v1 path once v2 lands
> (`pyramid.enabled: false`), or whether the pyramid becomes mandatory and v1's in-memory
> mode is deleted. Keeping both costs a second code path in `prepare`/`infer`; deleting
> costs storage on quick one-off runs. Lean: mandatory pyramid, since `infer` on ad-hoc
> frames never writes one anyway (it stays in-memory) — decide when v2 `prepare` is built.

### 1.1 `inspect` extensions

The v1 read-only `inspect` sub-app (v1 §5.1) gains a zarr target and its existing targets
learn the pyramid/streaming world; the read-only, `--json`-capable contract is unchanged.

- **`inspect pyramid <root> <fullframe_id>`** (new) — reads zarr group attrs only: level
  dims, requested vs realized per-axis scales, chunk shape, dtype, channel order, downsample
  method, and the pyramid's hash recipe. Optional `--render <level>` writes a single
  downsampled thumbnail (the sole pixel-touching, sole file-writing path in all of `inspect`,
  and only to an explicit output path, never into the store).
- **`inspect slices`** now reports slices as **pyramid-backed** (null `slice_uri`) vs
  **materialized**, and cross-checks each `slice_id`'s level window against the zarr level
  dims so a stale manifest against a rebuilt pyramid is caught.
- **`inspect results`** gains group-by views over `recombination_hash` (surfacing §6 sweep
  variants) and the §9 per-scale breakdowns, both read straight from provenance columns.

### 1.2 `inspect dataset` automatic scale recommendation

`inspect dataset` should be able to recommend a v2 resolution stack from the ground-truth
object size distribution and a nominal slice size. The output is advisory: it explains the
candidate `slicing.scales` list, object coverage, uncovered outliers, and whether
whole-frame slices are needed for extreme large objects. It does not mutate configs.

The basic mode chooses a high-resolution endpoint, a low-resolution endpoint, and
intermediate levels:

- The high-resolution endpoint is constrained by small objects. At the highest-resolution
  recommended level, the small-object percentile should occupy at least a configurable
  slice-area fraction. Example knobs: `small_object_percentile`, `min_object_area_fraction`
  or stricter `target_object_area_fraction`.
- The low-resolution endpoint is constrained by large objects. At the lowest-resolution
  recommended level, the large-object percentile's longest side should be no more than a
  configurable fraction of the slice size. This long-side rule is required so long/thin
  objects are not accepted by area while still carrying high slice-edge truncation risk.
- Intermediate levels are inserted so adjacent levels do not exceed a configurable linear
  scale ratio, with optional snapping to power-of-two levels when the pyramid/backend
  benefits from regular factors.
- Objects that cannot satisfy the large-object long-side rule at any normal level trigger a
  whole-frame recommendation rather than forcing the scale stack to grow without bound.

A more involved automatic mode treats scale selection as an object-coverage problem. For an
object with full-resolution width `w`, height `h`, area `A`, long side `L`, nominal slice
size `S`, and downsample factor `d`:

```text
scaled_area_fraction = A / (d^2 * S^2)
scaled_long_side_fraction = L / (d * S)
```

An object is admissible at a level when:

```text
scaled_area_fraction >= min_object_area_fraction
scaled_long_side_fraction <= max_object_long_side_fraction
```

Each object therefore has a valid downsample interval:

```text
d >= L / (max_object_long_side_fraction * S)
d <= sqrt(A / (min_object_area_fraction * S^2))
```

The recommender can greedily choose actual levels from the distribution:

1. Start with the smallest relevant uncovered objects.
2. Choose a level that puts that cohort near `target_object_area_fraction` of slice area
   while satisfying the long-side constraint.
3. Tag every object admissible at that level as covered by that level, optionally allowing
   multi-level ownership when an object is valid at several scales.
4. Move to the next uncovered larger cohort and repeat until coverage is high enough or the
   remaining objects require whole-frame handling.

This mode reports both the scale list and the induced object bins: per-level object counts,
coverage percentage, size/shape ranges, objects below every minimum-area gate, objects above
every long-side gate, and the estimated mean adjacent width/height scale factor. With smooth
object-size distributions, a default adjacent linear ratio near `2.0x` is expected to be
conservative: `10%` target area versus `1%` minimum area implies a loose upper linear band
of `sqrt(10) ~= 3.16x`, but `2.0x` should give denser coverage and less abrupt
cross-level behavior.

## 2. On-the-fly datasets + custom ultralytics trainer (`sati.datasets`)

The storage tax ends: training reads slices lazily from the pyramid instead of from a
materialized tree.

- **`SliceDataset`** — indexable over `slices.parquet` rows (filtered to split); `__getitem__`
  reads the window from zarr, applies the emission transform (§3), and returns image +
  projected targets from `slice_annotations.parquet`. Deterministic per `(epoch_seed, index)`.
- **Ultralytics integration** — a thin `DetectionTrainer` subclass overriding
  `build_dataset`/`get_dataloader` to serve `SliceDataset` (and the §7 sampler). This is the
  *only* trainer surgery in v2: loss, val loop, and optimizer stay stock. The `Backend`
  protocol gains `train_streaming(dataset, cfg)` alongside the v1 materialized path, which
  **remains supported** — it is the fallback when ultralytics internals shift.
- **Multiprocessing rule:** zarr store handles must not cross `fork` — each dataloader
  worker opens its own store on first access (`worker_init_fn`), and per-worker chunk caches
  are bounded (`dataset.worker_cache_mb`).
- **Reproducibility:** on-the-fly ordering/augmentation is seeded; eval slice enumeration is
  always the deterministic `slices.parquet` order.

**Config:** `training.data_mode: materialized | streaming` (v2 default `streaming`),
`dataset.num_workers`, `dataset.worker_cache_mb`.

> [!NOTE]
> Undecided: how ultralytics' mosaic/augment pipeline composes with `SliceDataset` —
> subclassing their dataset inherits augmentations for free but couples to internals;
> a clean torch `Dataset` + our own light augmentation keeps control but loses their
> pipeline. Prototype both against the pinned ultralytics version before committing; the
> materialized fallback exists precisely so this decision can't block training.

> [!NOTE]
> Benchmark gate before making `streaming` the default: slices/sec from
> `SliceDataset` across N workers vs GPU consumption rate on a reference box. If zarr reads
> starve the GPU, the default stays `materialized` and streaming remains opt-in.

## 3. Native rectangular & whole-frame slices — stretch removed

v2 stops distorting. Per-axis tiling keeps v1's rules for axes `≥ nominal`, but an axis
`< nominal` now yields a **single native span** at its true length — no stretch, no pad. So
coarse-level slices become rectangular and the terminal level is fed as one whole-frame
input. Scale stays isotropic; stored pixels are never resampled at emission.

Manifest impact (why v1's schema was shaped the way it was): `emitted_wh` now equals
`slice_wh`, `stretched_axes` is always empty, and `affine_slice_to_level` collapses to
`sx = sy = 1` — same columns, new values. Nothing downstream changes shape.

**Backend preprocessing capture.** Off-square inputs mean the backend now letterboxes/
resizes to its `imgsz`, and SATI must invert that to get boxes back into slice pixels
before recombination. The backend adapter records, per slice batch:
`actual_slice_wh`, `backend_input_wh`, resize scale, pad offsets, letterbox color, stride
alignment — and exposes `invert(boxes) → slice pixels`. The v1 §7 coordinate tripwire test
grows a rectangular-slice case; expect ±1 px convention friction (half-open `xyxy` vs
ultralytics' scaling) and pin it down with golden tests.

**Short-side floor** (extreme aspect ratios, e.g. 11260×2280): optional `value` + `mode`:
- `stop` (default) — don't add a level whose short side would fall below the floor.
- `clamp` — add it, but clamp the short side at the floor while the long side keeps
  scaling; the realized per-axis scale lands in pyramid metadata like any other level.
  (The sole exception to "affines are isotropic scale within a level.")

**Config:** `slicing.emission: stretch | native` (v2 default `native`; `stretch` retained
for v1 parity and for export targets that demand one fixed shape — see the note below),
`slicing.short_side_floor.{value,mode}`.

> [!NOTE]
> **Export vs variable shapes — decision needed.** ONNX/TensorRT engines prefer one static
> input shape; native emission produces per-level shapes. Options: (a) export with dynamic
> spatial axes and accept engine-side cost, (b) export one engine per level shape (the
> level set is small and known from the sidecar), (c) deploy exported engines with
> `emission: stretch` while eager PyTorch uses `native`. Option (c) silently makes
> deployment's `preprocessing_hash` differ from eval's — which is exactly what that hash is
> for, so it *gates ensembling correctly* but means deployed accuracy ≠ evaluated accuracy.
> Lean: (b). Decide before v2 export ships; the inference-config sidecar schema must record
> whichever is chosen.

## 4. Resolution stack: explicit scale lists

Alongside v1's automatic factor mode:

- `slicing.scales: [1.0, 0.5, 0.25, 0.125]` — applied isotropically, used exactly as given.
  **Not** required to reach a whole-frame level: a coarsest scale that still tiles is
  intentional ("don't miss the forest for the trees"). Validation rejects only degenerate
  lists (empty, non-positive, or `> 1.0` scales), not lists that stop short.
- Exactly one of `scales` / `downsample_factor` may be set (`model_validator` enforces).
- Explicit target *resolutions* remain unsupported: levels always derive from source dims ×
  scale, preserving isotropy and avoiding aspect ambiguity.
- The dry-run tool (`sati prepare --dry-run`) reports per-level dimensions and slice counts
  for either mode.

## 5. Recombination strategies (full `Strategy` surface)

Stage boundaries, provenance stamping, and `recombination_hash` scoping are exactly v1
§6.2; v2 widens what each stage can do.

**Stage 1 (within-level)** gains SAHI-style postprocess options: `NMS`, `NMM`,
`GREEDYNMM`, optional `LSNMS` (extra), IoU + confidence thresholds, class-aware/agnostic —
implemented in `sati.tiling`, not imported from `sahi.postprocess` (our merge is
provenance-aware). Merge (as opposed to suppress) modes union boxes and record all
contributing `slice_id`s.

Stage 1 also computes **`near_edge_count`** on each winner: the number (0–4) of the winning
raw detection's own slice edges lying within `edge_proximity` of its box, in that slice —
a per-detection count, not a sum across slices. Stored on the `level_merge` row so Stage 2
consumes stable metadata.

**Stage 2 (cross-level)** clusters lifted boxes by IoU (class-aware by default) and selects
per cluster by **favorability**:

```text
favorability = confidence * scale_multiplier * edge_multiplier

scale_multiplier = sqrt(box_area_in_slice_pixels) / sqrt(slice_area_pixels)
edge_multiplier  = (1 - edge_favorability_penalty) ** near_edge_count
```

- `edge_favorability_penalty = 0.0` disables edge weighting (and with
  `scale_multiplier` forced to `1`, Stage 2 degrades exactly to v1's confidence NMS — the
  v1 strategy stays available as `cross_level: nms_confidence`).
- `edge_proximity` units: `0.0` = touching only; `(0,1)` = fraction of the relevant slice
  dimension; `≥ 1.0` = pixels. Multiple near edges compound multiplicatively.

**Config (new fragment group `recombination/`):** `within_level.method`, `within_level.iou`,
`cross_level.method` (`nms_confidence` | `favorability`), `cross_level.iou`,
`cross_level.edge_proximity`, `cross_level.edge_favorability_penalty`,
`confidence_threshold`, `class_agnostic`.

> [!NOTE]
> The `edge_proximity` unit trichotomy is easy to misconfigure silently (`1.0` means "1
> pixel", not "100% of the slice"). Consider splitting into `edge_proximity_frac` /
> `edge_proximity_px` (mutually exclusive fields) at schema level instead of overloading
> one float. Decide when writing the schema; the overloaded form is what SAHI users expect,
> the split form is what the strict-config philosophy prefers.

> [!NOTE]
> `scale_multiplier` as specified always favors detections that fill more of their slice —
> i.e. coarser levels — even when a fine-level detection is more precise. Whether it needs
> a configurable exponent (`scale_multiplier ** alpha`) or a cap should be settled
> empirically via §6 sweeps, not guessed now. Ship the formula as written, flagged as the
> research knob it is.

## 6. Post-hoc recombination sweeps

Because raw `slice_prediction` rows can be cached (v1 §4.5 flag), recombination parameters
can be swept **without re-running the model**:

`sati recombine --results <parquet> within_level.iou=0.4,0.6 cross_level.method=favorability …`

- Reads cached `slice_prediction` rows, re-runs Stages 1–2 per parameter combination,
  appends new `level_merge`/`final_merge` rows stamped with each combination's
  `recombination_hash` — this is precisely why that hash earns its own column while
  `config_hash` stays constant.
- Requires the cache flag to have been on; without it, only a re-run of inference can sweep.
- `sati eval --by recombination_hash` scores each variant; results pool in one table and
  group cleanly by hash.

## 7. Sampling controls

Two orthogonal knobs, both implemented in the §2 dataloader (streaming mode) or at tree
generation (materialized mode, by duplication/omission of files):

- **`training.per_level_weights`** — relative sampling weight per level (`auto` sentinel:
  inverse-frequency so each level contributes equally per epoch). Fixes coarse-level
  starvation: whole-frame slices are rare but hold the largest objects. Balances **scale**.
- **`training.negative_sample_ratio`** — target ratio of zero-target slices to positive
  slices per epoch (`null` = v1 behavior, use all). Balances **positive/negative**.

Materialized-mode caveat: baked-in duplication conflates sampling policy with dataset
identity — changing weights forces re-materialization, and the training tree's identity
shifts. This is contained because `dataset_hash` **gates metric comparison on eval sets**,
which carry no sampling weights; training-tree sampling must stay out of the eval
`dataset_hash` recipe. Streaming mode has no such coupling (weights live only in config →
`config_hash`), which is another reason it is the intended default.

Watch the per-scale eval breakdown (§9) to verify a starved level was actually fixed.

## 8. Parameter-sweep config generation

SATI owns sweep expansion (the Compose API was chosen over `@hydra.main` precisely so
multirun never runs the process):

- A **sweep spec** (YAML) names a base experiment plus override axes:

  ```yaml
  base: experiment/train_base
  axes:
    slicing.nominal_slice_size: [640, 1024]
    slicing.downsample_factor: [2, 4]
    recombination.cross_level.edge_favorability_penalty: [0.0, 0.2]
  ```

- `sati sweep <spec>` expands the cartesian product, calls `compose_and_resolve` per
  combination, and **writes N run directories each containing its `resolved.yaml`** — sweep
  generation is complete at that point and each run is independently `--resolved-config`
  re-runnable.
- Execution is decoupled from generation: `sati sweep --run` executes sequentially;
  anything smarter (parallelism, schedulers, resumption) is out of scope — the run dirs are
  plain artifacts any orchestrator can dispatch.
- Rows from all sweep members pool in `validation_results.parquet` and group by
  `config_hash` / `recombination_hash`; recombination-only axes should instead use §6
  sweeps, which skip model re-runs entirely.

> [!NOTE]
> Undecided: whether the sweep spec supports paired/zipped axes and exclusion rules
> (à la Hydra's sweeper plugins) or stays pure cartesian. Start cartesian; add pairing only
> when a real experiment needs it.

## 9. Per-scale evaluation semantics

v2 promotes v1's diagnostic breakdown to defined per-scale metrics, computing **both**
views (they answer different questions):

- **Prediction-attributed:** for each level *L*, mAP of `final_merge` predictions with
  `L ∈ source_levels` against **all** GT. Answers: *what does level L contribute?* (A
  prediction with multiple source levels counts for each — this view is not a partition.)
- **GT-size-binned:** GT boxes binned by which levels' size gate admits them (equivalently,
  by full-res size thresholds mapped through the stack); recall/mAP per bin against all
  predictions. Answers: *which object sizes are we missing?* This is the "starved level"
  detector, and reduces to COCO's small/medium/large convention when the bins are set to
  32²/96².

Both are computed by `sati.evaluation` from provenance columns already present — no schema
change.

## 10. Schema & manifest evolution

v2 changes manifest *values*, not shapes, by design (see §1, §3). Rules going forward:

- `schema_version` (table + sidecar, bumped together) is checked by every reader; readers
  accept `N` and `N−1`, refuse older, and never silently coerce.
- New columns land nullable-with-default so v1-era Parquet remains readable; column
  *removal or retyping* requires a major `schema_version` bump plus a `sati convert`
  migration path.
- Hash recipes are versioned inside the hash source (v1 §4.7); changing a recipe changes
  the hashes, which is correct — cross-version hash equality would be a lie. The sidecar's
  paired `*_source` objects let old rows be re-hashed under new recipes offline if a
  cross-version comparison is ever genuinely needed.

## 11. Further features — noted, not fleshed out

Directions with real pull but no design commitment yet; each needs its own design pass
before any schema fields appear (absent = unsupported):

- **Resolution-aware loss** (the original "later phase"): custom loss via `init_criterion`
  override on the §2 trainer subclass, threading per-object metadata (source level,
  truncation flags) through dataset/collate; reweight by level, down-weight edge-truncated
  targets, both, or neither. The §2 trainer is the deliberate landing pad. Expected and
  documented: per-slice loss and recombined-frame metrics will not track one-to-one.
- **Weighted Box Fusion / cross-level NMM** as Stage-2 strategies — box-averaging across
  levels rather than winner-take-all; needs a story for provenance of a fused box.
- **mmdetection backend** — documented stub only; mmdet is effectively unmaintained
  (2025–2026), so this stays a protocol-compliance exercise, never a migration plan.
  Recombined-full-frame validation is equally custom in any framework — intrinsic to SATI.
- **OME/NGFF interop** for the pyramid (§1's layout is deliberately minimal/purpose-built).
- **Remote stores** — zarr-on-S3 and remote Parquet via `sati.io`'s abstraction; mostly a
  matter of fsspec plumbing plus cache policy.
- **Source content hashing** — pixel-level `dataset_hash` strengthening (currently
  path+size by design).
- **Random/grouped split generation** — v1/v2 support user-provided splits only.
- **Segmentation / keypoints / OBB** — the affine machinery extends to polygons (shapely is
  already a dependency); the recombination research question gets harder.
- **Rectangular nominal slices** — `nominal_slice_wh` as a pair; every "square" assumption
  in §3–§5 is per-axis already, so this is mostly schema + validation work.
