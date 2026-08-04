# SATI Synth — Procedural Synthetic Dataset Builder

A standalone tool that generates very large (100+ megapixel) synthetic images populated
with class-recognizable procedural objects at **all scale levels** — almost-fullframe,
large, medium, small, and tiny — plus bbox/mask/polygon sidecars and rich diagnostic
metadata.

**Primary purpose:** a *diagnostic stress suite* for SATI, not merely synthetic training
data. Every generated object carries enough metadata (scale band, difficulty, tile-boundary
condition, grammar parameters, seed) that SATI's evaluation can slice mAP by *why* a
detection should have been hard.

In one sentence: **build a procedural object generator with compositional class grammars,
stamp those objects onto multi-scale procedural backgrounds, force a balanced log-scale
size distribution, and export detailed metadata so results can be evaluated by scale,
difficulty, and tile-boundary condition.**

---

## 1. Relationship to SATI

- Lives at `src/synth_dataset_builder/` as its own package, independent of `sati.*`
  imports. It must be runnable without SATI installed.
- Installed as an **optional dependency group**: `pip install sati[synth]` → console
  script `sati-synth`.
- Output is consumed by SATI through its **COCO import adapter** (`sati prepare` /
  `sati convert`, v1 §4) — the synth tool never writes SATI-internal parquet manifests.
  `sati prepare` assigns the COCO export a logical `dataset_hash` and writes a reusable,
  content-addressed prepared product under `prepare_outputs.dir`; synth does not predict or
  author those SATI identities.
  Diagnostic metadata rides alongside in its own parquet (§7.2) keyed by
  `(image_id, ann_id)` so SATI eval can join it to `final_merge` results post-hoc.
- Coordinate conventions match SATI v1 §4: top-left origin, half-open `xyxy` internally;
  COCO `xywh` only in the exported JSON.

**Naming (settled 2026-07-02):** Python import names cannot contain dashes, so the package
directory is `synth_dataset_builder`; the *console script* keeps the dash (`sati-synth`).

**Packaging (settled 2026-07-02):** one shared `pyproject.toml` at repo root with
`[project.optional-dependencies] synth = [...]` (only `scikit-image` and `pycocotools`
beyond SATI core deps). If SATI later ships as its own distribution, the synth tool can be
promoted to a sibling distribution with no code change (it has no `sati` imports).

---

## 2. Design Principles

1. **Determinism.** One top-level `seed` in config; every image, object, and background
   derives its own independent stream via `numpy.random.SeedSequence.spawn()`. Same config
   + same seed ⇒ byte-identical annotations and (up to library versions) identical pixels.
   Every object's spawn key is recorded in metadata, so any single object can be
   re-rendered in isolation for debugging.
2. **Grammar over sprites.** Classes are defined by *structural rules* with randomized
   parameters, not by fixed bitmaps — so intra-class variation is unbounded and hard
   negatives ("almost a ring") can be produced from the same grammar with a rule violated.
3. **Vector-first, supersampled rendering.** Objects are authored in a continuous local
   coordinate frame, rendered onto a small local canvas at ≥4× the target size, then
   area-downsampled to target. This is what makes a 3 px object *class-recognizable
   in principle* (subpixel-accurate antialiased structure) instead of a blurry dot.
4. **Truth from geometry, not pixels.** Bboxes and polygons come from the object's vector
   geometry / pre-composite alpha mask — never re-derived from the composited image, where
   background blending would corrupt them.
5. **Deliberate coverage, never uniform sampling.** Scale bands, layout types, difficulty
   knobs, and tile-boundary conditions are all *quota-driven* (§8), and realized quotas are
   reported so a generated dataset is auditable.
6. **Config mirrors SATI conventions.** Pydantic schemas as the contract, strict
   `extra="forbid"`, absent = unsupported, authored + resolved YAML written into the output
   dataset root.

---

## 3. Pipeline Overview

```
config.yaml ──▶ resolve ──▶ scene plan (per image) ──▶ render ──▶ export
                              │                          │
                              │  scale-quota sampler     │  background synth (full canvas)
                              │  layout sampler          │  object rasterizer (local canvas)
                              │  class/grammar sampler   │  compositor (+ occlusion order)
                              │  boundary-condition      │  degradations (stage 3)
                              │    placement             │
                              ▼                          ▼
                        scene_plan rows            images/*.png (or tiled TIFF)
                                                   annotations: COCO JSON
                                                   diagnostics: objects.parquet
                                                   report: dataset_report.json + contact sheets
```

Two clean phases per image, separated by an explicit intermediate:

- **Plan** — pure logic, no pixels: decide image size, background recipe, object list
  (class, grammar params, position, size, rotation, z-order, difficulty knobs). The plan is
  a serializable dataclass list; `--dry-run` stops here and prints realized quotas.
- **Render** — pure execution of the plan. Deterministic given the plan. 

This split is what makes the tool a *diagnostic* instrument: the plan **is** the ground
truth, and rendering is just its materialization.

---

## 4. Object Grammar System (`grammar/`)

### 4.1 Primitives

Small vocabulary of continuous-coordinate primitives, each returning a
`shapely` geometry (or a signed-distance / scalar-field description for the blobby ones):

| primitive | parameters | backing |
|---|---|---|
| `stroke` | polyline pts, width profile (constant / tapered) | shapely `LineString.buffer` |
| `ellipse` / `capsule` | center, axes, angle | shapely |
| `ring` | center, radius, thickness, arc span, gap positions | buffered arc |
| `polygon_glyph` | n vertices, irregularity, spikiness | analytic |
| `blob` | base radius, Perlin/harmonic boundary perturbation (freq, amp) | radial function → polygon |
| `metaball_field` | k centers, radii, threshold | scalar field → `skimage.measure.find_contours` |
| `dots` | count, radius distribution, placement region | circles |
| `fractal_branch` | depth, branch angle dist, decay, jitter | recursive strokes (L-system-lite) |

### 4.2 Operators

Grammar operators compose primitives into class instances:

`repeat_along(path, element, n, jitter)` · `radial_arrange(center, element, n, r)` ·
`attach(parent, child, anchor, angle)` · `union` / `difference` (shapely) ·
`taper(element, profile)` · `perturb_boundary(geom, noise_params)` ·
`texture_fill(geom, texture_fn)` (raster-stage fill: cellular/Worley, stripes, gradient).

### 4.3 Class definitions

Each class is a Python object registered in a **class registry**:

```python
@register_class("ladder")
class Ladder(ObjectClass):
    structural_rule = "long spine; 3–9 crossbars roughly perpendicular; optional terminal caps"
    def sample_params(self, rng) -> LadderParams: ...
    def build_geometry(self, params) -> ObjectGeometry: ...   # shapely + field parts
    def hard_negative(self, rng) -> ObjectGeometry: ...       # rule deliberately violated
```

- `sample_params` draws from per-class distributions (declared as Pydantic fields with
  ranges, so config can override any of them under `classes.<name>.params.*`).
- `build_geometry` is deterministic given params.
- `structural_rule` is documentation *and* the contract: a class is "recognizable" iff its
  rule holds at the rendered size. Rules are what hard negatives violate (§6.3).
- Every class must declare `min_recognizable_px` — the smallest long-side in pixels at
  which its rule survives rasterization (e.g. `ring: 4`, `ladder: 12`, `gear: 16`). The
  scale sampler (§8) only assigns a class to a band it can survive; the realized
  class×band matrix is part of the dataset report.

### 4.4 The 18 starter classes

| family | classes | structural rule sketch |
|---|---|---|
| **geometric** | `ring` | closed loop, 0 gaps, thickness 5–20% of radius |
| | `triangle_glyph` | 3-vertex closed glyph, optional inner dot |
| | `cross_glyph` | two strokes crossing 60–120°, arm ratio 0.6–1.0 |
| | `spiral` | 1.5–4 turns, monotonic radius growth |
| | `starburst` | 5–16 radial strokes from common center |
| **organic** | `smooth_blob` | low-freq boundary perturbation, convexity > 0.85 |
| | `spiky_blob` | high-freq boundary spikes, 8–24 protrusions |
| | `segmented_worm` | 4–12 capsules chained along a smooth path |
| | `branching_organism` | trunk + recursive branches, metaball-joined |
| | `clustered_cells` | 3–20 blobs, Worley interior texture, faint connecting lines |
| **texture / fractal** | `crystal_patch` | angular Voronoi facets, shared straight edges |
| | `fern_branch` | self-similar leaflets along a rachis, 2 recursion levels |
| | `lightning_branch` | jittered polyline, 2–5 forks, tapering width |
| | `nested_contour` | 3–6 concentric non-circular closed contours |
| **compositional** | `radial` | central blob, 5–12 appendages, optional inner dots, optional outline |
| | `ladder` | long spine, repeated crossbars, optional terminal caps |
| | `capsule_with_dots` | capsule body, 2–8 interior dots in a row |
| | `fork` | trunk, 2–5 branches, tapered ends |
| | `shell` | oval body, concentric rings, one asymmetric notch |
| | `cluster` | 3–20 small blobs connected by faint lines |
| | `gear` | ring + 6–18 regular teeth, optional hub hole |

(That's 21 — `shell`, `cluster`, `gear` included from the compositional brief; trim or
extend freely, the registry makes the set open.)

> [!NOTE]
> Class-pair confusability is a feature (`starburst` vs `radial`, `fork` vs
> `lightning_branch`, `cluster` vs `clustered_cells`). Decide whether the v0 class list
> should *maximize* separability (cleaner MVP debugging) or keep the confusable pairs
> (better stress suite). Recommendation: keep pairs but put them in a config-selectable
> `class_set` (`easy8` / `full21`) so both modes exist.

### 4.5 Appearance

Geometry is class identity; appearance is nuisance variation. Per-object sampled:
fill/stroke color (from configurable palettes with a contrast-vs-background control, §6),
interior texture (`flat`, `gradient`, `cellular`, `striped`), outline on/off, opacity
90–100%, rotation, mild anisotropic scale (bounded so it can't destroy the structural rule).

---

## 5. Rendering (`render.py`)

1. **Local canvas.** Each object renders into its own RGBA float32 canvas at
   `supersample × target_size` (supersample = 4 default, automatically raised to keep the
   *supersampled* long side ≥ 64 px for tiny objects — a 3 px object renders at 64+ px).
2. **Rasterization.** Shapely geometry → `cv2.fillPoly`/`polylines` on the supersampled
   grid; scalar-field parts (metaballs, cellular textures) evaluated directly on the grid.
3. **Downsample** with `cv2.resize(..., INTER_AREA)` to target size. The alpha channel
   downsamples identically — it becomes the soft mask.
4. **Truth extraction** happens at the supersampled stage: polygon = simplified contour of
   the binary (α > 0.5) supersampled mask (`skimage.measure.find_contours` +
   `shapely.simplify` with tolerance = 0.25 px at target scale); bbox = polygon bounds.
   For objects with target long-side < 8 px, the polygon degenerates gracefully — export
   RLE instead (§7.1).
5. **Composite** onto the background in z-order with alpha blending. Occluded-object truth
   is *not* re-clipped: full (amodal) polygon + visible-fraction estimate both exported
   (visible fraction from the pre/post occlusion alpha overlap).
6. **Full-canvas memory:** a 100 MP RGB float32 canvas is ~1.2 GB; render the *background*
   in uint8 and composite objects into it region-by-region (objects touch only their own
   patch), so peak memory stays ≈ one uint8 canvas + one local object canvas.

> [!NOTE]
> Output image format for 100+ MP frames: PNG is lossless but slow and huge; tiled TIFF
> (via `tifffile`) is what SATI-with-zarr will want to ingest in v2. Recommendation: PNG
> for stage 1 (simple, universally readable), add `output.format: tiff` with tiling in
> stage 2, and record the choice in the dataset report.

---

## 6. Backgrounds (`backgrounds.py`)

Backgrounds are first-class difficulty controls, generated per-image from a sampled
recipe. All are procedural (no photo assets).

### 6.1 Smooth naturalistic
Fractal Brownian motion (octaved value/Perlin noise via vectorized numpy — see §10),
cloud textures, large soft gradients, optional low-amplitude banding. **Spectral content
must span SATI's pyramid**: compose noise octaves so structure exists at full-res *and*
survives 2×/4×/8× downsampling — otherwise low-res levels are trivially flat and the
stress suite under-tests them.

### 6.2 Structured clutter
Non-class geometry scattered densely: random strokes, arcs, speckle fields, grid/ruling
patterns, Voronoi cracks. Clutter elements are drawn from the *same primitive vocabulary*
as objects (§4.1) but composed without any class's structural rule.

### 6.3 Hard negatives
Grammar-derived distractors: each class's `hard_negative()` produces near-miss instances —
broken rings (arc gap > gap threshold), ladders with one crossbar, radial with 2
appendages, spirals that don't grow monotonically. Placed like objects but **exported as
background** (no annotation) — with one exception: they *are* recorded in
`objects.parquet` with `role="hard_negative"` so false positives on them can be counted
specifically.

### 6.4 Contrast control
Per-object `contrast_target` ∈ [0,1] adjusts the object palette against the local
background patch mean (measured pre-composite). Low-contrast quota is a difficulty knob
(§8.3); realized local contrast (RMS, object-vs-annulus) is measured post-composite and
recorded per object.

---

## 7. Outputs & Sidecars

Dataset root layout:

```
<out_root>/
  resolved_config.yaml
  dataset_report.json          # realized quotas, class×band matrix, warnings
  images/{training,validation,holdout}/<image_id>.png
  annotations/
    instances_training.json    # COCO
    instances_validation.json
    instances_holdout.json
  diagnostics/
    objects.parquet            # per-object diagnostic ledger (incl. hard negatives)
    images.parquet             # per-image recipe summary
  contact_sheets/              # stage 3: per-class × per-band visual grids
```

### 7.1 COCO annotations (`annotations/`)

Standard COCO instances: `bbox` (xywh), `segmentation` as polygon lists — except objects
with long-side < 8 px or fragmented visible masks, which use RLE (`pycocotools`). `area`
from the mask, `iscrowd=0`. Category ids stable across the whole dataset (declared once in
config, not inferred). This is the file `sati prepare` ingests; splits map to SATI's
`training`/`validation`/`holdout` via one COCO file per split (SATI v1 §4.6).

### 7.2 `diagnostics/objects.parquet`

One row per placed object **and** per hard negative:

| column group | columns |
|---|---|
| keys | `image_id`, `ann_id` (null for hard negatives), `role` (`object` \| `hard_negative`) |
| class | `class_name`, `class_id`, `grammar_seed`, `params_json` |
| geometry | `bbox_xyxy` (fullres, half-open — SATI convention), `long_side_px`, `area_px`, `polygon_n_pts` |
| scale | `scale_band`, `tiny_sub_band` (null unless tiny) |
| difficulty | `contrast_target`, `contrast_realized`, `occlusion_frac`, `blur_applied`, `layout_type` |
| tile-boundary | `straddle_conditions` — list of structs `{nominal, overlap, level, n_boundaries_crossed, min_center_dist_to_boundary_px}` computed for each configured probe grid (§9) |
| provenance | `spawn_key`, `z_order` |

`(image_id, ann_id)` joins 1:1 to COCO, and — after SATI eval matches predictions to GT —
to SATI `final_merge` rows, giving mAP by band / sub-band / difficulty / straddle condition
with a single parquet join. `params_json` + `spawn_key` make any object exactly
re-renderable.

### 7.3 `dataset_report.json`
Realized vs requested quotas per scale band, sub-band, class, layout type, and straddle
condition; class×band coverage matrix; per-class contrast histograms; generation warnings
(e.g. "gear excluded from tiny band by min_recognizable_px"). `sati-synth verify` (stage 3)
re-derives this from the sidecars and diffs it.

---

## 8. Scale, Layout & Difficulty Strategy (`scale.py`, `layout.py`)

### 8.1 Scale bands — quota-driven, log-spaced

Band membership by **long side in full-res pixels** (matches how detection difficulty
actually scales; area double-counts elongated classes like `ladder`):

| band | long side (px, at 12288×8192 reference) | default share |
|---|---|---|
| almost-fullframe | ≥ 60% of image short side | 5% |
| large | 512 – 60% short side | 10% |
| medium | 128 – 512 | 20% |
| small | 41 – 128 | 40% |
| tiny | 3 – 40 | 25% |

Tiny sub-bands (each with its own quota within tiny's 25%): **3–5 px, 6–10 px, 11–20 px,
21–40 px** — default equal quarters. All thresholds and shares are config
(`scale.bands`, `scale.tiny_sub_bands`); within a band, sizes sample **log-uniformly**.

The sampler is a **quota filler**, not a per-object coin flip: for a requested total object
count it computes per-band integer targets, then per (band × class) targets subject to
`min_recognizable_px`, then shuffles. Realized counts go in the report.

### 8.2 Layout types

Sampled per image from a configurable mixture:

| layout | mechanics |
|---|---|
| `sparse` | Poisson-disk placement, large min-distance, near-zero occlusion |
| `dense` | high count, small min-distance, occlusion allowed up to cap |
| `scale_mixed` | almost-fullframe object(s) with small/tiny objects scattered *on top of* them |
| `clustered` | Thomas cluster process: parent points → Gaussian child clouds; tiny objects cluster hardest |
| `occlusion` | deliberate pairwise overlaps with target occlusion fractions (10–60%) |

Occlusion is realized by z-order compositing; per-object visible fraction is measured and
recorded, and objects with visible fraction < a configurable floor (default 0.15) are
demoted to `iscrowd`-free "still annotated" status but flagged — SATI eval can then choose
to score with or without them.

### 8.3 Difficulty axes (all quota-controlled, all recorded)

contrast (low-contrast share), clutter density of the background recipe, occlusion share,
tiny-band share, hard-negative density, and (stage 3) degradations — Gaussian blur bands,
sensor noise, JPEG artifacts, illumination gradients across the frame.

---

## 9. Tile-Boundary Conditioning — the SATI-specific stress axis

SATI slices with a nominal size (default 1024) and overlap ratio (default 0.10) over a
downsample pyramid. The synth tool exploits knowing this in advance:

- Config declares one or more **probe grids**: `{nominal, overlap_ratio, downsample_factor,
  n_levels}` — normally copied from the SATI config under test.
- The layout stage supports a `boundary_placement` quota: a configurable share of objects
  (default 20%) is deliberately positioned relative to probe-grid slice boundaries —
  centered on a boundary, straddling a boundary at 25/50/75% of body, inside the
  overlap strip of two slices, at a four-corner junction, or (control group) certified
  ≥ margin away from every boundary at every level.
- Every object (placed deliberately or not) gets its `straddle_conditions` computed
  analytically against every probe grid and stored (§7.2).

This turns "does recombination lose objects cut by tile seams?" from anecdote into a
first-class, quota-balanced measurable.

> [!NOTE]
> Probe grids assume SATI's v1 slicing arithmetic (§6.1 of DESIGN-DOC-v1: stretch rule,
> last-tile clamp). The straddle computation must replicate that arithmetic exactly —
> decide whether to (a) copy the ~30 lines of grid math into the synth tool with a
> cross-test fixture shared between repos, or (b) accept an optional import of
> `sati.tiling` when SATI is installed, falling back to the copy. Recommendation: (a) —
> keeps standalone-ness honest; the shared fixture guards drift.

---

## 10. Libraries

| library | role | stage |
|---|---|---|
| `numpy` | everything; `SeedSequence` for determinism | 1 |
| `opencv-python-headless` | rasterization (`fillPoly`, `polylines`), `INTER_AREA` resize, blur/degradations | 1 |
| `shapely` (≥2.0) | grammar geometry: buffer, union, difference, simplify, validity | 1 |
| `scikit-image` | `measure.find_contours` (fields→polygons), morphology, Worley/cellular helpers | 1 |
| `pydantic` (v2) + `PyYAML` | config contract (mirrors SATI conventions) | 1 |
| `typer` + `rich` | CLI + progress/reporting | 1 |
| `pyarrow` | diagnostics parquet (explicit schema, per SATI house style) | 1 |
| `pycocotools` | RLE encoding for tiny/fragmented masks | 1 |
| `scipy` | distance transforms (metaballs), Poisson-disk acceleration (`cKDTree`) | 2 |
| `opensimplex` *or* vectorized numpy fBm | smooth noise fields | 1 (numpy fBm), 2 (opensimplex if speed demands) |
| `tifffile` | tiled TIFF output for 100+ MP frames | 2 |
| `matplotlib` | contact sheets only | 3 |

Deliberately avoided: Pillow for drawing (OpenCV covers it), `noise` (C Perlin lib,
packaging pain — numpy fBm is fast enough and dependency-free), any game/graphics engine.

---

## 11. Code Structure (`src/synth_dataset_builder/`)

```
synth_dataset_builder/
  __init__.py
  cli.py               # typer app: generate, plan (--dry-run), verify, preview
  config.py            # pydantic schemas + YAML load/resolve; writes resolved_config.yaml
  rng.py               # SeedSequence tree helpers; spawn_key bookkeeping
  planner.py           # phase 1: scene plans (pure, no pixels)
  scale.py             # band definitions, quota filler, log-uniform sizing
  layout.py            # placement processes incl. boundary_placement
  probe_grid.py        # SATI slicing arithmetic replica + straddle computation
  grammar/
    base.py            # ObjectClass ABC, registry, ObjectGeometry container
    primitives.py      # §4.1 vocabulary
    operators.py       # §4.2 combinators
    classes/           # one module per family: geometric.py, organic.py,
                       #   fractal.py, compositional.py
  backgrounds.py       # fBm/clouds, clutter, hard-negative placement, recipes
  render.py            # phase 2: local-canvas rasterize, downsample, composite
  truth.py             # mask→polygon/RLE, bbox, visible-fraction, contrast measure
  export/
    coco.py            # per-split instances JSON
    diagnostics.py     # objects.parquet / images.parquet (explicit pyarrow schema)
    report.py          # dataset_report.json; verify re-derivation
    contact_sheet.py   # stage 3
```

Structural rules for the code:

- **planner never imports cv2; render never makes random draws.** All randomness is
  consumed in the plan phase; render is a pure function of `(plan, config)`. This is the
  invariant that makes `verify` and single-object re-rendering possible.
- Class registry is entry-point-free (plain decorator + import in `classes/__init__.py`);
  config selects classes by name and unknown names are a validation error.
- Parquet writers use explicit hand-written schemas, column-by-column, per SATI house
  style (v1 §4.5) — no Arrow inference.
- `probe_grid.py` carries the cross-test fixture against SATI's tiling tests (§9 NOTE).
- CLI: `sati-synth generate <config.yaml> [-o out_root] [overrides…]` ·
  `sati-synth plan <config.yaml> --dry-run` (quota tables, no pixels) ·
  `sati-synth verify <out_root>` (sidecars ↔ report consistency) ·
  `sati-synth preview --class ring --band tiny` (render a param grid to one PNG).

---

## 12. Staging

### Stage 1 — MVP (prove the loop end-to-end)

- 5 geometric classes (`ring`, `triangle_glyph`, `cross_glyph`, `spiral`, `starburst`).
- One background family: numpy fBm smooth naturalistic.
- Quota scale sampler with all 5 bands + tiny sub-bands; `sparse` and `dense` layouts only;
  no occlusion, no hard negatives.
- Boundary-condition *measurement* (straddle columns computed for one probe grid) but no
  deliberate boundary placement yet.
- Full export path: PNG images, COCO per split, `objects.parquet`, `dataset_report.json`,
  resolved config. Deterministic seeds. `generate` + `plan --dry-run` CLI.
- **Exit test:** `sati prepare` ingests the COCO output; a stock YOLO trains on it; per-band
  join against `objects.parquet` produces a scale-binned mAP table.

### Stage 2 — full grammar & difficulty

- Organic + fractal + compositional families (full class set), `hard_negative()` per class.
- Structured clutter and hard-negative backgrounds; contrast control with realized-contrast
  measurement.
- `clustered`, `occlusion`, `scale_mixed` layouts with visible-fraction accounting.
- Deliberate `boundary_placement` quotas against multiple probe grids.
- Tiled TIFF output option for 100+ MP frames.

### Stage 3 — diagnostic instrumentation

- Degradations: blur bands, sensor noise, JPEG quality, illumination gradients — all
  quota'd and recorded per image/object.
- `verify` command; contact sheets (class × band grids, hard-negative sheets).
- **Named suites**: curated configs shipped in-package —
  `suite_scale_ladder` (single class, every band, clean background: isolates pure scale),
  `suite_boundary_probe` (boundary quotas maxed: isolates recombination),
  `suite_confusables` (near classes + hard negatives: isolates classification),
  `suite_tiny_swarm` (tiny sub-bands dense/clustered: isolates small-object recall),
  `suite_kitchen_sink` (everything: regression suite).
- A short doc section (or notebook) showing the SATI-results ⋈ objects.parquet join recipe.

### Stage 4 — ideas not yet fleshed out

- **Scale-continuum probes:** the *same object instance* rendered at 20 sizes across bands
  in otherwise-identical scenes — response curves per class instead of band averages.
- Label-noise injection modes (jittered/dropped/wrong-class GT) for robustness studies.
- Sequence mode: slowly drifting scenes for tracking-ish or temporal-consistency tests.
- Domain-randomized color/texture themes to test train-on-theme-A/eval-on-theme-B transfer.
- Overlap-aware *annotation* of hard negatives (score FP rates per negative type).
- GPU rasterization path if generation throughput becomes the bottleneck.

---

## 13. Config Sketch (authored YAML)

```yaml
seed: 1234
output: { root: ./synth_out, format: png, splits: {training: 200, validation: 40, holdout: 40} }
image: { width: 12288, height: 8192, color_mode: RGB }
classes:
  set: full21                # or easy8, or explicit list
  overrides: { ring: { params: { thickness_frac: [0.05, 0.20] } } }
scale:
  bands: { almost_fullframe: 0.05, large: 0.10, medium: 0.20, small: 0.40, tiny: 0.25 }
  tiny_sub_bands: { "3-5": 0.25, "6-10": 0.25, "11-20": 0.25, "21-40": 0.25 }
objects_per_image: [40, 120]  # sampled per image
layout: { mixture: { sparse: 0.3, dense: 0.3, clustered: 0.2, scale_mixed: 0.1, occlusion: 0.1 } }
background: { mixture: { smooth: 0.5, clutter: 0.3, hard_negative: 0.2 } }
difficulty: { low_contrast_share: 0.2, hard_negative_density: 0.05 }
probe_grids: [ { nominal: 1024, overlap_ratio: 0.10, downsample_factor: 2, n_levels: 4 } ]
boundary_placement: { share: 0.20, conditions: [center_on_boundary, straddle_50, overlap_strip, corner_junction, certified_clear] }
```

---

## 14. Pitfalls & Mitigations

| pitfall | mitigation |
|---|---|
| Tiny objects unrecognizable after downsample | supersampled local canvas + `min_recognizable_px` gate per class; `preview` command to eyeball each class×band cell |
| Truth polygons from composited pixels | truth extracted from pre-composite supersampled alpha only (§5.4) |
| Shapely invalid geometries from grammar ops | `make_valid` + buffer(0) at every union/difference; reject-and-resample on failure with a report counter |
| Quota starvation (e.g. only 2 classes fit tiny 3–5 px) | quota filler resolves class×band feasibility first; report warns on any empty cell |
| Boundary math drifting from SATI's actual slicer | shared cross-test fixture on `probe_grid.py` (§9) |
| 100 MP memory blowups in workers | uint8 background canvas + patch-local compositing; one image per process |
| Nondeterminism from thread pools / cv2 | `cv2.setNumThreads(1)` in workers; all RNG through the SeedSequence tree |
| Contrast target vs realized divergence on busy clutter | measure realized contrast post-composite and record both; quota audits use realized |
