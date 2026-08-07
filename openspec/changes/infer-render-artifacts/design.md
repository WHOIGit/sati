## Context

**Shipped today.** `src/sati` contains no rendering code. `grep` for
`ImageDraw`/`overlay`/`annotated` over the package returns nothing but unrelated
`annotated_field_paths` helpers. `cv2` is a core dependency but is used only for
decode (`io.py:358`), level resizing (`materialize.py:78`), and slice emission
(`materialize.py:202`). `InferOutputsConfig` (`config_schemas/root.py:265`) declares
two leaves — `results` (enabled) and `figures` (disabled) — and nothing writes to
`figures`. The 274 MB `37_U1553E_..._annotated.png` sitting in a run's `results/`
was produced by hand outside the repo; it is untracked and not reproducible.

**coconut-crab, as it exists** (`/home/sidney/Projects/SIBERT/coconut-crab`, commit
`09e604b`, `git@github.com:sbatchelder/coconut-crab.git`):

- `run_render(config: RenderConfig) -> int` (`render/engine.py:86`) is a library
  entry point, so SATI need not shell out to `coco-render`.
- `render_config_from_values(values: dict) -> RenderConfig` (`render/config.py:161`)
  builds a config from a plain dict, so SATI can compose one in memory and still let
  a user-supplied TOML provide the style.
- `RenderStyle` (`render/config.py:28`) carries `default`, `bbox`, `mask`,
  `opacity`, `line_weight`, `text_size`, `text_align`, and — critically —
  `label_colors` / `bbox_label_colors`, resolved by **casefolded category name**
  (`color_for`, `render/config.py:43`).
- `text` is a template over fields from `annotation_field_context`
  (`render/engine.py:496`), which falls through to raw annotation keys and an
  optional `meta` dict, case-insensitively.
- `Image.MAX_IMAGE_PIXELS = None` (`render/engine.py:19`) — the PIL
  decompression-bomb ceiling is already lifted, so a 308 MP frame loads.
- Inputs are a **COCO JSON file path** plus an `image_root`; images are matched by
  file identifiers (`image_identifiers`, `render/engine.py:233`).
- Requires Python ≥3.11 (SATI is ≥3.12) and depends on `pillow`, `pycocotools`,
  `tabulate`, `tqdm`. Not published to PyPI.

## Goals / Non-Goals

**Goals:**

- Every pipeline state a separate image, per level and stacked frame-wide, so a
  box's appearance or disappearance is attributable to a specific stage.
- Zero effect on any detection, metric, or hash. Rendering is strictly downstream
  observation.
- Style is owned by coconut-crab in its own config format; SATI owns geometry,
  canvas choice, layout, and naming.
- Rendering never becomes a hard requirement for `prepare`, `train`, or `infer`.

**Non-Goals:** per-slice overlays, ground-truth or comparison rendering,
segmentation masks, rendering during `train`, and any drawing primitive that
belongs upstream.

## Decisions

### D1 — The artifact set is one image per pipeline state

```
<infer output>/artifacts/<image-stem>/fullframes/
  raw-level-1.jpg          level-1 canvas   slice predictions ≥ confidence_threshold,
  raw-level-2.jpg          level-2 canvas   lifted to level coords, uncombined
  raw-level-3.jpg          level-3 canvas
  raw-levels-stacked.jpg   full-res canvas  ALL levels' uncombined boxes, by level
  level-1.jpg              level-1 canvas   level_merge winners for that level
  level-2.jpg              level-2 canvas
  level-3.jpg              level-3 canvas
  levels-stacked.jpg       full-res canvas  ALL levels' level_merge boxes, by level
  final.jpg                full-res canvas  final_merge boxes
```

`2N + 3` files; 9 for the reference frame's 3-level stack. Level indices come from
the frame's realized stack, so a stack starting below full-res is numbered from 1 and
there is no `level-0.jpg` — matching the existing index convention (`tiling.py:159`).

### D2 — Two stacked artifacts, symmetric with the per-level pair

The naming is a 2×2: `raw-` versus combined, per-level versus stacked. Both stacked
variants exist, which removes an ambiguity an earlier draft had to guess at (whether
a single `levels.jpg` meant raw-across-levels or combined-across-levels — it is now
both files, named for which).

```
   per level (level canvas)              stacked (full-res canvas)

   raw-level-N.jpg  ───────────────────▶ raw-levels-stacked.jpg
        │  Stage 1: within-level              │
        ▼                                     ▼
   level-N.jpg      ───────────────────▶ levels-stacked.jpg
                        lift + overlay        │
                                              ▼  Stage 2: cross-level
                                         final.jpg
```

Reading down the left column shows Stage 1 per level; reading down the right column
shows the same transition frame-wide; `levels-stacked` → `final` is Stage 2. Every
edge of the pipeline is now depicted on both sides.

Both stacked artifacts color **by contributing level**, not by class — locating a box
in the level stack is the question they answer, and the per-level and final artifacts
already color by class. Implemented with no upstream change by synthesizing COCO
categories named `level-1`, `level-2`, `level-3` and letting
`RenderStyle.label_colors` key off them (`render/config.py:43`).

### D3 — Canvases

| artifact | canvas | source |
| --- | --- | --- |
| `raw-level-N`, `level-N` | that level's realized dims | `build_level_pyramid` |
| `raw-levels-stacked`, `levels-stacked`, `final` | full-res frame dims | decoded source frame |

`build_level_pyramid` (`materialize.py:65`) is reused rather than reimplemented: it
already resizes each level from full-res directly with `INTER_AREA` and never
cascades, so a level canvas is pixel-identical to what the slicer saw. Boxes are
drawn in the coordinate space of their own canvas — level coords for `level-N`,
full-res for the stacked and final artifacts — so no artifact needs a scale factor
applied to geometry.

### D4 — `raw-level-N` is confidence-filtered

It shows slice predictions at or above `recombination.confidence_threshold`, lifted
through `affine_slice_to_level`. Filtered, not raw-raw, because the confidence
filter is the first thing Stage 1 does: the image is meant to depict *what the merge
stage saw*, and an unfiltered image would attribute the filter's removals to
merging. Showing what the filter discards is a separate artifact and a non-goal.

It does **not** require `results.cache_slice_predictions`. That flag controls whether
raw predictions are persisted to Parquet; they exist in memory during `infer`
regardless, and rendering consumes the in-memory values.

### D5 — Style lives in coconut-crab's config, reached by a `SatiPath`

SATI's config is strict, Hydra-composed, and hashed. coconut-crab's is TOML with its
own loader and its own schema. Rather than mirror one in the other:

- SATI owns **structure**: `enabled`, which artifact kinds, the directory layout,
  the image stem, level selection, the confidence filter. These decide *what exists*.
- coconut-crab owns **style**: colors, `line_weight`, `text_size`, `text_align`,
  `opacity`, the label template, and encoding options. These decide *how it looks*.
- SATI holds `rendering.style_config`, a `SatiPath` to a coconut-crab render TOML,
  and does not re-validate its contents — coconut-crab's `ConfigError` is the
  authority.

The `SatiPath` declaration is what makes this free: `hashing.py:159` already pops
every `SatiPath` field from every behavioral hash source, so a style change moves no
digest and no hash version needs bumping. This is correct rather than merely
convenient — restyling an overlay must not invalidate a result.

SATI overrides exactly three coconut-crab keys programmatically, because they are
structural: `coco_file`, `image_root`, and `output`. Everything else comes from the
TOML, or coconut-crab's own defaults when no TOML is configured.

### D6 — COCO is the interchange format, built in memory

For each artifact SATI synthesizes a COCO document from rows it already has:
`categories` are either the class map (`final`, `level-N`, `raw-level-N`) or
`level-K` names (`levels`, per D2); `annotations` carry `bbox` in that canvas's
coordinates plus `score`, so `{score}` is available to the label template with no
upstream change. `iscrowd` is 0 and `segmentation` is omitted.

Because `run_render` takes a `coco_file` path, the document is written to a
temporary file per artifact. Accepted for a first cut; upstream item 3 (in-memory
input) removes it.

### D7 — Optional extra, not a core dependency

`sati[render]`. coconut-crab pulls `pillow`, `pycocotools`, `tabulate`, and `tqdm`,
none of which any other SATI code path needs — `pycocotools` is today only in the
`synth` extra, and `pillow` is not a dependency at all. `prepare`, `train`, and
`infer`-without-rendering must keep working in an environment without them, so the
import is deferred to the point of use and a missing install produces a clear
configuration error rather than an `ImportError` at startup.

Because coconut-crab is not on PyPI, this needs a `flake.nix` input in addition to
the `pyproject.toml` entry, pinned to a revision like any other flake input.

### D8 — Enabled for `infer`, disabled for `eval`

`infer` runs over explicit ad-hoc imagery — usually one frame — where the whole
point is looking at it. `eval` runs over a prepared split; rendering a full pyramid
per frame there would silently produce hundreds of gigapixel JPEGs and dominate
runtime. `eval` can enable it explicitly. This mirrors `figures`, already
`enabled=True` for `train` and `enabled=False` for `eval`/`infer`
(`config_schemas/root.py:241`, `:260`, `:270`).

### D9 — Encoding: PIL's JPEG defaults, with one caveat

**Settled:** JPEG at PIL's default quality (75) is accepted; no quality field is
added, and no output long-side cap is added. `final.jpg`,
`raw-levels-stacked.jpg`, and `levels-stacked.jpg` are written at true full
resolution (19321×15948 = 308 MP on the reference frame), on the assumption that
inspection means zooming in. Label text is correspondingly allowed to be small
(D10).

Format alternatives were weighed and JPEG stands:

| format | 19321×15948 | note |
| --- | --- | --- |
| JPEG | works | smallest; 65535 px/axis ceiling is not close |
| PNG | works, ~274 MB | lossless and crisp, but slow; measured from the hand-made overlay in `runs/` |
| WebP | **fails** | hard 16383 px/axis limit — the frame is 19321 wide |
| Tiled pyramidal TIFF | works | the only option supporting partial decode, i.e. real zoom |

**The caveat, and the one encoding item worth upstreaming:** PIL applies 4:2:0
chroma subsampling below quality 95, which color-bleeds thin lines. A 3 px saturated
box edge on a dark background renders muddy and desaturated — degrading exactly the
feature these artifacts exist to show. `subsampling=0` (4:4:4) fixes it at modest
size cost and is a smaller change than a quality knob. Whether to pursue it is open
question 1.

If zooming proves to be the dominant workflow, a flat 308 MP JPEG is a poor vehicle
regardless of quality settings, because viewers decode the whole frame. Tiled
pyramidal TIFF is the domain-standard answer and the source data is already TIFF.
Recorded as a future direction, not proposed.

### D10 — Labels on, small; zooming is assumed

Text is enabled on every artifact with a small `text_size`, rather than suppressed on
the full-res canvases. The reader is assumed to zoom, so a label that is illegible at
fit-to-window is acceptable; one that does not exist is not. Dense clusters will
overlap at low zoom and resolve at high zoom. Scores are available with no upstream
change (D6).

### D11 — Upstream prerequisite in coconut-crab

One item, and it is a performance/quality issue rather than a hard blocker — the
artifact contract does not change when it lands.

1. **In-memory image input.** `run_render` takes a `coco_file` path plus an
   `image_root` and resolves images from disk, so SATI must write each base canvas
   out for coconut-crab to read back and re-save. For a 308 MP canvas that is two
   full encodes per artifact, and three artifacts are that size. A JPEG intermediate
   is additionally double-lossy; a PNG intermediate is ~274 MB each. An entry point
   accepting a PIL image or ndarray removes both.

Two items from an earlier draft were dropped by decision: configurable JPEG quality
(PIL's default accepted) and an output long-side cap (full resolution accepted).
Score labels never needed upstream work — `annotation_field_context`
(`render/engine.py:496`) already falls through to raw annotation keys
case-insensitively.

## Risks / Trade-offs

- **Render cost dominates a one-frame `infer`.** 9 artifacts, three of them 308 MP,
  each currently double-encoded (D11). Likely minutes against seconds for the actual
  inference. Mitigations: D8's `eval` default, per-kind enable flags, and the upstream
  in-memory entry point. Task 7.2 measures it before this is treated as acceptable.
- **The three full-res artifacts may be unopenable in ordinary viewers.** 308 MP
  needs roughly 900 MB for the RGB array, and most previewers decode the whole frame.
  Accepted deliberately (D9): full resolution is the point, and zooming is the assumed
  workflow. The exposure is that "zoom in" may not be practical in the user's actual
  viewer, in which case tiled pyramidal TIFF becomes the real requirement rather than
  a future note.
- **JPEG chroma subsampling degrades thin box edges** (D9). At PIL's default quality
  the 4:2:0 subsampling color-bleeds saturated lines on dark backgrounds. This is the
  single most likely way the artifacts under-deliver on their purpose while appearing
  to work. Open question 1.
- **Two config systems.** A user editing overlay colors edits a TOML that Hydra never
  sees and no hash covers. Correct boundary (D5), but a second place to look, and
  `sati inspect config` shows only the path, not the style.
- **Coupling to an unpublished sibling repo.** A coconut-crab change can break SATI
  rendering with no PyPI version to pin. Mitigated by pinning a flake revision and by
  rendering being an optional extra, so a break never blocks training or inference.
- **Level count drives file count** (`2N + 3`). Fine at N=3; worth a level allowlist
  if stacks grow.

## Migration Plan

1. Land the two hard upstream items (JPEG quality, long-side cap) in coconut-crab
   and tag a revision.
2. Add the flake input and `render` extra, pinned to that revision.
3. Add `RenderingConfig`, the `artifacts` output leaf, and `sati.render`; wire it
   into `infer` after recombination.
4. Enable it in `configs/experiment/v1/combo1b/*.yaml` so the
   `greedy-nmm-recombination` A/B/C comparison produces overlays for all three arms.
5. Upstream item 3 (in-memory input) lands after, replacing the temp-file path with
   no change to the artifact contract.

No existing artifact, hash, or schema version changes, so there is nothing to
migrate on the SATI side and no clean-break to coordinate.

## Open Questions

1. **Pursue `subsampling=0` upstream?** PIL's default 4:2:0 chroma subsampling
   color-bleeds thin saturated lines, which is precisely what these artifacts consist
   of (D9). It is a one-line upstream change and smaller than the quality knob that
   was dropped. Recommendation: **render one artifact both ways on the reference frame
   first** (task 7.3) and decide from the image, not from theory — if 3 px edges read
   cleanly at q75/4:2:0, this closes with no upstream work at all.
2. **Should the stacked artifacts distinguish boxes that survived Stage 2 from those
   absorbed by it?** Coloring by level answers "where did this come from". A variant
   coloring by survived/absorbed would answer "what did Stage 2 do", which is closer to
   the actual research question — and with `greedy_nmm` it is the only way to see a
   merge as opposed to a suppression. It needs per-box provenance from `final_merge`
   back to contributing `level_merge` rows; `source_levels` gives level granularity
   only. Deferred pending whether the nine artifacts prove sufficient.
3. **Is `<image-stem>` collision-safe?** Two input frames with the same stem in
   different source directories collide inside one run's `artifacts/`, which `infer`
   over a directory makes reachable. Options: accept and document, or fall back to
   `fullframe_id`, which is unique but far less readable.
