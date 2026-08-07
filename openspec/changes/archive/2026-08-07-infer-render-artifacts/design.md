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

Because `load_coco` takes a `coco_file` path, the document is written to a temporary
file per artifact. That stayed — it is kilobytes, and it is the *only* thing that
round-trips through disk. The **base canvas does not**: see D11, which turned out to
need no upstream change at all.

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
added, and no output long-side cap is added. **Chroma subsampling is the one
exception, and it is not a quality setting:** encoding writes 4:4:4 rather than PIL's
default 4:2:0, fixed upstream rather than configured (open question 1). `final.jpg`,
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

### D11 — Upstream prerequisite in coconut-crab: **none, on inspection**

**Resolved during implementation: the in-memory entry point already exists.** The
premise of this item was that `run_render` is the only way in, so SATI would have to
write each base canvas out for coconut-crab to read back and re-save — two full
encodes per artifact on three 308 MP canvases, double-lossy through JPEG or ~274 MB
through PNG.

But `run_render` is a thin loop. Its per-image body is:

```python
image_path = resolve_image_path(image, config.image_root)
with Image.open(image_path) as src:
    base = src.convert("RGBA")
render_annotations_on_image(base, dataset, image, config)   # <- takes a PIL image
save_rendered_image(base, out_path)
```

`render_annotations_on_image` (`render/engine.py:270`) is already an in-memory entry
point, and it is module-level and public. `sati.render` calls it and
`save_rendered_image` directly with `Image.fromarray(canvas)`, so no base canvas ever
touches disk and nothing is encoded twice. The cost this item existed to remove does
not exist.

What SATI gives up by bypassing `run_render`: image selection (`include`/`exclude`),
the `clobber`/`dry_run` checks, the output-path template, and the progress bar. All
four are SATI's own concern here — it renders exactly one image per call to a path it
chose — so none is a loss. `render_config_from_values` is still used, so style, the
label template, and encoding come from coconut-crab exactly as D5 requires.

Two items from an earlier draft were dropped by decision: configurable JPEG quality
(PIL's default accepted) and an output long-side cap (full resolution accepted).
Score labels never needed upstream work — `annotation_field_context`
(`render/engine.py:496`) already falls through to raw annotation keys
case-insensitively.

That left `subsampling=0` (open question 1) as the only real upstream item, and it
landed: rev `6356462`.

## Risks / Trade-offs

- ~~**Render cost dominates a one-frame `infer`.**~~ **Measured, and it does not.** On
  the reference frame (19321×15948, 3 levels, 9 artifacts): **19 s** wall with
  rendering, **9 s** without — so rendering costs ~10 s against 9 s of inference, not
  the predicted minutes. Two reasons the prediction was pessimistic: the double encode
  never happened (D11), and `combo1b`'s `start_scale: 0.25` means only ~35 slices, so
  "seconds for the actual inference" was itself an overestimate. Total output 173 MB:
  54.7 / 54.6 / 54.5 MB for the three full-res artifacts, 4.1 MB each at level 1,
  0.4 MB at level 2, 0.1 MB at level 3. `eval`'s default (D8) still matters, because
  173 MB and 10 s *per frame* across a whole split is a different proposition.
- **The three full-res artifacts may be unopenable in ordinary viewers.** 308 MP
  needs roughly 900 MB for the RGB array, and most previewers decode the whole frame.
  Accepted deliberately (D9): full resolution is the point, and zooming is the assumed
  workflow. The exposure is that "zoom in" may not be practical in the user's actual
  viewer, in which case tiled pyramidal TIFF becomes the real requirement rather than
  a future note.
- ~~**JPEG chroma subsampling degrades thin box edges**~~ (D9). **Confirmed, then
  fixed upstream.** Default 4:2:0 carried 4.3× the line-pixel error of 4:4:4 and cost
  40% of edge saturation on the real frame; `quality=95` did not help.
  `save_rendered_image` now writes 4:4:4 (coconut-crab `6356462`), for +14% bytes. This
  was named as "the single most likely way the artifacts under-deliver while appearing
  to work" — it was real, and it is closed. See open question 1.
- **Two config systems.** A user editing overlay colors edits a TOML that Hydra never
  sees and no hash covers. Correct boundary (D5), but a second place to look, and
  `sati inspect config` shows only the path, not the style.
- **Coupling to an unpublished sibling repo.** A coconut-crab change can break SATI
  rendering with no PyPI version to pin. Mitigated by pinning a flake revision and by
  rendering being an optional extra, so a break never blocks training or inference.
- **Level count drives file count** (`2N + 3`). Fine at N=3; worth a level allowlist
  if stacks grow.

## Migration Plan

Steps 1 and 5 of the original plan both dissolved: the "two hard upstream items" were
dropped by decision earlier in this change, and the in-memory entry point turned out
to already exist (D11). Nothing upstream gated anything.

1. Add the coconut-crab flake input and the `render` extra, pinned to rev `6356462`.
2. Add `RenderingConfig`, the `artifacts` output leaf, and `sati.render`; wire it
   into `infer` after recombination.
3. Enable it in `configs/experiment/v1/combo1b/{default,greedy_nmm}.yaml` so
   `recombination-match-and-merge`'s comparison produces overlays for both arms.
4. `subsampling=0` upstream (open question 1), then re-pin. Done, and never a gate:
   it changes how an artifact is encoded, never which artifacts exist.

No existing artifact, hash, or schema version changes, so there is nothing to
migrate on the SATI side and no clean-break to coordinate.

## Open Questions

1. **Pursue `subsampling=0` upstream? — YES. Landed upstream at `6356462`.**
   Decided from pixels, as planned. The same rendered canvas (3 px saturated box edges
   and small label text on a near-black ground, the exact pathological case) saved
   three ways and compared against the lossless PNG, over pixels belonging to a drawn
   line:

   | save | size | line-pixel mean \|err\| | max | background mean \|err\| |
   | --- | --- | --- | --- | --- |
   | PIL default (q75, 4:2:0) | 43.1 KB | **20.50** | 161 | 0.45 |
   | `subsampling=0` (4:4:4) | 67.0 KB | **4.73** | 119 | 0.18 |
   | `quality=95` | 59.1 KB | 19.63 | 147 | 0.30 |

   Visibly, not just numerically: at 8× the default's label text is mushy and the line
   edges are soft and desaturated, while `subsampling=0` is near-indistinguishable
   from the PNG.

   Worth recording: **`quality=95` does not fix it** (19.63, essentially the default),
   because `quality` tunes quantization *after* chroma has already been discarded. So
   the quality knob dropped earlier in this change was correctly dropped — it was never
   the right lever. `subsampling` is.

   Encoding is coconut-crab's (D5), so it landed there rather than being shadowed in
   SATI: `save_rendered_image` now passes `subsampling=0`, pinned at rev `6356462`.

   **Confirmed on the reference frame after re-rendering, and the cost estimate above
   was wrong.** Over the 120×60 region at the traced detection's corner, mean
   saturation on colored pixels went **93.6 → 131.1** (+40%) and peak **141 → 185**;
   the label text reads crisp instead of haloed. But the whole tree grew only
   **173.1 MB → 197.3 MB (+14%)**, not the +55% the synthetic probe implied —
   `final.jpg` 54.5 → 62.1 MB, not the ~85 MB projected. The probe overestimated
   because its canvas was near-black with the boxes as essentially the entire signal;
   on a real frame the photographic content dominates the chroma channels, so removing
   subsampling costs proportionally far less. Cheaper and better than predicted.

2. **Should the stacked artifacts distinguish boxes that survived Stage 2 from those
   absorbed by it? — Deferred; the nine artifacts did answer the question.**
   Verified on the reference frame (task 7.3): a 638×471 detection at (2395, 2094)
   with `source_levels [1, 2, 3]` and `merged_class_ids [1, 2]` shows as **two boxes**
   in `raw-levels-stacked` and `levels-stacked` (orange level-1, blue level-2) and as
   **one green `tooth` box** in `final`. Reading down the right column attributes the
   collapse to Stage 2 unambiguously, and the per-level pair shows the class collapse
   directly: `level-1.jpg` labels it `fragment:0.19` where `final.jpg` says `tooth`.
   Coloring by level was sufficient here, so this stays deferred.

3. **Is `<image-stem>` collision-safe? — Resolved: refuse the collision.**
   Neither listed option was needed. `_discover_images` is **non-recursive** — it
   handles one file or one directory's immediate children — so "same stem in different
   source directories" is not reachable within a single run. The only reachable case is
   two files in one directory differing solely by extension (`a.tif` and `a.png`).

   That case was already broken *before* rendering: the stem is the `fullframe_id` on
   every results row, so a collision silently merges two frames' rows under one ID.
   Rendering only made a latent identity bug visible by also giving it a directory to
   overwrite. So discovery now refuses it with a message naming both files, which fixes
   the results table too, and `<image-stem>` stays as the readable artifact key.
