## Why

Recombination is the project's research axis and it is currently **invisible**.
`greedy-nmm-recombination` is being decided from Parquet queries and ad-hoc
simulation scripts, because there is no way to look at a frame and see what the
merge stages did. The one existing overlay in `runs/` — a 274 MB
`*_annotated.png` — was produced outside this repository by hand; `src/sati` has
no rendering code at all.

The artifact set proposed here is a **visual trace of the two-stage pipeline**, one
image per pipeline state:

```
   per level (level canvas)              stacked (full-res canvas)

   raw-level-N.jpg  ───────────────────▶ raw-levels-stacked.jpg
        │  Stage 1: within-level merge        │
        ▼                                     ▼
   level-N.jpg      ───────────────────▶ levels-stacked.jpg
                        lift + overlay        │
                                              ▼  Stage 2: cross-level merge
                                         final.jpg
```

Reading down either column shows which stage created or destroyed a box. That is
the difference between choosing `max_merge_growth` from a number and choosing it
from a picture.

Rendering is delegated to **coconut-crab** (`git@github.com:sbatchelder/coconut-crab`)
rather than reimplemented. It already does COCO-over-image box rendering with
per-category colors, label templating, opacity, line weight, and text placement,
and it already disables PIL's decompression-bomb limit (`engine.py:19`,
`MAX_IMAGE_PIXELS = None`) so gigapixel frames load. SATI supplies geometry and
layout; coconut-crab owns style.

## What Changes

- **NEW** An `artifacts/` output leaf under the `infer` run directory, laid out as
  `artifacts/<image-stem>/fullframes/`, containing `final.jpg`,
  `levels-stacked.jpg`, `raw-levels-stacked.jpg`, and `level-N.jpg` +
  `raw-level-N.jpg` for every level in the frame's stack (`2N + 3` files). The
  `<image-stem>/` and `fullframes/` nesting reserves room for the per-slice
  overlays already anticipated by `DESIGN-DOC-v1.md` §9 without a later move.
- **NEW** `sati.render`: builds one in-memory COCO document per artifact from the
  detection rows SATI already computes, chooses the base canvas, and calls
  coconut-crab. No new geometry — every box comes from `slice_prediction`,
  `level_merge`, or `final_merge` state that recombination already produces.
- **NEW config** A `rendering` block: `enabled`, which artifact kinds to emit, and
  `style_config` — a `SatiPath` to a coconut-crab render TOML. Default **enabled
  for `infer`, disabled for `eval`**: eval runs a whole split, and rendering every
  frame's full pyramid would produce hundreds of gigapixel JPEGs unasked.
- **NEW dependency** coconut-crab as an **optional extra** (`sati[render]`), not a
  core dependency. It brings `pillow`, `pycocotools`, `tabulate`, and `tqdm`;
  `prepare`, `train`, and `infer`-without-rendering must not require them. This
  follows the existing pattern where `ultralytics` and `pycocotools` are already
  extras. Requires a flake input as well as a `pyproject.toml` entry, since
  coconut-crab is not published to PyPI.
- **NON-BREAKING** for every existing hash and artifact. Rendering cannot change a
  detection, a metric, or a digest. `style_config` is declared `SatiPath`, which
  `hashing.py:159` already pops from every behavioral source, so no hash version
  moves and no hash machinery changes.
- **ENCODING settled** JPEG at PIL's default quality, at true full resolution — no
  quality field and no long-side cap. The three full-res artifacts are 308 MP on the
  reference frame, accepted because inspection means zooming in; label text is
  correspondingly small. WebP was rejected outright: its 16383 px/axis ceiling cannot
  hold a 19321-wide frame.
- **UPSTREAM (coconut-crab)** One item, a performance and image-quality issue rather
  than a hard blocker: `run_render` resolves images from disk paths only, so SATI must
  write each base canvas out for coconut-crab to read back and re-save — two full
  encodes per artifact, on three 308 MP canvases. An in-memory entry point removes it
  without changing the artifact contract.

  Two other gaps need **no** work. Score labels already resolve —
  `annotation_field_context` (`engine.py:496`) falls through to raw annotation fields
  case-insensitively, so `text = "{CATEGORY}:{score}"` reads COCO's own `score`.
  Gigapixel input already loads — `engine.py:19` sets `MAX_IMAGE_PIXELS = None`.

  One candidate remains open: PIL applies 4:2:0 chroma subsampling below quality 95,
  which color-bleeds the thin saturated lines these artifacts consist of.
  `subsampling=0` is a one-line upstream fix, gated on looking at a rendered frame
  first (design open question 1).

## Capabilities

### New Capabilities

- `detection-rendering`: the artifact tree, its layout and naming, which pipeline
  state each image depicts, the canvas each is drawn on, and the boundary between
  SATI-owned geometry and coconut-crab-owned style.

### Modified Capabilities

- `inference-evaluation-export`: `infer` emits the artifact tree alongside results;
  `eval` does not by default. Rendering is explicitly excluded from the
  recombination-override identity rules, because it moves no identity.
- `config-and-cli`: the root schema gains a `rendering` block whose style pointer
  is a portable path.
- `run-artifact-lifecycle`: `artifacts/` joins `results/` as a declared output leaf
  of the `infer` block, subject to the same collision-safety rules.

## Non-Goals

- Per-slice overlays (`artifacts/<stem>/slices/`). The layout reserves the path;
  this change does not fill it.
- Rendering ground truth, or GT-vs-prediction comparison images. coconut-crab's
  `compare` module could do it later; scoring visualization is not this change.
- Segmentation masks. SATI is boxes-only in v1.
- Rendering during `train`.
- Replacing `figures` (the existing disabled-by-default leaf for plots). Metric
  figures and detection overlays stay separate concerns.
- Reimplementing any drawing primitive that coconut-crab already provides. If a
  style knob is missing, it is added upstream, not shadowed in SATI.

## Impact

- New `src/sati/render.py` and a `RenderingConfig` in `src/sati/config_schemas/`.
- `src/sati/infer.py`: invoke rendering after recombination, from in-memory state.
- `src/sati/materialize.py`: reuse `build_level_pyramid` (`materialize.py:65`) for
  level-resolution canvases — it already resizes each level from full-res directly
  with `INTER_AREA`, never cascaded, which is exactly the correct canvas.
- `pyproject.toml` (`render` extra) and `flake.nix` (coconut-crab input).
- `src/sati/config_defaults/`: a new `rendering/` group with a `default.yaml`, plus
  a packaged example coconut-crab style TOML.
- Docs: `DESIGN-DOC-v1.md` §9 (the debug-flag sentence becomes this artifact tree)
  and the §3 config table.
- Not blocked upstream: the single coconut-crab item is a performance and
  image-quality optimization, so this change can land first and absorb it later
  without altering the artifact contract.
