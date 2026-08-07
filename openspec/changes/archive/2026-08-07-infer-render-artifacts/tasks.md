## 0. Upstream in coconut-crab

Not a blocker — this change can land first. Tracked here for sequencing; these are
commits in `git@github.com:sbatchelder/coconut-crab`, not in this repository.

- [x] 0.1 ~~Add an in-memory image entry point~~ **No upstream change needed.** The
  entry point already exists: `render_annotations_on_image(base, dataset, image,
  config)` (`render/engine.py:270`) takes a PIL image directly — `run_render` is just
  `resolve_image_path` + `Image.open` + that + `save_rendered_image`. `sati.render`
  calls those three public functions itself, so no base canvas ever round-trips
  through disk and the double-encode this item existed to remove never happens.
  Only the COCO *document* goes through a temp file, because `load_coco` takes a path
  (D6), and that is kilobytes rather than 308 MP.
- [x] 0.2 **Done, condition met.** The rendered reference frame did show 4:2:0
  color-bleeding, so `save_rendered_image` now passes `subsampling=0`
  (coconut-crab `6356462`). Measured after re-rendering: edge saturation +40%, tree
  size +14%. See design open question 1 for both numbers.
- [x] 0.3 ~~Tag a revision to pin~~ Pinned the revision
  `63564625c5609459d2d3aa5b49089bcc02c4d052` (the 0.2 commit) instead of cutting a
  tag; a rev pin is what the flake needs either way.

Dropped by decision, recorded so they are not rediscovered: configurable JPEG quality
(PIL's default 75 accepted) and an output long-side cap (full resolution accepted,
zooming assumed).

## 1. Dependency plumbing

- [x] 1.1 Add a `render` optional extra to `pyproject.toml` depending on
  coconut-crab; do **not** add it to core dependencies. It transitively brings
  `pillow`, `pycocotools`, `tabulate`, `tqdm`, none of which any other code path
  needs. Not on PyPI, so it is a direct `git+ssh` reference.
- [x] 1.2 Add the coconut-crab flake input to `flake.nix` pinned to rev `6356462`
  (see 0.3 — there is no `0.3` tag), and install it in the dev shell. The install
  copies the store tree out first: coconut-crab builds with setuptools, which writes
  `src/*.egg-info` into the source tree, and the Nix store is read-only.
- [x] 1.3 Defer the coconut-crab import to the point of use and raise a
  configuration error naming the `render` extra when rendering is enabled without it
  (D7). Add a test that inference with rendering disabled succeeds with the import
  unavailable.

## 2. Config surface

- [x] 2.1 Add `RenderingConfig` to `config_schemas/root.py`: `enabled: bool`,
  per-kind toggles, and `style_config: SatiPath | None`. **Five** kinds, not four —
  the count predates D2's split of `levels.jpg` into `raw-levels-stacked` and
  `levels-stacked`. Declare `style_config` as `SatiPath` so `hashing.py` pops it from
  every behavioral source (D5) — verified no hash version changes.
- [x] 2.2 Add `rendering` to `RootConfig`, and **deliberately not** to the
  active-block projection. This task originally said the opposite; that would have
  put `rendering.enabled` into `config_hash` and directly contradicted the
  `detection-rendering` requirement that an enabled and a disabled run be
  identity-equal, plus task 2.6. `rendering` is absent from every `_ACTIVE_BLOCKS`
  entry, like `prediction`, and a guard test asserts it.
- [x] 2.3 Add an `artifacts` leaf to `InferOutputsConfig` (enabled) and
  `EvalOutputsConfig` (disabled) alongside `results`/`figures`, following the
  existing `OutputLeafConfig` pattern. This is where D8's per-command default lives,
  so the `rendering` block itself stays command-agnostic.
- [x] 2.4 Add a `config_defaults/rendering/default.yaml` group; the `infer`/`eval`
  split comes from the output leaf (2.3), not from this file (D8).
- [x] 2.5 Ship a packaged example coconut-crab style TOML with per-class colors, the
  `level-N` colors used by both stacked artifacts, and labels enabled at a small
  `text_size` on every artifact (D10) — no quality or cap keys. Colors sit **flat**
  under `[style]`: coconut-crab has no `label_colors` sub-table, and nesting them
  yields one junk color entry rather than an error. A test loads the example through
  coconut-crab's own parser and asserts the colors resolve.
- [x] 2.6 Add a test that a rendering-only config difference leaves `config_hash`
  and every compatibility identity unchanged.

## 3. `sati.render`

- [x] 3.1 Build the per-artifact in-memory COCO document from detection state
  (D6): `bbox` in the canvas's coordinate space, `score` carried through so
  `{score}` resolves in the label template, `iscrowd` 0, no `segmentation`.
- [x] 3.2 Resolve canvases (D3): reuse `materialize.build_level_pyramid` for level
  canvases and the decoded frame for full-resolution canvases. `infer_frame` now
  carries the pyramid it already built on `FrameInference`, so rendering never
  resizes the stack a second time.
- [x] 3.3 Emit the uncombined per-level artifacts from confidence-filtered slice
  predictions lifted through `affine_slice_to_level` (D4), reading in-memory
  predictions so the artifact does not depend on
  `results.cache_slice_predictions`.
- [x] 3.4 Emit the combined per-level artifacts from `level_merge` state.
- [x] 3.5 Emit `raw-levels-stacked` from every level's confidence-filtered slice
  predictions lifted to full resolution, and `levels-stacked` from every level's
  `level_merge` boxes lifted to full resolution. Both use COCO categories named
  `level-N` so per-label style colors by level with no upstream change (D2). The lift
  is the same `Affine` `merge_cross_level` applies, not a second derivation.
- [x] 3.6 Emit the final artifact from `final_merge` state.
- [x] 3.7 Compose the coconut-crab config via `render_config_from_values`,
  overriding only `coco_file`, `image_root`, and `output`; everything else comes from
  the referenced TOML or coconut-crab's defaults (D5). `image_root`/`output` are
  nominal, since the canvas is handed over in memory and the output path is chosen
  per artifact — they are still supplied because the constructor requires them.
- [x] 3.8 Write artifacts to `artifacts/<image-stem>/fullframes/` with the names
  fixed in D1; resolve open question 3 (stem collisions across source directories)
  before shipping directory-input rendering.

## 4. Wiring

- [x] 4.1 Invoke rendering from `infer.py` after recombination, from in-memory
  state, and register `artifacts` in the lifecycle artifact map alongside `results`.
- [x] 4.2 Confirm collision safety and clobber behavior treat `artifacts/` the same
  as `results/`.
- [x] 4.3 Enable rendering in `configs/experiment/v1/combo1b/default.yaml` and
  `greedy_nmm.yaml` (the task said `nms.yaml`, renamed during
  `recombination-match-and-merge`) so that change's comparison produces overlays for
  both arms. Two arms here, not three: `combo1/default` is the third and is pinned to
  pre-change recombination.

## 5. Tests

- [x] 5.1 On a small synthetic frame with a multi-level stack, assert the exact file
  set: `2N + 3` files with the D1 names, and no level-zero file when the stack starts
  below full resolution.
- [x] 5.2 Assert each level artifact's pixel dimensions equal that level's realized
  dimensions, and both stacked artifacts plus final equal full-resolution dimensions.
- [x] 5.3 Assert the uncombined per-level artifact is written when
  `cache_slice_predictions` is false.
- [x] 5.4 Assert both stacked artifacts use `level-N` COCO categories, that
  `levels-stacked`'s annotation count equals the total `level_merge` row count, and
  that `raw-levels-stacked`'s equals the total confidence-filtered slice-prediction
  count — the test that pins the two stacked states apart (D2).
- [x] 5.5 Assert a box count regression per artifact against known recombination
  state, so a future merge-strategy change that silently drops boxes shows up here.
- [x] 5.6 Assert rendering produces byte-identical outputs across two runs with
  equal inputs.

## 6. Documentation

- [x] 6.1 `DESIGN-DOC-v1.md` §9: replaced the "optional debug flag materializes
  slices + per-slice overlays" sentence with §9.1, the artifact tree, and noted that
  per-slice overlays remain unimplemented with `artifacts/<stem>/slices/` reserved.
- [x] 6.2 Add `rendering.*` to the §3 config table, marking `style_config` as a
  portable path excluded from hashing, and the whole block as unhashed.
- [x] 6.3 Document the SATI/coconut-crab boundary (D5) so it is clear where to
  change a color versus where to change which images exist (§9.2), including the §4.7
  note on why the exclusion is a guarantee rather than a gap.

## 7. Verification and closeout

- [x] 7.1 Run `pytest tests/` and `ruff check src tests` in the Nix dev shell.
- [x] 7.2 Rendered the reference frame (twice: before and after 0.2)
  (`datasets/infer-me/ODP-1209C-DP36-16-11H-3W-130-132cm_picked_N1of2_z200x_2d.tif`,
  19321×15948, 3 levels → 9 artifacts). **19 s** wall with rendering, **9 s** without,
  so rendering costs ~10 s. 173 MB total: 54.7 / 54.6 / 54.5 MB full-res, 4.1 MB
  ×2 at level 1, 0.4 MB ×2 at level 2, 0.1 MB ×2 at level 3. The "render cost
  dominates" risk is closed, not merely accepted.
- [x] 7.3 Traced a cross-level, class-collapsed detection end to end: 638×471 at
  (2395, 2094), `source_levels [1, 2, 3]`, `merged_class_ids [1, 2]`. Two boxes in
  `raw-levels-stacked` and `levels-stacked` (orange level-1, blue level-2), one green
  `tooth` box in `final` — Stage 2's collapse is unambiguous from the images. The
  per-level pair also shows the class collapse directly: `level-1.jpg` labels it
  `fragment:0.19` where `final.jpg` says `tooth`. The specific 2658×1206 blob from D14
  belongs to the eval split rather than this frame, so the equivalent largest-box case
  (712×575, single-level) was used instead.
- [x] 7.4 Resolved open questions 1 and 3 and recorded both in `design.md`, with the
  4:2:0 measurement table. OQ1 resolves **yes, pursue it** — but the change is
  coconut-crab's, so task 0.2 stays open. OQ2 stays deferred, now with evidence that
  the nine artifacts were sufficient.
- [x] 7.5 Run `openspec validate --all --strict`, reconcile the main specs with these
  deltas, and leave the change ready for archive.
