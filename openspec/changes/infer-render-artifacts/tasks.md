## 0. Upstream in coconut-crab

Not a blocker — this change can land first. Tracked here for sequencing; these are
commits in `git@github.com:sbatchelder/coconut-crab`, not in this repository.

- [ ] 0.1 Add an in-memory image entry point accepting a PIL image or ndarray instead
  of only a `coco_file` path plus `image_root`, removing SATI's base-canvas round trip
  through disk (two full encodes per artifact, three of them 308 MP).
- [ ] 0.2 Conditional on open question 1 only: add `subsampling=0` to the JPEG save
  path if a rendered reference frame shows 4:2:0 color-bleeding on thin box edges.
- [ ] 0.3 Tag a revision to pin from SATI's flake.

Dropped by decision, recorded so they are not rediscovered: configurable JPEG quality
(PIL's default 75 accepted) and an output long-side cap (full resolution accepted,
zooming assumed).

## 1. Dependency plumbing

- [ ] 1.1 Add a `render` optional extra to `pyproject.toml` depending on
  coconut-crab; do **not** add it to core dependencies. It transitively brings
  `pillow`, `pycocotools`, `tabulate`, `tqdm`, none of which any other code path
  needs.
- [ ] 1.2 Add the coconut-crab flake input to `flake.nix` pinned to the 0.3
  revision, and make it available in the dev shell.
- [ ] 1.3 Defer the coconut-crab import to the point of use and raise a
  configuration error naming the `render` extra when rendering is enabled without it
  (D7). Add a test that inference with rendering disabled succeeds with the import
  unavailable.

## 2. Config surface

- [ ] 2.1 Add `RenderingConfig` to `config_schemas/root.py`: `enabled: bool`,
  per-kind toggles for the four artifact kinds, and
  `style_config: SatiPath | None`. Declare `style_config` as `SatiPath` so
  `hashing.py:159` pops it from every behavioral source (D5) — verify no hash version
  changes.
- [ ] 2.2 Add `rendering` to `RootConfig` and to the active-block projection for
  `infer`/`eval`.
- [ ] 2.3 Add an `artifacts` leaf to `InferOutputsConfig` (and `EvalOutputsConfig`,
  disabled) alongside `results`/`figures`, following the existing `OutputLeafConfig`
  pattern at `config_schemas/root.py:265`.
- [ ] 2.4 Add a `config_defaults/rendering/default.yaml` group: enabled true for
  `infer`, false for `eval` (D8).
- [ ] 2.5 Ship a packaged example coconut-crab style TOML with per-class colors, the
  `level-N` colors used by both stacked artifacts, and labels enabled at a small
  `text_size` on every artifact (D10) — no quality or cap keys.
- [ ] 2.6 Add a test that a rendering-only config difference leaves `config_hash`
  and every compatibility identity unchanged.

## 3. `sati.render`

- [ ] 3.1 Build the per-artifact in-memory COCO document from detection state
  (D6): `bbox` in the canvas's coordinate space, `score` carried through so
  `{score}` resolves in the label template, `iscrowd` 0, no `segmentation`.
- [ ] 3.2 Resolve canvases (D3): reuse `materialize.build_level_pyramid`
  (`materialize.py:65`) for level canvases — do not re-derive level pixels — and the
  decoded frame for full-resolution canvases.
- [ ] 3.3 Emit the uncombined per-level artifacts from confidence-filtered slice
  predictions lifted through `affine_slice_to_level` (D4), reading in-memory
  predictions so the artifact does not depend on
  `results.cache_slice_predictions`.
- [ ] 3.4 Emit the combined per-level artifacts from `level_merge` state.
- [ ] 3.5 Emit `raw-levels-stacked` from every level's confidence-filtered slice
  predictions lifted to full resolution, and `levels-stacked` from every level's
  `level_merge` boxes lifted to full resolution. Both use COCO categories named
  `level-N` so `RenderStyle.label_colors` colors by level with no upstream change
  (D2).
- [ ] 3.6 Emit the final artifact from `final_merge` state.
- [ ] 3.7 Compose the coconut-crab config via `render_config_from_values`
  (`render/config.py:161`), overriding only `coco_file`, `image_root`, and `output`;
  everything else comes from the referenced TOML or coconut-crab's defaults (D5).
- [ ] 3.8 Write artifacts to `artifacts/<image-stem>/fullframes/` with the names
  fixed in D1; resolve open question 3 (stem collisions across source directories)
  before shipping directory-input rendering.

## 4. Wiring

- [ ] 4.1 Invoke rendering from `infer.py` after recombination, from in-memory
  state, and register `artifacts` in the lifecycle artifact map alongside `results`.
- [ ] 4.2 Confirm collision safety and clobber behavior treat `artifacts/` the same
  as `results/`.
- [ ] 4.3 Enable rendering in `configs/experiment/v1/combo1b/default.yaml` and
  `nms.yaml` so the `greedy-nmm-recombination` A/B/C comparison produces overlays
  for all three arms.

## 5. Tests

- [ ] 5.1 On a small synthetic frame with a multi-level stack, assert the exact file
  set: `2N + 3` files with the D1 names, and no level-zero file when the stack starts
  below full resolution.
- [ ] 5.2 Assert each level artifact's pixel dimensions equal that level's realized
  dimensions, and both stacked artifacts plus final equal full-resolution dimensions.
- [ ] 5.3 Assert the uncombined per-level artifact is written when
  `cache_slice_predictions` is false.
- [ ] 5.4 Assert both stacked artifacts use `level-N` COCO categories, that
  `levels-stacked`'s annotation count equals the total `level_merge` row count, and
  that `raw-levels-stacked`'s equals the total confidence-filtered slice-prediction
  count — the test that pins the two stacked states apart (D2).
- [ ] 5.5 Assert a box count regression per artifact against known recombination
  state, so a future merge-strategy change that silently drops boxes shows up here.
- [ ] 5.6 Assert rendering produces byte-identical outputs across two runs with
  equal inputs.

## 6. Documentation

- [ ] 6.1 `DESIGN-DOC-v1.md` §9: replace the "optional debug flag materializes
  slices + per-slice overlays" sentence with this artifact tree, and note that
  per-slice overlays remain unimplemented with `artifacts/<stem>/slices/` reserved.
- [ ] 6.2 Add `rendering.*` to the §3 config table, marking `style_config` as a
  portable path excluded from hashing.
- [ ] 6.3 Document the SATI/coconut-crab boundary (D5) so it is clear where to
  change a color versus where to change which images exist.

## 7. Verification and closeout

- [ ] 7.1 Run `pytest tests/` and `ruff check src tests` in the Nix dev shell.
- [ ] 7.2 Render the reference frame
  (`datasets/infer-me/ODP-1209C-DP36-16-11H-3W-130-132cm_picked_N1of2_z200x_2d.tif`,
  19321×15948, 3 levels → 9 artifacts) and record wall-clock time and per-file sizes —
  the numbers that decide how urgent upstream 0.1 is.
- [ ] 7.3 Visually confirm the 2658×1206 false-positive blob and the nested real
  detections from `greedy-nmm-recombination` D14 are identifiable in `raw-level-N.jpg`
  and traceable through `raw-levels-stacked.jpg` → `levels-stacked.jpg` →
  `final.jpg` — the concrete test that this artifact set answers the question it was
  built for. While there, zoom a thin box edge and settle open question 1
  (4:2:0 bleeding) from the pixels.
- [ ] 7.4 Resolve open questions 1 and 3 and record the decisions in `design.md`.
- [ ] 7.5 Run `openspec validate --all --strict`, reconcile the main specs with these
  deltas, and leave the change ready for archive.
