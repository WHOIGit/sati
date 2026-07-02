# src/synth_dataset_builder

## Purpose

Standalone procedural synthetic dataset builder and SATI diagnostic stress suite, per `DESIGN-DOC-SYNTH.md`. Console script `sati-synth`; installed via the `synth` extra.

## Ownership

Owns all `synth_dataset_builder.*` modules. Target layout per DESIGN-DOC-SYNTH §11: `cli`, `config`, `rng`, `planner`, `scale`, `layout`, `probe_grid`, `grammar/`, `backgrounds`, `render`, `truth`, `export/`.

## Local Contracts

- **No `sati.*` imports, ever** — standalone-ness is a hard rule. SATI interop happens only through outputs (COCO JSON + `diagnostics/objects.parquet`).
- Plan/render split: the planner consumes all randomness (`SeedSequence.spawn()` tree, spawn keys recorded) and never imports cv2; render is a pure function of the plan and makes zero random draws.
- Truth (bbox/polygon/RLE) derives from pre-composite supersampled alpha, never from composited pixels.
- Scale/layout/difficulty coverage is quota-driven, never uniform-random.
- `probe_grid.py` copies SATI's slicing arithmetic (does not import it), guarded by a cross-test fixture shared with `sati.tiling` tests.
- Explicit hand-written PyArrow schemas; coordinates half-open `xyxy`, top-left origin.

## Work Guidance

- Implement DESIGN-DOC-SYNTH Stage 1 first (5 geometric classes, fBm background, quota sampler, full export path); exit test = SATI ingests the output and the per-band mAP join works.

## Verification

(none yet — update when the test suite exists)

## Child DOX Index

(none)
