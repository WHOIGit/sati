## Why

`sati infer` and `sati eval` build their level pyramid at runtime, per frame, from
`cfg.slicing` — nothing about the pyramid is baked into the checkpoint. Yet
`apply_inference_contract` replaces the whole `SlicingConfig` with the contract's
copy, so `slicing.level_policy.start_scale=0.5` resolves successfully and is then
discarded before a single slice is cut. No error, no warning, and the
`preprocessing_hash` check cannot catch it because it re-hashes the values it just
assigned. The run's `config/resolved.yaml` records the checkpoint's pyramid while
the shell history says otherwise.

Two problems, one cause. The silent discard is a provenance defect against the
*current* spec, which says infer/eval "SHALL reject changes to ... preprocessing" —
it neither rejects nor honors. And the blanket pin is too wide: only
`nominal_slice_size` is genuinely model-coupled, so asking "does this model do
better with a denser pyramid?" currently costs a full re-prepare and retrain for a
question the existing weights can answer.

## What Changes

- Narrow contract reconstruction to the model-coupled subset. `apply_inference_contract`
  restores `nominal_slice_size` and `image_format` from the contract and leaves
  `overlap_ratio`, `level_policy`, and `max_levels` at their resolved values.
- **BREAKING** Overriding `nominal_slice_size` or `image_format` at infer/eval time
  becomes a named refusal instead of a silent revert. Previously such a config ran to
  completion under the checkpoint's values.
- Honor `slicing.overlap_ratio`, `slicing.level_policy.*`, and `slicing.max_levels`
  as inline overrides on `infer` and `eval`, applying to the runtime pyramid.
- Warn, without refusing, when the runtime level stack shares no level scale with the
  trained stack — the divergence most likely to be misread as a model or data fault.
- Pin slice geometry on `sati export`: its portable inference config always describes
  the trained pyramid, and a geometry override supplied to `export` warns that it is
  inert rather than being silently dropped.
- **BREAKING** Record `effective_preprocessing_hash` on detection result rows and in
  the results sidecar — `preprocessing_hash` applied to the *runtime* slicing, using
  the existing source and version. It equals the contract's `preprocessing_hash` when
  nothing is overridden, so an unoverridden run is unchanged in value but gains a
  column. Adding a non-nullable column to `VALIDATION_RESULTS_SCHEMA` breaks readers
  of existing parquet.
- No change to `preprocessing_hash` itself: same source, same version, same role as
  the trained-pyramid compatibility identity carried in the contract. Existing
  checkpoints keep verifying.

Non-goals. `sati prepare` and `sati train` keep the authored slicing verbatim —
re-slicing is an inference-time question and the prepared product stays the training
pyramid's. Exporting a re-sliced geometry is out of scope by decision: an export means
the trained schema, and promoting a re-sliced pyramid into a deployable artifact is a
retraining decision rather than an export flag. The `min_area_ratio` and `size_gate_px`
gates are prepare-time annotation projection, are absent from `preprocessing_hash`, and
are untouched.

## Capabilities

### New Capabilities

(none — this narrows and corrects existing behavior rather than introducing a new
capability surface)

### Modified Capabilities

- `inference-evaluation-export`: "Checkpoint contract reconstruction" narrows to the
  model-coupled preprocessing subset; "Recombination override precedence" extends to
  admit slice geometry as an accepted override class, and its rejection clause
  narrows from all preprocessing to the model-coupled fields, with rejection becoming
  an explicit refusal rather than a silent restore; "Exported model identity" gains
  that export pins slice geometry to the contract and warns on an inert override.
- `results-provenance`: detection results and the metadata sidecar gain
  `effective_preprocessing_hash` alongside the contract's `preprocessing_hash`, so a
  re-sliced run is self-describing.
- `slicing-and-recombination`: "Resolution stack policies" gains that the level
  policy is evaluated per command from the resolved config, and is not fixed by the
  checkpoint.

## Impact

- `src/sati/checkpoints.py` — `apply_inference_contract` field-selective restore plus
  the new refusal path.
- `src/sati/hashing.py` — no source or version change; `preprocessing_hash` gains a
  second call site over runtime slicing.
- `src/sati/manifests.py` — `effective_preprocessing_hash` column on the detection
  results schema (hand-written PyArrow field, per the package contract).
- `src/sati/infer.py`, `src/sati/evaluation.py` — thread the effective hash through
  `run_hashes_for` / `rows_for_frame` and the results sidecar.
- `src/sati/errors.py` — refusal type for model-coupled preprocessing overrides.
- `src/sati/export.py` — geometry stays contract-sourced; inert-override warning.
- `src/sati/inspect.py` — the compatibility-projection field set at `inspect.py:101`
  splits into pinned and geometry groups; `config-compare` output changes shape.
- Existing `runs/**/results/*.parquet` lack the new column and will not satisfy the
  updated schema. Per the repo's no-backwards-compatibility rule for v1 development
  artifacts, they are not migrated.
- Unblocks the `combo1b` slicing question without a `combo1c` prepare + train.
