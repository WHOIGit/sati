## 1. Contract reconstruction

- [ ] 1.1 Add a `PreprocessingOverrideRefused` error to `src/sati/errors.py` carrying the offending field name.
- [ ] 1.2 In `src/sati/checkpoints.py`, split `apply_inference_contract` so it restores only `nominal_slice_size` and `image_format` from the contract and leaves `overlap_ratio`, `level_policy`, and `max_levels` at their resolved values.
- [ ] 1.3 Raise `PreprocessingOverrideRefused` when the resolved `nominal_slice_size` or `image_format` differs from the contract's, before any backend load or forward pass. Hard fail — this path never warns (D2).
- [ ] 1.4 Fix the tautological verification: rebuild a `SlicingConfig` from `contract.slicing` and verify `preprocessing_hash` against that, not against the mutated `cfg.slicing`.
- [ ] 1.5 Keep `sati export` sourcing slice geometry wholesale from the contract, so exported preprocessing metadata always describes the trained pyramid (D7).
- [ ] 1.6 Emit a warning from `export` naming any resolved geometry field that differs from the contract's, so an inert override is audible rather than dropped (D8).

## 2. Effective preprocessing identity

- [ ] 2.1 Compute `effective_preprocessing_hash` as `preprocessing_hash(cfg.slicing)` over the resolved runtime slicing; add no new hash source and do not bump `PREPROCESSING_HASH_VERSION`.
- [ ] 2.2 Thread it through `run_hashes_for` in `src/sati/infer.py` alongside the contract's `preprocessing_hash`.
- [ ] 2.3 Add the non-nullable `effective_preprocessing_hash` column to the detection results schema in `src/sati/manifests.py`, hand-written column-by-column per the package contract.
- [ ] 2.4 Emit the column from `rows_for_frame` so it appears on every row type that already carries `preprocessing_hash`.
- [ ] 2.5 Record the hash and its exact source in the results sidecar via `write_results_sidecar`, in the config key namespace with its `version` field.

## 3. Pyramid divergence warning

- [ ] 3.1 Record the trained level stack on the inference contract, or derive it from the contract's slicing at compare time, so the runtime stack has something to compare against.
- [ ] 3.2 Compare planned runtime level scales against the trained stack's after `plan_frame` and before the first forward pass; warn naming both stacks when they share no level scale (D6).
- [ ] 3.3 Emit the warning once per command, not once per frame — eval plans a stack for every validation frame.
- [ ] 3.4 Confirm no divergence path can refuse: the warning is advisory on every command that honors geometry.

## 4. Eval parity

- [ ] 4.1 Verify `execute_eval` in `src/sati/evaluation.py` needs no pyramid change — it already calls the shared `infer_frame` on full-res pixels from `source_uri`.
- [ ] 4.2 Confirm `prepared_dataset_hash` on eval rows keeps describing split/ground-truth provenance and is unaffected by a geometry override.
- [ ] 4.3 Confirm scale diagnostics and `n_final_merge` compute correctly over a re-sliced level stack.

## 5. Inspection

- [ ] 5.1 Split the flat compatibility-projection field set at `src/sati/inspect.py:101` into a pinned group (`nominal_slice_size`, `image_format`) and an overridable slice-geometry group (`overlap_ratio`, `level_policy`, `max_levels`).
- [ ] 5.2 Update `summarize_config_compare` so a slice-geometry difference reports under the overridable group and is not presented as a compatibility break.
- [ ] 5.3 Surface both preprocessing identities in `summarize_results`.

## 6. Tests

- [ ] 6.1 Regression test: an inline `slicing.level_policy.start_scale` override on `infer` changes the planned level count and the effective hash — the test that would have caught the silent discard.
- [ ] 6.2 Test that an unoverridden run produces `effective_preprocessing_hash == preprocessing_hash`.
- [ ] 6.3 Test that a `nominal_slice_size` or `image_format` divergence raises `PreprocessingOverrideRefused` before a forward pass, on infer, eval, and export alike.
- [ ] 6.4 Test that a contract whose stored `preprocessing_hash` disagrees with its stored preprocessing fields now fails verification.
- [ ] 6.5 Test eval re-slicing against an existing prepared product: unchanged `prepared_dataset_hash`, changed `effective_preprocessing_hash`.
- [ ] 6.6 Test that `sati inspect config-compare` groups a level-policy difference as overridable.
- [ ] 6.7 Test the divergence warning fires on a non-overlapping stack, stays silent on an overlapping one, and never raises.
- [ ] 6.8 Test that a geometry override on `export` warns and leaves the exported inference config carrying the contract's geometry.

## 7. Removal of superseded behavior

- [ ] 7.1 Remove the wholesale `cfg.slicing` reassignment from the infer/eval path; leave no compatibility path that restores the full contract slicing for commands that honor geometry.
- [ ] 7.2 Do not migrate existing `runs/**/results/*.parquet` — they predate the new column and are superseded per the v1 no-backwards-compatibility rule. Divergence between old and new results files is accepted (Sidney, decided).

## 8. Verification and closeout

- [ ] 8.1 `nix develop -c .venv/bin/python -m pytest tests/`
- [ ] 8.2 `SATI_RUN_TRIPWIRE=1 nix develop -c .venv/bin/python -m pytest tests/test_backends_train.py::test_ultralytics_coordinate_tripwire tests/test_checkpoints_lifecycle.py::test_real_ultralytics_finalized_checkpoint_round_trip`
- [ ] 8.3 `nix develop -c ruff check src tests`
- [ ] 8.4 End-to-end check against the combo1b checkpoint: `sati infer ... slicing.level_policy.start_scale=0.5 slicing.level_policy.subsequent_downsample_factor=2.0` produces a different level count and a differing effective hash.
- [ ] 8.5 DOX pass: update `src/sati/AGENTS.md` local contracts for the narrowed contract reconstruction and the two preprocessing identities; check `openspec/AGENTS.md` and `configs/AGENTS.md` for knock-on edits.
- [ ] 8.6 Record the remaining open questions from `design.md` (same-experiment eval separation, D6 trigger calibration) in `QUESTIONS-FOR-SIDNEY.md` with a timestamp.
- [ ] 8.7 `openspec validate --all --strict`
