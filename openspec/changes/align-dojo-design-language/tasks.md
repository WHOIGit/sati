## 1. Pin Decisions and Golden Contracts

- [ ] 1.1 Resolve the three documented open questions (coolname dependency, prepared-product default placement, and canonical figure minimum) with Sidney and update proposal/design/specs before code changes.
- [ ] 1.2 Add golden canonical-JSON source fixtures for config, dataset, prepared dataset, target schema, class mapping, model, preprocessing, checkpoint, export-model, and both recombination scopes.
- [ ] 1.3 Add ID/template golden tests covering seeded `config_id` and `prepared_dataset_id`, fresh `run_id`, dotted tokens, timestamps, relative-path classes, and invalid tokens.

## 2. Replace the Strict Config Surface

- [ ] 2.1 Implement strict `experiment`, `task`, `runtime`, raw/prepared data, structured model architecture/weights, and shared output-block Pydantic models.
- [ ] 2.2 Remove legacy `output.run_dir`, root `run_id`, `training.seed`, overloaded `data.source_path`, and scalar `model.weights` fields without aliases or compatibility validators.
- [ ] 2.3 Rewrite packaged YAML groups and every committed experiment under `configs/` to the new full-root shape, including mandatory `experiment.name`.
- [ ] 2.4 Add schema tests for strict unknown/deferred-key failures, data-source invariants, model initialization invariants, and active output requirements.

## 3. Align Composition, CLI Grammar, and Resolution

- [ ] 3.1 Change config search precedence to repeatable explicit directories, automatic `./configs`, then packaged defaults; remove local `./config_defaults` discovery.
- [ ] 3.2 Implement mutually exclusive `experiment=`, `--config`, and `--resolved-config` roots with authored-file composition and ordinary inline overrides.
- [ ] 3.3 Implement shared lexical parsing that sends Hydra override grammar to composition, Typer options to the command, and rejects unknown dash-prefixed options.
- [ ] 3.4 Add command context to compose → validate → resolve, including active-block selection, derived data/model values, hashes/IDs, and inactive-output preservation.
- [ ] 3.5 Implement dojo-style output template rendering and absolute/CWD-relative/parent-relative path resolution with traversal and invalid-target tests.
- [ ] 3.6 Make inspection resolution side-effect-free and make resolved-config replay preserve command, IDs, hashes, and paths while rejecting behavioral overrides.

## 4. Implement IDs, Hashes, and Portable Metadata

- [ ] 4.1 Implement domain-separated human ID generation and the new command-aware `config_hash` / `config_id` source projection.
- [ ] 4.2 Redefine logical `dataset_hash`; add `prepared_dataset_hash` / ID and `target_schema_hash`; rename `class_map_hash` to `class_mapping_hash` everywhere.
- [ ] 4.3 Redefine `model_config_hash` to exclude initialization weights and keep local initialization checkpoint bytes only in `config_hash`.
- [ ] 4.4 Bump affected hash recipes and result/sidecar schema versions, persist exact `*_source` objects, and remove every old recipe/name without compatibility shims.
- [ ] 4.5 Implement `sati_inference_contract` construction, checkpoint embedding/extraction, strict validation, and torch-free `checkpoint_metadata.json` inspection.
- [ ] 4.6 Verify a finalized checkpoint reloads in Ultralytics, performs a prediction, and has a filename/metadata hash matching its post-embedding bytes.

## 5. Add Product and Run Lifecycles

- [ ] 5.1 Implement safe target validation, default collision refusal, narrowly scoped `--clobber`, and atomic lifecycle-manifest updates.
- [ ] 5.2 Implement the five output blocks and common config provenance writer for composed YAML, resolved YAML/JSON, CLI text, and override text.
- [ ] 5.3 Move preparation to the content-addressed prepared-product layout with `prepare_manifest.json`, complete-product verification/reuse, and incomplete-product refusal.
- [ ] 5.4 Update training/evaluation consumers and gates to use `data.prepared.uri`, logical/prepared/preprocessing identities, and clear expected/observed mismatch messages.
- [ ] 5.5 Implement training-only exact `--resume` from a canonical last checkpoint and refuse resume for completed, mismatched, missing-checkpoint, or non-training contexts.

## 6. Normalize Training Artifacts

- [ ] 6.1 Stage all stock Ultralytics output under `backend/ultralytics/` and remove the public `<run>/train/` contract.
- [ ] 6.2 Finalize and inventory hash-stamped canonical best/last checkpoints under `checkpoints/`, using hard links when safe and copies otherwise.
- [ ] 6.3 Normalize Ultralytics epoch output into stable one-row-per-epoch `metrics/metrics.csv` columns while preserving native metrics under `backend/`.
- [ ] 6.4 Collect enabled recognized training plots under `figures/` with stable names and record unavailable optional plots in the lifecycle manifest.
- [ ] 6.5 Add end-to-end training artifact tests covering config provenance, manifests, checkpoints, metrics, figures on/off, backend isolation, collision, clobber, and resume.

## 7. Align Eval, Infer, Export, and Inspect

- [ ] 7.1 Change infer/eval/export to require `--checkpoint`, change infer to require `--input`, and prevent command I/O from mutating experiment configuration.
- [ ] 7.2 Reconstruct model/preprocessing from the embedded checkpoint contract, apply only allowed recombination/result overrides, and reject compatibility overrides.
- [ ] 7.3 Move evaluation artifacts to `eval_outputs.dir/{results,metrics,figures}` and inference artifacts to `infer_outputs.dir/results/predictions.parquet`, with canonical sidecars and lifecycle manifests.
- [ ] 7.4 Move exported models and the complete portable `inference_config.yaml` into `export_outputs.models.dir`; add exported `model_hash` and source `checkpoint_hash`.
- [ ] 7.5 Make `sati inspect config` config-first, add hash-source-aware `config-compare`, standardize `--format text|json`, and teach artifact inspectors the new envelopes.
- [ ] 7.6 Update `convert` to the shared config-source/parser/lifecycle conventions while preserving its no-slicing interchange semantics.

## 8. Verification and Documentation Closeout

- [ ] 8.1 Update all focused config, hash, manifest, materialization, backend, training, inference, evaluation, export, conversion, inspect, and CLI tests for the clean-break interfaces.
- [ ] 8.2 Run `.venv/bin/python -m pytest tests/` and `.venv/bin/ruff check src tests`; fix all regressions.
- [ ] 8.3 Run the real Ultralytics checkpoint round-trip and `SATI_RUN_TRIPWIRE=1` coordinate-contract test in the supported Nix development environment.
- [ ] 8.4 Update README command examples, DESIGN-DOC-v1 current contracts, forward references in DESIGN-DOC-v2, and any affected DESIGN-DOC-SYNTH integration wording.
- [ ] 8.5 Add a timestamped `REPORTS/` implementation entry, record any deferred non-blocking choices in `QUESTIONS-FOR-SIDNEY.md`, and complete the full DOX hierarchy/index pass.
- [ ] 8.6 Run `openspec validate --all --strict`, reconcile the implemented main specs with the change deltas, and leave the change ready for archive.
