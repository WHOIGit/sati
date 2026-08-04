## 1. Identity and Golden Contracts

- [x] 1.1 Add golden canonical-JSON source fixtures for config, dataset,
  prepared dataset, target schema, class mapping, model, preprocessing, and
  both recombination scopes; add fixed-byte fixtures for checkpoint and
  exported-model byte hashes.
- [x] 1.2 Add ID/template golden tests covering `coolname`-derived `config_id` and `prepared_dataset_id`, fresh `run_id`, dotted tokens, timestamps, relative-path classes, and invalid tokens.
- [x] 1.3 Add resolved-config golden tests proving that the persisted YAML is the immutable worker configuration, its JSON rendering has the same tree, hash sources derive from it, and exact replay revalidates without recomputing identities or paths.

## 2. Replace the Strict Config Surface

- [x] 2.1 Implement strict `experiment`, `task`, `runtime`, raw/prepared data, structured model architecture/weights, and shared output-block Pydantic models.
- [x] 2.2 Remove legacy `output.run_dir`, root `run_id`, `training.seed`, overloaded `data.source_path`, and scalar `model.weights` fields without aliases or compatibility validators.
- [x] 2.3 Rewrite packaged YAML groups and every committed experiment under `configs/` to the new full-root shape, including mandatory `experiment.name`.
- [x] 2.4 Add schema tests for strict unknown/deferred-key failures, data-source invariants, model initialization invariants, and active output requirements.

## 3. Align Composition, CLI Grammar, and Resolution

- [x] 3.1 Change config search precedence to repeatable explicit directories, automatic `./configs`, then packaged defaults; remove local `./config_defaults` discovery.
- [x] 3.2 Implement mutually exclusive `experiment=`, `--config`, and `--resolved-config` roots with authored-file composition and ordinary inline overrides.
- [x] 3.3 Implement shared lexical parsing that sends Hydra override grammar to composition, Typer options to the command, and rejects unknown dash-prefixed options.
- [x] 3.4 Add command context to compose → validate → resolve, including active-block selection, derived data/model values, hashes/IDs, and inactive-output preservation.
- [x] 3.5 Implement dojo-style output template rendering and absolute/CWD-relative/parent-relative path resolution with traversal and invalid-target tests.
- [x] 3.6 Make inspection resolution side-effect-free; atomically persist the authoritative resolved YAML before work; and make resolved-config replay preserve command, IDs, hashes, and paths while rejecting behavioral overrides.

## 4. Implement IDs, Hashes, and Portable Metadata

- [x] 4.1 Implement domain-separated human ID generation and the new command-aware `config_hash` / `config_id` source projection.
- [x] 4.2 Redefine logical `dataset_hash`; add `prepared_dataset_hash` / ID and `target_schema_hash`; rename `class_map_hash` to `class_mapping_hash` everywhere.
- [x] 4.3 Redefine `model_config_hash` to exclude initialization weights and keep local initialization checkpoint bytes only in `config_hash`.
- [x] 4.4 Bump affected hash recipes and result/sidecar schema versions, persist exact `*_source` objects, and remove every old recipe/name without compatibility shims.
- [x] 4.5 Implement `sati_inference_contract` construction, checkpoint embedding/extraction, strict validation, and torch-free `checkpoint_metadata.json` inspection.
- [x] 4.6 Verify a finalized checkpoint reloads in Ultralytics, performs a prediction, and has a filename/metadata hash matching its post-embedding bytes.

## 5. Add Product and Run Lifecycles

- [x] 5.1 Implement safe target validation, default collision refusal, narrowly scoped `--clobber`, and atomic lifecycle-manifest updates.
- [x] 5.2 Implement the five output blocks and common config provenance writer for composed YAML, resolved YAML/JSON, CLI text, and override text.
- [x] 5.3 Move preparation to the content-addressed prepared-product layout with `prepare_manifest.json`, complete-product verification/reuse, and incomplete-product refusal.
- [x] 5.4 Update training/evaluation consumers and gates to use `data.prepared.uri`, logical/prepared/preprocessing identities, and clear expected/observed mismatch messages.
- [x] 5.5 Implement training-only exact `--resume` from a canonical last checkpoint and refuse resume for completed, mismatched, missing-checkpoint, or non-training contexts.

## 6. Normalize Training Artifacts

- [x] 6.1 Stage all stock Ultralytics output under `backend/ultralytics/` and remove the public `<run>/train/` contract.
- [x] 6.2 Finalize and inventory hash-stamped canonical best/last checkpoints under `checkpoints/`, using hard links when safe and copies otherwise.
- [x] 6.3 Normalize Ultralytics epoch output into stable one-row-per-epoch `metrics/metrics.csv` columns while preserving native metrics under `backend/`.
- [x] 6.4 Copy every native Ultralytics figure under `figures/ultralytics/` with its filename unchanged; do not create figure inventory or absent-figure records.
- [x] 6.5 Add end-to-end training artifact tests covering config provenance, manifests, checkpoints, metrics, figures on/off, backend isolation, collision, clobber, and resume.

## 7. Align Eval, Infer, Export, and Inspect

- [x] 7.1 Change infer/eval/export to require `--checkpoint`, change infer to require `--input`, and prevent command I/O from mutating experiment configuration.
- [x] 7.2 Reconstruct model/preprocessing from the embedded checkpoint contract, apply only allowed recombination/result overrides, and reject compatibility overrides.
- [x] 7.3 Move evaluation artifacts to `eval_outputs.dir/{results,metrics,figures}` and inference artifacts to `infer_outputs.dir/results/predictions.parquet`, with canonical sidecars and lifecycle manifests.
- [x] 7.4 Move exported models and the complete portable `inference_config.yaml` into `export_outputs.models.dir`; add exported `model_hash` and source `checkpoint_hash`.
- [x] 7.5 Make `sati inspect config` config-first, add hash-source-aware `config-compare`, standardize `--format text|json`, and teach artifact inspectors the new envelopes.
- [x] 7.6 Update `convert` to the shared config-source/parser/lifecycle conventions while preserving its no-slicing interchange semantics.

## 8. Verification and Documentation Closeout

- [x] 8.1 Update all focused config, hash, manifest, materialization, backend, training, inference, evaluation, export, conversion, inspect, and CLI tests for the clean-break interfaces.
- [x] 8.2 Run `.venv/bin/python -m pytest tests/` and `.venv/bin/ruff check src tests`; fix all regressions.
- [x] 8.3 Run the real Ultralytics checkpoint round-trip and `SATI_RUN_TRIPWIRE=1` coordinate-contract test in the supported Nix development environment.
- [x] 8.4 Update README command examples, DESIGN-DOC-v1 current contracts, forward references in DESIGN-DOC-v2, and any affected DESIGN-DOC-SYNTH integration wording.
- [x] 8.5 Add a timestamped `REPORTS/` implementation entry, record any deferred non-blocking choices in `QUESTIONS-FOR-SIDNEY.md`, and complete the full DOX hierarchy/index pass.
- [x] 8.6 Run `openspec validate --all --strict`, reconcile the implemented main specs with the change deltas, and leave the change ready for archive.
