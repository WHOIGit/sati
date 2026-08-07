## 1. Shared canonical mechanics and marker helpers

- [x] 1.1 Implement the dojo-aligned canonical-JSON path (`_canonicalize` +
  `json.dumps(sort_keys=True, separators=(",", ":"))`, `sha256:` prefix) and
  route all hashing through it; delete the hand-rolled string builder.
- [x] 1.2 Add a pinned cross-tool guard test hashing a tricky tree (nested dicts,
  `-0.0`, non-finite floats, unicode) to a value shared with dojo's helper.
- [x] 1.3 Replace inline `recipe: "sati.<name>_hash/<n>"` strings with a `version`
  field convention; `config_hash` carries no synthetic version token.
- [x] 1.4 Implement `HASH_DERIVED`-driven and `SatiPathMarker`-driven exclusion
  walkers and retire the hand-maintained pop/exclusion lists in `hashing.py`.

## 2. Rebuild hash sources in the bare namespace

- [x] 2.1 Rebuild `config_hash_source` as a bare, command-aware projection of the
  resolved config (D4): drop the `{recipe, command, config}` envelope; select
  active blocks; pop output blocks and in-behavior `SatiPath` fields (D5); keep
  the `model.weights` byte-substitution exception.
- [x] 2.2 Rebuild `model_config_hash`, `preprocessing_hash`, `target_schema_hash`,
  `dataset_hash`, `prepared_dataset_hash`, `class_mapping_hash`, and scope-aware
  `recombination_hash` sources in the bare namespace with `version` fields; bump
  every version.
- [x] 2.3 Apply the `version`-field convention to checkpoint/export metadata where
  it carries source objects, leaving byte hashing unchanged.
- [x] 2.4 Simplify `TaskConfig` to `type` only (remove `box_format`, `box_bounds`,
  `origin`); encode the top-left/half-open/`xyxy` convention as a versioned
  constant in the `target_schema_hash` source; drop the removed keys from
  `config_defaults/base/default.yaml`.

## 3. Update consumers and stored provenance

- [x] 3.1 Update `manifests.py` sidecar readers/writers and the strict metadata
  model to the `version`-field sources in the config namespace.
- [x] 3.2 Update `checkpoints.py` embedded contract and `train/eval/infer/export`
  result provenance to the new source shapes; bump result/sidecar/checkpoint
  `schema_version` values.
- [x] 3.3 Simplify `inspect config-compare` to compute coverage via the now-aligned
  namespace (dojo's `_leaf_paths` idiom); remove the config-term translation
  workaround while preserving the true-residual `other`.
- [x] 3.4 Add a dataset-free resolution mode for `inspect config` (depends on 1.4):
  skip raw-dataset ingest and the dependent `prepare_outputs` path derivation;
  report `dataset_hash`/`prepared_dataset_hash`/`prepared_dataset_id` and paths
  needing them as deferred; rely on the `HASH_DERIVED` exclusion so the previewed
  `config_hash` matches a real run. Add a test that `inspect config` succeeds with
  the raw dataset absent.

## 4. Fixtures, guards, and docs

- [x] 4.1 Regenerate all golden hash-source fixtures and stored `*_source`
  snapshots (`tests/test_hashing.py`, `tests/fixtures/hash_sources.json`, and any
  sidecar/checkpoint snapshots) against the new mechanics.
- [x] 4.2 Add guard tests: no bare `pathlib.Path` field lacks `SatiPathMarker`; no
  generated identity lacks `HASH_DERIVED`; a portable-path-only config diff leaves
  all behavioral hashes equal.
- [x] 4.3 Add/adjust `config-compare` tests asserting a behavioral diff is never in
  `other` and a shared field appears under every reading hash group.
- [x] 4.4 Update DESIGN-DOC-v1 §4.7 (canonical-JSON policy, `version` convention,
  path handling, weights exception) and any DESIGN-DOC-SYNTH wording quoting recipe
  strings.

## 5. Verification and closeout

- [x] 5.1 Run `pytest tests/` and `ruff check src tests` in the Nix dev shell; fix
  regressions.
- [x] 5.2 Run the `SATI_RUN_TRIPWIRE=1` checkpoint round-trip to confirm embedded
  contract + byte-hash consistency after the metadata `version` change.
- [x] 5.3 Run `openspec validate --all --strict`, reconcile the implemented main
  specs with these deltas, and leave the change ready for archive.
