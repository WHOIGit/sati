## Why

SATI's hash-source construction diverged from its sibling `image-classifier-dojo`
in mechanically significant ways: `config_hash_source` wraps fields under
`{recipe, command, config:{…}}` — a different key namespace than the config
itself — while dojo returns a bare pruned dump sharing the config's namespace.
That nesting is the root cause of the `config-compare` residual bug
(`align-dojo-design-language` finding #4): the `other` bucket could not be
computed as a true residual because covered-path keys never matched full-config
keys. Aligning the hash mechanics now removes this whole bug class, keeps the two
tools predictable siblings, and lets a shared reader reason about either
project's identities the same way — before more artifacts pin the current recipes.

## What Changes

- **BREAKING** Rebuild every SATI hash source in the config's bare key namespace,
  matching dojo: hash sources project directly from the resolved config's field
  names (a pruned/derived `model_dump`) instead of nesting behavior under a
  `config` key or other synthetic wrappers.
- **BREAKING** Replace the inline `recipe: "sati.<name>_hash/<n>"` string marker
  in every source with dojo's `version` field convention, and bump every affected
  recipe/source version. `config_hash` carries no synthetic recipe token.
- **BREAKING** Adopt dojo's canonical-JSON mechanics — a shared `_canonicalize`
  normalizer (recursive key sort, `%.12g` floats, tokenized non-finite values)
  serialized through `json.dumps(sort_keys=True, separators=(",", ":"))` — and
  retire SATI's hand-rolled string builder so both tools hash byte-identically for
  equal source trees.
- **BREAKING** Standardize portable-path handling to one mechanical rule shared
  with dojo (see design for the sentinelize-vs-pop decision), replacing the
  hand-maintained path-exclusion list in `hashing.py`
  (`align-dojo-design-language` finding #6).
- **BREAKING** Simplify `TaskConfig` to `type` only: remove the single-value
  `box_format`, `box_bounds`, and `origin` literals (no code reads them; the
  coordinate convention is a hard code invariant, not a real choice). Encode the
  top-left/half-open/`xyxy` convention as a versioned constant inside the
  `target_schema_hash` source so the hash still flags a future target-contract
  change, without three fake config fields.
- Make the `HASH_DERIVED` marker load-bearing: the source projectors consume it
  to exclude generated identities, instead of the current hard-coded pop lists,
  retiring the inert marker (`align-dojo-design-language` finding #7).
- Make config inspection dataset-free (matching dojo): `sati inspect config`
  SHALL resolve `config_hash`/`config_id` and active output paths without
  ingesting the raw dataset, reporting `dataset_hash`/`prepared_dataset_hash`/
  `prepared_dataset_id` and any path that needs them as deferred
  (`align-dojo-design-language` finding #9). This is only correct *because* the
  `HASH_DERIVED` exclusion above makes `config_hash` independent of dataset
  identities, so inspection previews the same hash a real run produces.
- Preserve every SATI-only domain identity dojo lacks — `target_schema_hash`,
  `dataset_hash`, `prepared_dataset_hash`, `preprocessing_hash`, scope-aware
  `recombination_hash`, and the `checkpoint_hash`/`model_hash` byte identities —
  re-expressed with the aligned mechanics, not removed.
- Update golden hash-source fixtures, result/sidecar/checkpoint schema versions,
  and every stored `*_source` object to the new shapes, with no compatibility
  shims for old recipes or namespaces (SATI's clean-break norm).

Two design decisions are deliberately left open for `design.md` and MUST NOT be
silently resolved here: (1) whether SATI keeps its command-aware `config_hash`
projection — dojo is not command-aware — re-expressed with bare-namespace
mechanics, and (2) whether SATI's in-behavior path fields (`data.source.uri`,
`data.prepared.uri`, `model.weights.uri`) are sentinelized in place or popped as
whole subtrees.

## Capabilities

### New Capabilities
<!-- None: this change re-mechanizes existing identity behavior. -->

### Modified Capabilities

- `results-provenance`: config/model/preprocessing/target/dataset identity source
  projections change shape (bare namespace, `version` field, shared canonical
  JSON, type-driven path handling, `HASH_DERIVED`-driven exclusion) and bump
  their versions; the vocabulary and what each hash gates/identifies is unchanged.
- `config-and-cli`: `inspect config-compare` groups differences by contributing
  hash source in the config's own key namespace, so `other` is a true residual;
  `inspect config` resolves identities and paths without ingesting the raw
  dataset; the strict-schema and path-canonicalization requirements reference the
  shared mechanical rules rather than SATI-private ones.

## Impact

- Hashing core: `src/sati/hashing.py` (all `*_hash_source` functions, canonical
  JSON, recipe/version constants, path handling, coordinate convention as a
  `target_schema_hash` constant), `src/sati/config_schemas/` (`TaskConfig`
  simplification, `HASH_DERIVED` consumption, marker walkers),
  `src/sati/dataset_identity.py`.
- Packaged config: `src/sati/config_defaults/base/default.yaml` drops the
  `task.box_format`/`box_bounds` lines.
- Consumers of stored sources: `src/sati/manifests.py` sidecars,
  `src/sati/checkpoints.py` embedded contract, `src/sati/inspect.py`
  (`config-compare` grouping, hash preview/verify, dataset-free `config`),
  `src/sati/config_loader/resolver.py` (dataset-free inspection mode),
  `src/sati/train.py`, `evaluation.py`, `infer.py`, `export.py` result provenance.
- Every affected digest changes: all golden fixtures under `tests/`
  (`test_hashing.py`, `hash_sources.json`, and any stored `*_source` snapshots),
  plus result/sidecar/checkpoint `schema_version` values.
- Docs: DESIGN-DOC-v1 §4.7 (hash family, canonical-JSON policy, path-exclusion
  and weights-exception prose), and any DESIGN-DOC-SYNTH integration wording that
  quotes the recipe strings.
- No new runtime dependency; the change is mechanical alignment, not new behavior.
  Preserves the v2 seams (zarr/streaming/rectangular-slice preprocessing capture)
  by keeping `preprocessing_hash` and `recombination_hash` semantics intact.
