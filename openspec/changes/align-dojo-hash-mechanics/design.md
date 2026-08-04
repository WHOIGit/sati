## Context

SATI and `image-classifier-dojo` are sibling applications that deliberately
share a design language. Their hash layers drifted apart mechanically. dojo's
`config_hash_source` is a pruned `model_dump` in the config's own key namespace
(`model_dump` → pop `output_root`/`training_outputs`/`eval_outputs` and generated
runtime ids → return). Its other identity sources are small dicts carrying a
`version` field, serialized by a shared `_canonicalize` + `json.dumps` path.

SATI, by contrast:

- wraps `config_hash` behavior under `{recipe, command, config: {…active blocks…}}`;
- embeds a `recipe: "sati.<name>_hash/<n>"` string in every source;
- hand-rolls canonical JSON as a string builder (`canonical_json`);
- strips portable paths with a hand-maintained key list inside
  `config_hash_source` and a `<path>` sentinel, while a declared `HASH_DERIVED`
  marker that was meant to drive exclusion is never consumed.

The wrapper namespace is the direct cause of the `config-compare` residual bug
(`align-dojo-design-language` finding #4): covered-path keys (`config.slicing.…`)
never matched full-config keys (`slicing.…`), so `other` could not be a true
residual. The compare command has since been patched to compute coverage in
config terms, but the underlying hash sources still live in the wrong namespace,
the exclusion list is still hand-maintained (finding #6), and `HASH_DERIVED` is
still inert (finding #7). This change fixes the mechanics at the source.

SATI carries domain identities dojo has no concept of — `target_schema_hash`,
`dataset_hash`, `prepared_dataset_hash`, `preprocessing_hash`, scope-aware
`recombination_hash`, and byte-level `checkpoint_hash`/`model_hash`. These are
not up for removal; they are re-expressed with the aligned mechanics.

## Goals / Non-Goals

**Goals:**

- Every SATI hash source is constructed in the config's bare key namespace, so
  covered paths and full-config paths share one namespace by construction (the
  bug class cannot recur).
- One shared canonical-JSON mechanism and one `version`-field convention across
  both tools; equal source trees hash byte-identically in SATI and dojo.
- Portable-path exclusion and generated-identity exclusion are type-driven
  (`SatiPathMarker`, `HASH_DERIVED`), never hand-maintained lists.
- Clean break: no compatibility shims, no dual-recipe reads; all digests, golden
  fixtures, and stored `*_source` objects move to the new shapes together.

**Non-Goals:**

- Changing *what* any hash gates or identifies, or the hash vocabulary/names
  (except where a name already tracked a recipe string that becomes a version).
- Removing or merging any SATI-only domain hash.
- Altering `preprocessing_hash` / `recombination_hash` semantics or the v2 seams
  (zarr/streaming/rectangular-slice preprocessing capture) they anchor.
- Touching checkpoint/model *byte* hashing, which is file bytes, not source
  projection — it only inherits the shared `version`-field convention where it
  carries metadata.

## Decisions

### D1 — Sources project in the bare config namespace

Every `*_hash_source` returns a dict whose keys are config field names (or a
small derived leaf like `input_shape`), never a synthetic wrapper. `config_hash`
drops the `{recipe, command, config}` envelope; its active-block projection
becomes top-level keys. Rationale: this is the single mechanical property that
makes `config-compare` coverage subtraction correct by construction and matches
dojo's `_leaf_paths(config_hash_source(cfg))` idiom. Alternative (keep the
wrapper, translate keys at compare time) was rejected — it is the workaround the
current patch already applies and leaves the sources structurally foreign to
dojo.

### D2 — `version` field replaces `recipe` strings

Each source dict carries `version: "<n>"` (dojo's convention) instead of an
inline `recipe: "sati.<name>_hash/<n>"`. The hash's identity/name lives in the
function and the sidecar key, not inside the hashed bytes as a namespaced string.
`config_hash` carries no synthetic version token, matching dojo (its identity is
the config content). All affected versions bump on this change. Rationale:
uniform provenance shape across tools; the recipe string was redundant with the
function name and made the two projects' bytes gratuitously different.

### D3 — Shared canonical-JSON mechanics

Adopt dojo's `_canonicalize` (recursive key sort, tuples→lists, `%.12g` floats,
`"NaN"`/`"Infinity"`/`"-Infinity"` tokens) feeding
`json.dumps(ensure_ascii=False, sort_keys=True, separators=(",", ":"))`, prefixed
`sha256:`. Retire SATI's hand-rolled string builder. Rationale: one code path,
byte-identical output for equal trees, less surface for subtle float/escaping
drift. Trade-off: SATI's builder and dojo's normalizer can differ on edge floats
(e.g. `-0.0`, `%.12g` round-trip); the migration re-baselines all goldens, so any
such difference is absorbed once at cutover, not carried as a shim.

### D4 — Command-aware `config_hash` is KEPT, re-expressed in bare namespace

SATI keeps projecting only the active command's behavioral blocks into
`config_hash` (dojo is not command-aware because it has one training shape). The
active-block set is selected first, then emitted as top-level keys. Rationale:
command-awareness is a real SATI feature — a `train` config and an `eval` config
of the same experiment legitimately have different behavioral identities — and it
is orthogonal to the namespace bug. Losing it would over-couple identities across
commands. Alternative (drop command-awareness to mirror dojo exactly) was
rejected as a semantic regression dressed up as mechanical alignment.

### D5 — In-behavior path fields are POPPED as whole subtrees, not sentinelized

Portable paths are removed from sources rather than replaced with `<path>`.
Output blocks are popped wholesale (as dojo does); in-behavior path fields
(`data.source.uri`, `data.prepared.uri`, `model.weights.uri` when it is a local
file) are popped too, driven by the `SatiPathMarker` walk — no hand-listed keys.
Rationale: dojo's mechanic is "absent = not part of identity," which reads more
honestly than a sentinel that implies "a path is here but anonymized," and it
removes the special sentinel value from the hashed bytes entirely. The
`model.weights` byte-substitution (a local checkpoint contributes its
`checkpoint_hash`, a library asset its literal name) is preserved as an explicit,
documented exception because the *bytes* are identity, not the path. Alternative
(keep `<path>` sentinels) was rejected: it kept a SATI-only token in the hash and
still needed the marker walk to know which fields to touch, so popping is strictly
simpler.

### D6 — `HASH_DERIVED` becomes the exclusion driver

Generated runtime identities (`runtime.run_id`, `config_id`, `config_hash`,
`command`) and derived dataset identities carry `HASH_DERIVED`; the source
projectors exclude fields by walking that marker (as they already walk
`SatiPathMarker` for D5), replacing the hard-coded pop lists. Rationale: makes the
marker load-bearing (finding #7), keeps exclusion declarative next to the field,
and guarantees a newly added derived field is excluded without editing
`hashing.py`.

### D7 — `TaskConfig` collapses to `type`; coordinate convention becomes a hash constant

Remove `box_format`, `box_bounds`, and `origin` from `TaskConfig`, leaving only
`type` (matching dojo's single-value `task.type`). No code reads the three
removed fields — the top-left/half-open/`xyxy` convention is hard-coded across
tiling, IoU, affines, and recombination — so as config they are constants
wearing a config costume, implying a configurability that does not exist. The
convention is instead written as a versioned constant in the `target_schema_hash`
source (alongside the existing `structure`/`bbox_dtype` constants), preserving
the hash's purpose: a future output-target change (rotated boxes, segmentation,
closed bounds) still shifts the digest. Rationale: input-format variety
(COCO/YOLO/VOC/LabelStudio) is already absorbed by the adapters keyed on
`data.source.format`; `task.*` describes the internal *target*, which is
invariant in v1. Alternative (keep the three literals as config) was rejected —
a single-value Literal cannot describe variety, so it adds noise to every config
and hash source for no expressive gain. A genuinely richer *input* bbox
convention (e.g. VOC 1-indexed) would belong under `data.source`, not `task`,
and is out of scope here.

### D8 — Config inspection is dataset-free (depends on D6)

`sati inspect config` resolves through a mode that skips raw-dataset ingest and
the dependent `prepare_outputs` path derivation. `config_hash`, `config_id`, and
active output paths still resolve; `dataset_hash`, `prepared_dataset_hash`,
`prepared_dataset_id`, and any path needing them are reported as deferred.
Rationale: dojo already treats config inspection as a pure config operation (its
compare "does not read datasets"); requiring the full dataset just to preview a
config's structure and identities is a usability wart
(`align-dojo-design-language` finding #9). Critically, this is only *correct*
once D6 makes `HASH_DERIVED` exclude the dataset identities from `config_hash` —
otherwise a dataset-free inspection would show a `config_hash` that differs from
the real run (verified: today the two differ). So D8 is sequenced strictly after
D6 within this change. Alternative (a graceful-degradation stopgap that ingests
when data is present and partials when absent) was rejected as the durable answer
because it leaves `config_hash` dataset-dependent and non-previewable; it remains
available only if this change must be split.

## Risks / Trade-offs

- [Every digest changes at cutover] → Intended and bounded: this is a clean
  break. All goldens, sidecar/result/checkpoint `schema_version`s, and stored
  `*_source` snapshots are regenerated in the same change; no artifact produced
  before the change is expected to re-verify after it.
- [Canonical-JSON edge divergence from the old builder] → Re-baseline goldens
  against the new `_canonicalize` path; add a guard test that a fixed tricky tree
  (nested dicts, `-0.0`, non-finite, unicode) hashes to a pinned value shared with
  dojo's helper.
- [Popping in-behavior paths could hide a genuine behavioral difference] →
  Mitigated by D5's `model.weights` byte-substitution exception; a training-input
  change that matters (different local checkpoint) still shifts `config_hash` via
  its byte hash, while only the portable *location* is excluded.
- [Command-awareness kept while dojo lacks it] → Accepted divergence; documented
  as an intentional SATI extension so a shared reader is not surprised. The
  mechanics (namespace, versioning, JSON) still match.
- [Two markers now drive projection] → Add guard tests asserting no bare
  `pathlib.Path` field lacks `SatiPathMarker` and no generated identity lacks
  `HASH_DERIVED`, so the declarative exclusion cannot silently regress.

## Migration Plan

1. Land the shared canonical-JSON + `version` mechanics and the marker-driven
   exclusion helpers first (pure functions, unit-testable in isolation).
2. Rebuild each `*_hash_source` in the bare namespace; bump its version.
3. Regenerate golden fixtures and stored `*_source` snapshots; bump
   result/sidecar/checkpoint `schema_version`s.
4. Update `inspect config-compare` to rely on the now-aligned namespace (coverage
   subtraction collapses back to dojo's `_leaf_paths` idiom; the config-term
   translation patch becomes unnecessary).
5. Update DESIGN-DOC-v1 §4.7 prose and any SYNTH wording quoting recipe strings.

No rollback path within the change: because there are no shims, reverting means
reverting the commit. Artifacts created before cutover remain readable only by
pre-change code.

## Open Questions

- Should the shared canonical-JSON helper live in a small module importable by
  both projects (true single source), or be duplicated with a pinned cross-tool
  guard test? This change assumes duplication-plus-guard-test to avoid a new
  cross-repo dependency; revisit if a shared library emerges.
- (Resolved by D7) `target_schema_hash`'s constant block keeps its structural
  keys and additionally absorbs the coordinate convention removed from
  `TaskConfig`, with a `version` bump.
