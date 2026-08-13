## Context

Shipped behavior: `apply_inference_contract` (`src/sati/checkpoints.py`) reassigns
`cfg.slicing = SlicingConfig.model_validate(contract.slicing)` wholesale, then asserts
`preprocessing_hash(cfg.slicing) == contract.hashes["preprocessing_hash"]`. Because the
assertion re-hashes the object it just assigned, it is a tautology and can never fail
on an override. `infer`, `eval`, and `export` all route through this function.

Shipped behavior on the consuming side: `infer_frame` (`src/sati/infer.py:123`) calls
`plan_frame(fullres_wh, cfg.slicing)` and `build_level_pyramid(pixels, plan.levels)`
per frame, from in-memory pixels. `execute_eval` (`src/sati/evaluation.py:226`) calls
the same `infer_frame` on full-res pixels read from `source_uri`, using the prepared
product only for split membership, ground-truth annotations, and class mapping — never
for slices. So both commands already re-slice at runtime; neither reads prepared slice
files. Nothing in the pipeline requires the pyramid to match the trained one.

The constraint that does bind: `model.imgsz` is derived from
`slicing.nominal_slice_size` when unset (`src/sati/config_loader/resolver.py:119`), and
`apply_inference_contract` pins `imgsz` from the contract. Emitted slices are always
`spec.emitted_wh`, driven by `nominal_slice_size`. Change `nominal_slice_size` at infer
time and the pipeline cuts tiles at one size while telling Ultralytics another.

The current spec (`inference-evaluation-export`, "Recombination override precedence")
says infer/eval "SHALL reject changes to model architecture, class mapping, target
schema, or preprocessing." The implementation neither rejects nor honors — it discards.
This change resolves that in the direction of honoring the safe subset.

## Goals / Non-Goals

**Goals:**

- Make `slicing.level_policy.*`, `slicing.overlap_ratio`, and `slicing.max_levels`
  effective as inline overrides on `infer` and `eval`, using the existing dash-free
  override grammar with no new flag.
- Make a discarded override impossible: every field of `SlicingConfig` is either
  honored and recorded, or refused by name.
- Keep a re-sliced run self-describing — a reader of the results parquet must be able
  to tell the trained pyramid from the one actually used, without the run directory.
- Leave `preprocessing_hash`'s source and version untouched so existing checkpoint
  contracts keep verifying.

**Non-Goals:**

- Re-slicing at `prepare` or `train` time. The prepared product remains the training
  pyramid's; this is an inference-time question only.
- Changing `sati export`. Export metadata SHALL always describe the trained pyramid
  from the checkpoint contract, regardless of any slicing override supplied to the
  `export` invocation — an exported inference config means the trained schema, full
  stop (Sidney, decided). This closes the question the earlier draft deferred.
- Any claim about accuracy. Re-slicing shifts the apparent object-size distribution;
  whether that helps is the measurement this unblocks, not something the design asserts.
- Migrating existing `runs/**/results/*.parquet`. Per the root `AGENTS.md`
  no-backwards-compatibility rule for v1 development artifacts — accepted explicitly,
  including for artifacts already on disk (Sidney, decided).

## Decisions

**D1 — Split `SlicingConfig` by model coupling, not by convenience.**
Pinned at infer/eval: `nominal_slice_size` (derives `imgsz`; a mismatch is a real
defect) and `image_format`. Honored: `overlap_ratio`, `level_policy`, `max_levels`.
Untouched and out of scope: `min_area_ratio`, `size_gate_px` — prepare-time annotation
projection gates (`src/sati/tiling.py:331`), absent from `preprocessing_hash` entirely.

`image_format` is pinned despite having no infer-time effect at all — it is read only
at `src/sati/materialize.py:188` when writing slice files during prepare, and infer
writes no slice files. It is pinned rather than honored because honoring a field that
changes nothing would put a value in `effective_preprocessing_hash` that misrepresents
the run. *Alternative considered:* drop it from the pinned set and let it float. Rejected
— it would let two runs with identical pixels carry different effective hashes.

**D2 — Refuse on divergence by value, not by tracking override provenance.**
`apply_inference_contract` compares the resolved `nominal_slice_size` / `image_format`
against the contract's and raises a named refusal when they differ. This deliberately
does not distinguish "user passed an inline override" from "the authored config
disagrees with the checkpoint" — both are genuine incompatibilities and both should
stop the run. *Alternative considered:* thread override provenance from the Hydra layer
down to the contract check, refusing only on explicit overrides. Rejected — it adds a
channel through `ConfigResolution` for a distinction with no operational meaning, and
it would silently accept a config/checkpoint mismatch.

**D3 — Reuse `preprocessing_hash` over runtime slicing; do not invent a hash source.**
`effective_preprocessing_hash` is `preprocessing_hash(cfg.slicing)` — the same function,
the same `PREPROCESSING_HASH_VERSION`, the same inclusions (`nominal_slice_size`,
`overlap_ratio`, `level_policy` via `level_policy_hash_source`, `max_levels`,
`stretch_policy`, `image_format`, optional `level_stack`) and the same exclusions
(`min_area_ratio`, `size_gate_px`, every non-slicing block), applied to the resolved
runtime config instead of the contract's copy. When nothing is overridden the two
hashes are equal by construction.

*Alternative considered:* a new `slice_geometry_hash` covering only the three
overridable fields. Rejected on two counts — it needs its own version stamp and source
definition for no added discrimination, and a reader comparing
`effective_preprocessing_hash` to `preprocessing_hash` gets the "was this re-sliced?"
answer from a single equality, whereas a partial hash would answer it only in
combination with the pinned fields.

**D4 — Record on rows, not only in the sidecar.**
`effective_preprocessing_hash` becomes a non-nullable column on the detection results
schema, alongside the existing provenance columns, and appears in the metadata sidecar.
Rows are what survive being copied out of a run directory, and the whole defect being
fixed is a run whose artifacts misstate their own preprocessing. *Alternative
considered:* sidecar only. Rejected — a detached parquet would again be unable to say
which pyramid produced it.

**D5 — Keep `preprocessing_hash` as the trained-pyramid compatibility identity.**
Its meaning is unchanged: the pyramid the model was trained on, carried in the contract,
reported on rows. The contract's internal consistency is still verified, but against a
`SlicingConfig` reconstructed from the contract rather than against the mutated `cfg` —
which restores the check's ability to fail. This is the fix to the tautology, separate
from the override feature.

**D6 — Warn whenever the runtime stack differs from the trained one; never refuse.**
A geometry override is always honored (D1), but when the planned runtime level stack is
not equal to the trained stack, the command SHALL emit a warning naming both stacks
before the first forward pass. Recall can shift under a different pyramid, and the shift
is quiet and easy to misattribute to the model or the data; a warning makes it
attributable without taking the decision away from the operator (Sidney, decided).

Divergence is evaluated on the planned level scales, not on the policy fields, because
two different policies can produce the same stack and identical policies can produce
different stacks on differently-sized frames. Comparing planned stacks is what lets a
policy rewrite that plans an identical pyramid stay correctly silent.

**Any** difference warns; partial overlap is not a reason to stay quiet. An earlier
draft fired only when the two stacks shared *no* level scale, on the theory that an
overlapping pyramid was close enough to be unremarkable. That was wrong in both
directions and is superseded (Sidney, 2026-08-12):

- *False negatives, the serious half.* Stacks can share several scales and still be
  materially different. Planned against one frame, `1.0/2.0` gives
  `(1.0, 0.5, 0.25, 0.125)` and `0.5/2.0` gives `(0.5, 0.25, 0.125)` — three shared
  scales, yet the first carries a full-resolution level the second has no equivalent
  for. Of the four policies `0.5/2.0`, `0.5/4.0`, `1.0/2.0`, `1.0/4.0`, five of the six
  pairs share at least one scale, so the old rule silenced nearly all of them.
- *False positives.* Two ladders that interleave closely but coincide nowhere still
  fired: against a trained `0.4/3.0`, a runtime `0.5/2.0` produced a warning although
  `0.124962` and `0.133317` sit 1.07× apart, and the measured eval was slightly better,
  not degraded.

Exact set intersection on floats was never a proxy for "meaningfully different." Stack
equality is, it needs no threshold, it cannot silently miss a divergence, and it matches
the warning text exactly — so a run either warns or ran the trained pyramid, with no
third state to reason about. *Alternative considered:* a tunable ratio threshold
(warn when every runtime scale is >1.5× from its nearest trained scale). Rejected — it
reintroduces a tuning knob to soften a diagnostic that costs one line per deliberate
override, and "roughly similar coverage" is not the question the operator asked.
*Alternative considered:* refusing beyond some bound. Rejected by decision — the whole
point of the change is to let the operator ask questions the checkpoint cannot answer
for itself.

**D7 — Export metadata always describes the trained pyramid.**
`sati export` continues to reconstruct slicing wholesale from the contract, and its
portable inference config records the trained geometry even if the invocation carried a
slice-geometry override. An export is a deployable artifact whose preprocessing must
match what its consumer will reproduce; a re-sliced geometry is an experimental finding,
and promoting one into an export is a retraining decision, not an export flag (Sidney,
decided). Because export takes no honored geometry override, an override supplied to
`export` is inert — D8 covers making that audible rather than silent.

**D8 — An inert override is still not a silent one.**
The change's core invariant is that no slicing override is ever discarded without saying
so. Two paths lead to a field not taking effect, and each gets its own audible outcome:
a model-coupled divergence (`nominal_slice_size`, `image_format`) raises
`PreprocessingOverrideRefused` on every command including `export` (D2, unchanged); a
geometry override supplied to `export` — where geometry is pinned by D7 — emits a warning
naming the ignored fields. Neither path may reach a forward pass having quietly restored
a contract value.

## Risks / Trade-offs

- **A re-sliced eval's mAP is silently non-comparable to the trained-pyramid eval's.**
  → The two runs differ in `effective_preprocessing_hash`, and `sati inspect
  config-compare` surfaces it; the comparison discipline stays with the operator, as it
  already does for recombination overrides.
- **Detections may degrade sharply at pyramids far from the trained one.** → D6 warns
  when the runtime stack shares no level scale with the trained one, which is the case
  most likely to be misread as a model or data fault. Between that trigger and an
  unchanged pyramid the operator gets no signal, by design — the degradation is the
  quantity under measurement, and `n_final_merge` plus the existing scale diagnostics
  make a collapse visible in the eval table.
- **A warning that fires on every run trains the operator to ignore it.** → The trigger
  is the weakest useful one (no shared level scale), so an ordinary denser or sparser
  pyramid overlapping the trained stack stays silent. If it turns out to fire routinely
  in practice, the trigger is wrong and should be narrowed rather than muted.
- **Adding a non-nullable results column breaks existing parquet readers.** → Accepted
  and marked BREAKING in the proposal; covered by the v1 no-compatibility rule.
- **`inspect.py:101` currently hard-codes the compatibility-projection field set** as a
  single flat set; splitting it into pinned and geometry groups changes
  `config-compare` output shape for every caller, not just re-sliced runs. → Contained
  to one module, but the spec delta must state the new grouping so the change is not
  mistaken for a cosmetic diff.
- **`max_levels` interacts with `level_policy.stop_when_long_side_lte_nominal`.** A
  denser policy plus an unchanged `max_levels` can truncate the stack in a way that is
  easy to misread as a policy bug. → Level indices are already recorded per row, and
  level 0 is always full-res, so a truncated stack is visible in the results rather
  than inferred.

## Migration Plan

No data migration. Existing checkpoints are unaffected — `preprocessing_hash` keeps its
source and version, so every embedded contract verifies exactly as before. Existing
result parquet files predate the new column and are not rewritten; re-run the command to
produce a conforming artifact. Rollback is reverting the change: unoverridden runs
produce identical hashes before and after, so nothing downstream is pinned to the new
behavior except artifacts written while it was live.

## Open Questions

Resolved during proposal (Sidney):

- Export geometry — settled by D7. Export metadata always describes the trained pyramid.
- Model-coupled divergence — settled by D2, unchanged. `nominal_slice_size` and
  `image_format` hard fail; they do not warn.
- Pyramid divergence — settled by D6. Warn, never refuse.
- Existing and future `results/*.parquet` diverging on the new column — accepted, no
  migration.

Still open:

- Should `eval` refuse to write into a run tree that already holds a trained-pyramid
  eval under the same `experiment.name`, or is the timestamped run directory plus the
  differing effective hash sufficient separation? Current design assumes sufficient.
- D6's trigger is a first guess, not a measured boundary. Once there are re-sliced runs
  to look at, check whether "shares no level scale" fires where it should and stays
  quiet where it should; the alternative shapes (scale-ratio bound, count of shared
  levels) are cheap to swap in and were rejected only for lack of evidence.
