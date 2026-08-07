## Context

**Shipped today.** `UltralyticsBackend.predict` (`backends.py:143`) takes
`slices`, `imgsz`, and `confidence_threshold`, and calls
`self._model.predict(bgr, imgsz=imgsz, conf=confidence_threshold, verbose=False)`
(`backends.py:150`). Ultralytics fills in the rest from its own defaults, of which
two are behavioral for SATI: `iou=0.7` and `agnostic_nms=False`. So every slice's
raw detections have already been through a class-aware NMS at 0.7 before SATI's
recombination sees them.

`_ACTIVE_BLOCKS` (`hashing.py:140`) is a per-command allowlist of root-config blocks
that contribute to `config_hash`:

```python
"train": ("experiment", "task", "runtime", "data", "slicing", "model", "training"),
"eval":  ("experiment", "task", "runtime", "data", "slicing", "recombination", "results"),
"infer": ("experiment", "task", "runtime", "slicing", "recombination", "results"),
```

A block absent from a command's tuple contributes nothing to that command's
`config_hash`. This is the existing exclusion mechanism for command-scoping, distinct
from the type-driven `SatiPathMarker` / `HashDerivedMarker` exclusions at
`hashing.py:158`.

**Measured relevance.** On
`runs/v1/combo1/default/2026-08-06_22-58-09_normal-galago_infer/`, 347 of 369
duplicate detection pairs were **cross-class** — a high-confidence class-1 box
shadowed by a class-2 or class-3 ghost at 0.05–0.22. Class-aware per-slice NMS is
what lets those pairs through in the first place; `agnostic_nms=True` would suppress
many at the source. `recombination-match-and-merge` handles them at recombination
time instead, which is the better place (it is provenance-aware and hashed), but the
existence of a second, invisible lever on the same phenomenon is itself the problem.

## Goals / Non-Goals

**Goals:**

- Both values are explicit configuration and recorded in every run's config bundle.
- Defaults reproduce today's behavior exactly, so this change alone moves no
  detection.
- The decision *not* to hash them is explicit, greppable, and guarded against
  accidental reversal.
- The reproducibility gap that remains is stated plainly rather than implied.

**Non-Goals:** hashing `prediction`; folding it into `preprocessing_hash`; exposing
other predict arguments; changing any default; retuning `iou`.

## Decisions

### D1 — A new `prediction` block, not fields on `model` or `recombination`

`model` describes the architecture and weights and is hashed for `train`.
`recombination` describes what SATI does to model outputs across slices and levels,
and is hashed for `infer`/`eval`. Per-slice NMS is neither: it happens inside the
backend, on one slice, before any SATI merging. Putting it in either block would both
mislabel it and — because both blocks are hashed for some command — force the
hashing decision (D2) rather than leaving it open.

A separate `prediction` block also makes the eventual `prediction_hash` (if it is
ever wanted) a clean scoping problem, mirroring how `recombination_hash` scopes over
`recombination`.

### D2 — Excluded from hashing by omission from `_ACTIVE_BLOCKS`, with a guard

`prediction` is added to `RootConfig` but to **no** entry in `_ACTIVE_BLOCKS`. It is
therefore absent from every `config_hash` projection with no new marker, no new
walker, and no version bump.

Two alternatives were considered and rejected:

- **A new `UnhashedBehaviorMarker`.** More explicit and greppable, and it would make
  the exemption visible at the field rather than by absence from a tuple elsewhere.
  Rejected because it adds a third marker type and a fourth exclusion path for two
  fields, and because a marker reads as permanent when this decision is explicitly
  "for now".
- **Adding it to the tuples and bumping the hash version.** That is the eventual fix,
  not this change.

The real hazard with omission is that it looks like an oversight. `align-dojo-hash-mechanics`
exists partly because hand-maintained exclusion lists were the bug class it removed.
Mitigations, both required:

1. A comment at `_ACTIVE_BLOCKS` naming `prediction` as intentionally omitted, with
   the reason and a pointer to this change.
2. A guard test asserting `prediction` appears in no `_ACTIVE_BLOCKS` entry. If
   someone later decides to hash it, that test fails loudly and forces the version
   bump to be deliberate.

### D3 — Unhashed but recorded: what actually improves

The distinction that matters, because it is easy to overclaim this change:

| | before | after |
| --- | --- | --- |
| value visible in the run directory | no | **yes** (`config/resolved.yaml`) |
| value adjustable without editing code | no | **yes** |
| value covered by `config_hash` | no | no |
| two runs differing only here are distinguishable by digest | no | no |

Rows 1 and 2 improve. Rows 3 and 4 do not. So this change converts an *invisible*
dependency into a *documented* one; it does not make provenance complete. That
limitation is recorded in the spec, not just here, so a reader of the hash family
learns it from the contract rather than from the change log.

### D4 — Defaults mirror the library, not SATI's preference

`iou: 0.7`, `agnostic_nms: False`. These are Ultralytics' current defaults, so a run
before and after this change produces identical detections and identical digests.
The alternative — defaulting `agnostic_nms: True` because it would suppress cross-class
ghosts at the source — is a behavioral change dressed as a refactor, and it would
also make `recombination-match-and-merge`'s measurements non-comparable. If
`agnostic_nms: True` is wanted, it should be a separate, measured decision.

Consequence worth stating: pinning `iou: 0.7` in SATI's config **freezes** the value
against Ultralytics drift. That is a real robustness gain independent of hashing — the
library can change its default without silently changing SATI's behavior.

### D5 — Protocol widening, not a new method

`Backend.predict` gains two keyword parameters rather than growing a config-object
argument. The protocol has one implementation and its existing shape already passes
`imgsz` and `confidence_threshold` as scalars; matching that keeps the seam narrow
and keeps the protocol free of SATI config types, which is what makes v2's alternate
backends plausible.

## Risks / Trade-offs

- **Omission still looks like an oversight**, mitigated but not eliminated by D2's
  comment and guard test. Anyone auditing "is every behavioral field hashed?" will
  find one place where the answer is no, on purpose.
- **The gap is real while it lasts.** Two runs with identical hashes can produce
  different detections. Any comparison across runs must check `resolved.yaml` for
  `prediction` rather than trusting digests alone — the opposite of what the hash
  family is designed to allow. This is the cost of the deferral, and the guard test
  keeps it from being forgotten.
- **A `prediction_hash` later is a break, not an addition.** When these become
  hashed, every existing digest for `infer`/`eval` moves. Deferring makes that break
  later and larger than doing it now.
- **Freezing `iou: 0.7`** locks in a value chosen by Ultralytics for natural-image
  benchmarks, not for dense small objects in gigapixel imagery. Making it visible
  invites tuning it; nothing here measures whether it should change.

## Migration Plan

No migration. Defaults reproduce current behavior, no hash moves, no schema version
changes, and no artifact layout changes. Authored configs that omit `prediction` get
the defaults.

Independent of `recombination-match-and-merge` and `infer-render-artifacts`; any
order.

## Open Questions

1. **Should `agnostic_nms: True` become the default after measurement?** It would
   suppress cross-class ghosts at the slice rather than at recombination, which is
   cheaper and arguably more correct — SATI's recombination is now class-agnostic by
   default anyway, so leaving them class-aware per-slice only to collapse them later
   is doing the work twice. Deliberately not decided here (D4). Measurable directly:
   run `combo1b/default` with `agnostic_nms` both ways and compare final detection
   counts and mAP.
2. **Is `iou: 0.7` appropriate for this imagery?** It is Ultralytics' natural-image
   default. Dense ichthyoliths at 200x may want it lower (more aggressive per-slice
   suppression) or higher (defer all deduplication to SATI's provenance-aware
   recombination, which can record what it removed). The second is more in keeping
   with the project's design; nothing has measured it.
3. **When should `prediction` become hashed, and as what?** Adding it to
   `_ACTIVE_BLOCKS` is the minimal move but makes it part of `config_hash`, which is
   command-scoped rather than stage-scoped. A `prediction_hash` stamped on
   `slice_prediction` rows would match how `recombination_hash` is scoped per merge
   stage and would let a cached slice-prediction table be validated against the
   parameters that produced it. That is the better design and a larger change.
