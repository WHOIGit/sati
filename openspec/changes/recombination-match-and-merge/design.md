## Context

**Shipped today.** `sati.tiling` recombines in two stages behind a single hard-coded
strategy. `greedy_nms` (`tiling.py:379`) sorts by score, and for each surviving
box suppresses every lower-scored box whose IoU **strictly exceeds** the stage
threshold, skipping other classes unless `class_agnostic`. The winner's own box,
score, and class become the output row; the suppressed boxes contribute only
their `slice_id` (Stage 1) or `level` (Stage 2) to a provenance list. Strategy
identity is the pair of module constants `MERGE_STRATEGY_NAME = "nms"` /
`MERGE_STRATEGY_VERSION = 1`, imported by `infer.py` and stamped on both merge
row types. `recombination_hash` is scope-aware: within-level params on
`level_merge`, within-level plus cross-level on `final_merge`.

`DESIGN-DOC-v1.md:491` records the known bias this leaves: a near-slice-edge
fragment can outrank a whole detection, and split duplicates dedup purely by IoU.
The documented mitigation is to raise `overlap_ratio` — which does not work,
because overlap changes *whether* a whole detection exists, not whether IoU can
match the fragment to it.

**Measured.** On `runs/v1/combo1/default/2026-08-06_05-15-58_elated-salmon_infer/`
(154 `final_merge` rows from one 21117×9597 frame), 22 pairs exceed IoS 0.5:

```
16 pairs  different class, area ratio 0.86–1.00, IoU 0.78–1.00   → class-aware matching
 6 pairs  same class,      area ratio 0.31–0.48, IoU 0.31–0.48   → IoU cannot reach these
                                                  IoS 0.96–1.00
```

Five of the six nested pairs preserve one axis and cut the other, confirming
slice-window bisection rather than a model localization error. In two of the six
the fragment scores *higher* than the whole box.

The geometric root cause, with `nominal=1024`, `overlap_ratio=0.2` (stride 820)
and a 100×100 object at level x ∈ [1000, 1100]:

```
   0            820        1024      1100
   ├─ slice A (0..1024) ─────┤
                ├─ slice B (820..1844) ──────────┤

   A sees  [1000,1024)  clipped   area  2,400
   B sees  [1000,1100)  whole     area 10,000

   IoU = 2400/10000 = 0.24   ✗ under any usable threshold
   IoS = 2400/2400  = 1.00   ✓
```

A clipped fragment is a strict subset of its whole self, so its IoU against that
whole box equals the area ratio. IoU is not a mis-tuned threshold here; it is the
wrong metric. This is why SAHI defaults `postprocess_match_metric="IOS"` alongside
`postprocess_type="GREEDYNMM"`.

## Goals / Non-Goals

**Goals:**

- Both observed duplicate classes are eliminated by configuration that is
  hashed, recorded, and independently ablatable.
- A merged box is correct regardless of which cluster member won the score
  ordering, so near-ties stop being coin-flips on output geometry.
- Class-agnostic collapse never destroys the fact that classes disagreed.
- One flat config namespace that v2's nested per-stage shape can supersede in a
  single later migration, not two.
- Strategy identity in the results table is a real strategy name with a
  per-strategy version, not a global constant that lies when a sibling strategy
  changes.

**Non-Goals:**

- Non-greedy `nmm`, `lsnms`, WBF, cross-level box averaging.
- Favorability scoring, `near_edge_count`, edge penalties. Stage 2 keeps ranking
  by raw confidence.
- Per-stage divergent method/metric.
- Changing the stage boundaries, the confidence pre-filter, the coordinate
  convention, or `recombination_hash` scoping.
- Exposing Ultralytics' per-slice NMS parameters (see proposal Non-Goals).

## Decisions

### D1 — Flat, stage-shared `method` and `match_metric`; not per-stage

Both stages need the same treatment: the 16 cross-class pairs are within-level
duplicates, and one of the six nested pairs has `source_levels (1,2)` against
`(1,)`, so cross-level union is load-bearing too. No measurement supports
divergence. v2 §5 will need `cross_level.method: favorability`, whose value space
differs from `within_level.method`; adopting the nested shape now would mean
migrating authored configs twice. Flat now, nested once, later.

### D16 — The default method is `nms`; merging is an option

Measured across both reference frames, `nms` and `greedy_nmm` at IoS 0.8
class-agnostic produce the **identical detection count** (402 level / 376 final on
frame 2). Every measured reduction — 664 to 376, a 43% cut — came from the match
metric and class-agnostic matching, not from merging. Merging's only demonstrated
effect on geometry was one 248x winner-box inflation (D14), plus restoring the
fragment's whole box in the minority of nested pairs where the fragment outranks it.

A default should not carry an unproven risk. `greedy_nmm` therefore ships selectable
and fully specified, and `combo1b/greedy_nmm` (D15) exists to measure whether it earns
promotion on mAP. `max_merge_growth` is inert under the default.

This makes the change's real content **configurable matching** — IoS and
class-agnostic — with optional merging layered on top, which is why it is named
`recombination-match-and-merge` rather than for NMM.

### D2 — The method value is `greedy_nmm`, not `nmm`

What is being built is SAHI's GREEDYNMM: score-ordered, single pass, winner
absorbs its own matches, absorbed boxes leave the pool. `DESIGN-DOC-v2.md:247`
reserves `nmm` for the non-greedy variant (transitive closure over the match
graph). Taking the shorter name for the greedy algorithm would burn the name and
make the v2 doc wrong.

**Why the non-greedy variant is deferred, not merely unbuilt.** Non-greedy NMM is
single-linkage clustering; a cluster's union is bounded by nothing, and IoS
returns ≈1.0 for *any* nested box, so links form readily:

```
   object A          spanner           object B
   score 0.9         score 0.5         score 0.8
   ┌────────┐   ┌──────────────────┐   ┌────────┐
   │        │   │                  │   │        │
   └────────┘   └──────────────────┘   └────────┘
        └── IoS≈1.0 ──┴── IoS≈1.0 ──┘
        (A and B do not match each other)

   greedy:      A wins, absorbs spanner; B untouched  → 2 detections
   non-greedy:  A—spanner—B is one component          → 1 box over both objects
```

Greedy bounds every union to boxes matching *the winner*; non-greedy bounds it to
the transitive closure, whose diameter can exceed any threshold. Empirically on
the reference run, connected components equal greedy clusters at every stage at
both IoS 0.5 and 0.3 (max cluster size 2), so the risk buys no measured benefit.
Non-greedy only wins for objects spanning 3+ slices where end fragments miss the
middle winner; the largest observed box is 351×819 against nominal ≈1024.

Caveat on that measurement: the reference run has `cache_slice_predictions: false`,
so it contains no `slice_prediction` rows and the comparison ran on post-merge
populations, not Stage 1's true input. The structural argument still holds —
same-object duplicates mutually match, so greedy and non-greedy agree; divergence
requires *distinct* objects linked through an intermediary — but a run with slice
caching enabled would settle it by measurement. Recorded as the trigger for
revisiting.

### D3 — Metric values spelled out: `intersection_over_union` | `intersection_over_smaller`

`ios: 0.5` in a YAML file reads as the Apple OS. The repo's naming is already
verbose (`stop_when_long_side_lte_nominal`, `subsequent_downsample_factor`), so
spelling both out is consistent and unambiguous in hash sources too.

### D4 — Rename both thresholds

`within_level_iou` with `match_metric: intersection_over_smaller` is a field name
that lies. `within_level_threshold` / `cross_level_threshold` are metric-neutral.
Cost is four YAML files plus the `DESIGN-DOC-v1.md:152` table; `extra="forbid"`
turns any stale authored config into an immediate validation error rather than a
silent default. No alias is provided.

### D5 — `class_agnostic` stays a knob, default `true`

Agnostic matching is what eliminates 16 of the 22 observed pairs, and because
both stages are score-ordered greedy, "most confident class wins" is already the
selection rule — no new tie-break logic. The flag is retained because turning it
off is how the failure mode is *measured*; removing it would delete the ablation.

Every **existing** authored experiment config
(`v1/manual/baseline.yaml`, `v1/combo1/default.yaml`, `v1/sam2dirty/start_zoom.yaml`)
takes the mechanical field rename and is then **explicitly pinned to pre-change
values** (`method: nms`, `match_metric: intersection_over_union`,
`class_agnostic: false`, thresholds `0.5`). No existing experiment changes
behavior as a side effect of a default flip; the new defaults are adopted by
exactly one new config (D13).

### D12 — Default thresholds are 0.8, paired with the metric

`0.5` on intersection-over-smaller is substantially *looser* than `0.5` on
intersection-over-union was, and over-merging is this change's entire risk
surface. All 22 duplicate pairs measured on the reference run sit at IoS ≥ 0.93,
so `0.8` catches every observed case with a wide margin while materially
shrinking the nested-object exposure. Both `within_level_threshold` and
`cross_level_threshold` default to `0.8`.

The threshold default is therefore metric-coupled in intent but a plain constant
in the schema: `0.8` is correct for the default metric, and a config that selects
`intersection_over_union` SHALL set `0.5` explicitly (as every pinned pre-change
config does). A validator coupling the two is deliberately not added — it would
encode a tuning choice as a schema constraint.

**Confirmed on a second frame.** On
`runs/v1/combo1/default/2026-08-06_22-58-09_normal-galago_infer/` (19321×15948,
664 shipped final detections), 369 duplicate pairs exceed IoS 0.5 and **every one
of them exceeds 0.8** — zero fall in the 0.5–0.8 band that a `0.8` default would
silently under-merge. Two frames, 391 pairs, none in the risk band.

### D6 — Merge semantics: envelope, max score, single pass over original boxes

A merged detection's box is the axis-aligned envelope
(`min x1, min y1, max x2, max y2`) of the winner and every box matched to it. Its
score and class are the winner's. Matching is evaluated against the **original**
winner box, once — the box that grows during merging does not re-match. This is
SAHI's behavior and it keeps the stage a single deterministic pass.

Consequence, to be documented rather than engineered around: `recombine` is
**not idempotent**. Feeding `final_merge` boxes back through the same stage can
merge further, because unioned boxes overlap more than their inputs did. Nothing
in the pipeline does this; tests SHALL NOT assert idempotence.

Suppression mode (`nms`) is unchanged: the winner's own box survives. The two
modes differ only in the output geometry, never in which clusters form, so
`nms` remains an exact ablation of the merge decision alone.

Envelope growth is bounded — see D14.

### D14 — Merging may not grow the winner's box beyond `max_merge_growth`

**Deviation from SAHI, adopted on measured evidence.** SAHI unions
unconditionally. SATI SHALL take the envelope only when its area is at most
`max_merge_growth` times the winner's own area; otherwise the winner keeps its
own box. The matched boxes are consumed either way, so the cluster still
deduplicates and the detection count is unaffected — a blocked merge degrades
that one cluster to `nms` behavior, which is the safe fallback. Default `4.0`.

**Why.** On the second reference frame
(`runs/v1/combo1/default/2026-08-06_22-58-09_normal-galago_infer/`, 19321×15948,
1427 cached slice predictions), a 2658×1206 false-positive blob at score 0.510
geometrically contains many genuine small detections. IoS against a 137×97 real
detection is 1.000 while IoU is 0.002, so class-agnostic IoS matching clusters
them — and because the small box scores *higher* (0.933 > 0.510) it wins, and an
unbounded union replaces its 137×97 box with the whole blob:

```
  simulated on real slice predictions, ios 0.8, class-agnostic
  ───────────────────────────────────────────────────────────────
  variant                lvl  final  max winner-box inflation
  uncapped union         402    376              248.3×
  max_merge_growth 4.0   402    376                2.7×
  max_merge_growth 2.0   402    376                1.9×
```

Detection count is identical at every cap; `4.0` loses exactly one merge out of
299, the pathological one.

**Settled at `4.0`** on current evidence; open question 1 tracks confirmation.

**Why `4.0` and not `2.0`.** A slice-bisected fragment that wins
the score ordering needs inflation of `1 / area_ratio` to be restored to its whole
object. The first reference frame's nested pairs sit at area ratio 0.31–0.48, i.e.
they require **2.1×–3.2×**. A cap of `2.0` would block the exact case this change
exists to fix. Observed legitimate merges need ≤3.2×; the observed pathological
merge needed 248×. Any value in 4–10 separates them; `4.0` is the tight end of
that range.

**Consequence to accept:** a fragment smaller than a quarter of its object is not
restored — it is suppressed instead. That is still strictly better than today
(where it survives as a duplicate) and still yields the correct box whenever the
whole detection wins the ordering.

`max_merge_growth` is inert under `method: nms` — which is now the default (D16) — and
SHALL still be included in the `recombination_hash` params regardless, so a run's
identity always records the bound that would have applied. The alternative, omitting
it when inert, makes the param set method-dependent and the hash harder to reason
about for no benefit.

### D7 — Match ordering and determinism

Unchanged from today: rank by `(-score, input_index)`, iterate matched candidates
in ascending index order. Contributor lists (`source_slice_ids`, `source_levels`,
`merged_class_ids`) are sorted and de-duplicated. Byte-identical output for
identical input is preserved.

### D8 — The strict comparison stays strict

Matching remains `metric(a, b) > threshold`. A perfectly bisected half gives
IoU exactly 0.5 and does **not** match at threshold 0.5 — with IoS this is moot
(1.0), but the boundary is now a recorded decision rather than an accident, and a
test pins it.

### D9 — `merge_strategy_name` is the method alone; version is per strategy

The Parquet column carries `nms` or `greedy_nmm` — a readable strategy identifier.
The match metric, thresholds, confidence filter, and `class_agnostic` live in the
`recombination_hash` params, where a change already moves the digest. Encoding
them into the name (`greedy_nmm_ios_0.5`) would duplicate the hash's job in a
string.

Strategy version becomes a per-strategy value from a registry
(`{"nms": 1, "greedy_nmm": 1}`) rather than one module-level integer. A global
bump would misrepresent `nms` as having changed when only its sibling was added.

### D10 — Hash source inclusions and exclusions

`recombination_hash_full_source(recombination, scope)` keeps its existing shape
(`version`, `strategy`, `strategy_version`, `scope`, `params`) and its scoping
rule. Changes to `params`:

**Included, both scopes** — `confidence_threshold`, `class_agnostic`,
`match_metric`, `max_merge_growth`, `within_level_threshold`.
**Included, `final_merge` only** — `cross_level_threshold`.
**Carried outside `params`** — `strategy` = the resolved method name,
`strategy_version` = that strategy's registry version.
**Excluded** — nothing new. No paths, no generated identities, and no
`SatiPath`/`HASH_DERIVED` fields exist on `RecombinationConfig`, so the
type-driven exclusion walk is a no-op here and stays that way.

`method` is deliberately *not* duplicated into `params`; it is already the
`strategy` field. `RECOMBINATION_HASH_VERSION` goes `"3"` → `"4"` because the
param key set changes and two keys are renamed, so old digests are not
reproducible under the new source builder.

### D11 — `merged_class_ids` semantics

A list column on `level_merge` and `final_merge`: every distinct `class_id` in
the cluster, **winner's class first**, remaining classes ascending. A cluster of
one writes a single-element list, not a null — so "no disagreement" and "column
absent" stay distinguishable. Written in both `nms` and `greedy_nmm` modes and
under class-aware matching (where it is always single-element), because a column
whose presence depends on config is a schema the reader cannot rely on.

It mirrors `source_slice_ids` / `source_levels`, which already record what a
winner absorbed. Scores of absorbed boxes are **not** recorded: that is a
`slice_prediction`-level question already answered by enabling
`results.cache_slice_predictions`.

**Non-merge row types.** The column lives in the single shared table schema, so
its presence is not conditional; the existing typed-null rule fills it. On
`frame_metadata` and `slice_prediction` rows it SHALL be **null**, not a
one-element list — a raw per-slice prediction has been through no merge, and
`[class_id]` there would claim one happened. This distinction is exactly why a
merge row's single-member cluster writes `[class_id]`: within merge rows, "cluster
of one" and "not a merge row" must remain different values.

### D13 — New behavior lands in a new experiment: `v1/combo1b`

The new defaults are adopted by new configs under
`configs/experiment/v1/combo1b/`, rather than by editing `v1/combo1/default.yaml` in
place (D5 pins that to pre-change values). combo1b is combo1 with only the
recombination block changed.

Two consequences worth stating:

- **No retraining.** Recombination is infer/eval-time policy — the
  `inference-evaluation-export` override requirement permits changing it while
  rejecting architecture/class-mapping/target-schema/preprocessing changes. combo1b
  reuses combo1's checkpoint and re-runs `infer`/`eval` only.
- **combo1b references combo1's split file**
  (`configs/experiment/v1/combo1/all_fullframes_split.toml`) rather than copying
  it, so `dataset_hash`, `prepared_dataset_hash`, `preprocessing_hash`,
  `model_config_hash`, and `checkpoint_hash` are provably identical across the pair
  and `recombination_hash` is the only identity that moves. A copied split file
  could silently drift and turn a controlled A/B into two experiments.

This breaks the one-split-file-per-experiment-directory pattern the three existing
configs happen to follow. That pattern is a coincidence of those experiments using
different datasets, not a rule, and the cross-reference is the point here.

### D15 — Three arms, each changing one variable

```
  combo1/default        nms         + iou 0.5  + class-aware      ← v1 behavior, pinned
  combo1b/default       nms         + ios 0.8  + class-agnostic   ← new defaults
  combo1b/greedy_nmm    greedy_nmm  + ios 0.8  + class-agnostic   ← merging, opt-in
```

Each adjacent pair differs in one thing. combo1 to combo1b/default isolates
**matching** (metric plus class-agnostic); combo1b/default to combo1b/greedy_nmm
isolates **merging** (`method` alone). Neither comparison confounds the two, which the
earlier two-arm layout did.

The second comparison is the one that decides whether `greedy_nmm` is ever worth
selecting. Its expected shape from the measurements: identical detection counts, so
any mAP difference comes purely from box geometry — merging trades slight inflation on
coincident pairs for correct extents on fragments that outrank their whole detection
(10 of 21 same-class nested pairs on frame 2). If that trade is negative, `greedy_nmm`
stays specified but unused, and `max_merge_growth` never fires in practice.

All three arms reuse combo1's checkpoint and split file, so each is one `infer`/`eval`
pass with no training.

## Risks / Trade-offs

- **Box inflation from merging — the largest risk, and bounded by D14.** On
  near-coincident pairs union grows the box slightly (1.07× worst case on frame 1).
  On frame 2 an *unbounded* union reached **248×**, because a large false-positive
  blob was absorbed into a small real detection that outscored it. `max_merge_growth`
  caps this at 2.7× measured, with no change to detection count. `method: nms`
  eliminates merge inflation entirely and remains selectable as the safe ablation —
  it produces the identical 376 detections on frame 2, differing only in geometry.
- **IoS matches distinct objects that overlap without being duplicates.** IoS ≈1.0
  for any box inside a larger box. Objects in this imagery do not nest (confirmed
  with the domain owner), and across both frames every same-class nested pair was a
  slice fragment. But **false-positive** boxes nest freely — frame 2's 2658×1206
  blob at score 0.51 contains many real detections — so "objects don't nest" does
  not imply "boxes don't nest". This is what D14 exists for. Matching is unchanged
  by the cap: the blob is still consumed, so the duplicate is still removed; only
  the geometry is protected. Residual guard is that
  `match_metric: intersection_over_union` remains selectable.
- **Score attribution across a bounded merge.** A merged detection keeps the
  winner's score even when the envelope is mostly another box's extent. Accepted;
  scoring by cluster is out of scope (v2 favorability).
- **Non-idempotence** (D6). Documented, not defended against.
- **Prior results are not comparable.** Every existing `recombination_hash`
  changes. Intentional and version-flagged; there is no re-scoring path for old
  rows and none is provided (SATI's clean-break norm).
- **Per-level diagnostic counts inflate.** Audited: `per_level_diagnostic`
  (`evaluation.py:124`) already counts a detection under *every* level in its
  `source_levels`, by design, so no single-level assumption exists to break.
  But larger clusters mean more multi-level detections, so per-level counts sum
  to more than the detection count more often than before. Semantics unchanged;
  the §11 wording needs one clarifying sentence.
- **The upstream hole stays open.** Ultralytics' un-configured, un-hashed
  per-slice `iou=0.7` class-aware NMS still emits the cross-class ghosts this
  change cleans up downstream. Scoped out, explicitly named so it is not lost.

## Migration Plan

Single atomic change, no shims, no dual-read:

1. Schema, defaults, and the three existing authored experiment configs move
   together; all three are pinned to pre-change values (D5), and
   `v1/combo1b/default.yaml` is added as the sole adopter of the new defaults
   (D13). No existing experiment changes behavior.
2. `RECOMBINATION_HASH_VERSION` and the results/sidecar `schema_version` bump in
   the same commit as the param rename, so no artifact exists with a v4 source
   and v3 keys.
3. `MERGE_STRATEGY_NAME` / `MERGE_STRATEGY_VERSION` are deleted, not deprecated;
   `infer.py` resolves the strategy from config.
4. Existing runs under `runs/` are left as-is and are not re-scorable. Any
   before/after comparison is produced by re-running with `method: nms` and
   `match_metric: intersection_over_union`, which reproduces v1 *behavior* under a
   v4 hash — behavioral equivalence, not digest equivalence.

## Open Questions

1. **Is `max_merge_growth: 4.0` the right bound?** Settled at `4.0` for now. It is
   pinned between two measured quantities — legitimate fragment merges need ≤3.2×, the one observed
   pathological merge needed 248× — so anything in 4–10 works on current evidence
   and `4.0` is the tight end. What is unmeasured is how small slice fragments get
   in practice: a fragment under a quarter of its object silently falls back to
   suppression rather than being restored, and nothing currently reports when the
   cap fires. **Recommendation: count cap activations per run and surface the count**
   so under-restoration is observable instead of invisible; decide the field's
   final default once that number is known across the split.
2. **Does `greedy_nmm` ever earn promotion to the default?** It now ships as an
   option (D16) because it has no measured advantage — identical counts, one 248x
   inflation case. The `combo1b/greedy_nmm` arm (D15) measures whether correct
   fragment extents outweigh slight inflation on coincident pairs, on mAP. If the
   answer is no across the split, the honest follow-up is to consider **removing**
   `greedy_nmm` rather than leaving a specified-but-never-used strategy and an inert
   `max_merge_growth` field in the config surface.

**Closed during design** (recorded so the reasoning is not re-litigated):

- *Is the IoS threshold `0.8` safe, or does it under-merge?* Confirmed on two
  frames, 391 duplicate pairs, none in the 0.5–0.8 band (D12).

- *Does the per-level diagnostic assume one detection maps to one level?* No —
  `evaluation.py:124` already iterates `for level in row["source_levels"]` and
  counts the detection under each. Nothing to redesign; see the Risks entry for
  the count-inflation consequence.
- *Should `merged_class_ids` land on `slice_prediction` rows?* The question was
  malformed: the column lives in the single shared table schema, so its presence
  was never conditional. The existing typed-null rule answers it — null on
  non-merge rows. Folded into D11.
