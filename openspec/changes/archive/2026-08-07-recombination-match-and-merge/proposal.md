## Why

Measured on a real run (`runs/v1/combo1/default/2026-08-06_05-15-58_elated-salmon_infer/`,
one 21117×9597 frame, 154 `final_merge` rows): **22 duplicate pairs at
intersection-over-smaller > 0.5, roughly 14% of the output.** They split cleanly
into two distinct failure modes that today's class-aware IoU NMS cannot reach:

- **16 pairs — different class, near-coincident** (area ratio 0.86–1.00, IoU
  0.78–1.00). Uniformly a high-confidence class-1 box shadowed by a class-2/3
  ghost at 0.05–0.22. Their IoU is already far above `within_level_iou: 0.5`;
  the only reason both survive is `class_agnostic: false`.
- **6 pairs — same class, one box nested inside the other** (area ratio
  0.31–0.48, **IoU 0.31–0.48**, IoS 0.96–1.00). Five of six preserve one axis
  exactly and cut the other (e.g. 351×819 vs 165×807), i.e. a detection
  bisected by a slice window edge. Their IoU is *structurally* below any usable
  threshold, because a clipped fragment is a strict subset of its whole self, so
  IoU degenerates to the area ratio. No IoU threshold can match them.

In 2 of those 6 nested pairs the **fragment outranks the whole detection**
(0.752 vs 0.710, 0.839 vs 0.388), so suppression alone would keep the half-box
and delete the real object; a third pair is a 0.998/0.987 coin-flip. Merging by
union makes the outcome independent of which box won.

This is the accepted v1 bias recorded at `DESIGN-DOC-v1.md:491` ("a near-slice-edge
fragment outranks a whole detection from a coarser level, and duplicates split
across levels dedup purely by IoU"). It is fixable now with the SAHI-standard
match metric and matching mode, without any of v2's favorability/edge machinery.

## What Changes

- **NEW (config)** Add `recombination.method`: `nms` | `greedy_nmm`, defaulting to
  **`nms`**. `greedy_nmm` takes the union (envelope) of the winner and every box
  matched to it, keeping the winner's score and class, and records all contributors —
  SAHI's GREEDYNMM semantics, implemented in `sati.tiling`, never imported from
  `sahi.postprocess`, because SATI's merge is provenance-aware. It ships as an
  **option, not the default**: on the reference frames merging produced identical
  detection counts to suppression while introducing a 248x box-inflation case, so it
  has no measured advantage yet to justify defaulting to it (see design D16).
- **BREAKING (config)** Add `recombination.match_metric`:
  `intersection_over_union` | `intersection_over_smaller`, defaulting to
  **`intersection_over_smaller`**. This is the load-bearing half of the fix: it
  is the only thing that can match a slice-clipped fragment to its whole self.
- **BREAKING (config)** Rename `recombination.within_level_iou` →
  `within_level_threshold` and `cross_level_iou` → `cross_level_threshold`, and
  raise both defaults `0.5` → **`0.8`**. The old names assert a metric that is now
  configurable, and `0.5` on intersection-over-smaller is substantially *looser*
  than `0.5` on intersection-over-union was — every observed duplicate pair sits at
  IoS ≥ 0.93, so `0.8` catches all of them with margin while shrinking the
  over-merge risk. No alias, no shim; `extra="forbid"` makes the break loud.
- **BREAKING (config)** Add `recombination.max_merge_growth: float = 4.0`. A merge
  SHALL take the envelope only when its area is at most this multiple of the
  winner's own area; otherwise the winner keeps its box while its matched boxes are
  still consumed. **This is a deliberate deviation from SAHI**, which unions
  unconditionally, adopted because an unbounded union measured **248× winner-box
  inflation** on a second reference frame — a 2658×1206 false-positive blob at
  score 0.510 absorbed into a 137×97 real detection scoring 0.933 (IoS 1.000, IoU
  0.002). The cap costs one merge out of 299 and leaves the detection count
  unchanged.
- **BREAKING (config)** Flip `recombination.class_agnostic` default `false` →
  **`true`**. The flag itself is retained — it is how the 16-pair failure mode
  gets measured — but agnostic matching is the correct default, and because both
  stages are score-ordered greedy, "most confident class wins" already falls out
  of the existing selection order.
- **BREAKING (hash)** `recombination_hash` sources gain `method` and
  `match_metric` and carry the renamed threshold keys; `RECOMBINATION_HASH_VERSION`
  `"3"` → `"4"`. Existing result rows become non-reproducible, deliberately and
  visibly rather than silently.
- **BREAKING (results schema)** Add a `merged_class_ids` list column to
  `level_merge` and `final_merge` rows, recording every distinct class absorbed
  into the winner (winner's class first). With class-agnostic merging the losing
  class is otherwise unrecoverable, and the observed near-ties make that a real
  loss. Bump results/sidecar `schema_version`.
- **BREAKING (results schema)** `merge_strategy_name` widens from the constant
  `"nms"` to the resolved method name, and strategy version becomes **per
  strategy** rather than one global integer, so `nms` stays at version 1 forever
  instead of appearing to have changed.
- Retire the `MERGE_STRATEGY_NAME` / `MERGE_STRATEGY_VERSION` module constants in
  `sati.tiling` (a development-stage single-strategy interface) in favour of a
  method → `(name, version)` registry resolved from config. Migration-free
  removal: no shim, no re-export.
- Move NMM out of the v2 delta: `DESIGN-DOC-v1.md` §6.2 stops claiming "NMS only",
  §37's exclusion list drops NMM, the config table at §152–155 is updated, and
  `DESIGN-DOC-v2.md` §5 retains only what is still deferred.

## Capabilities

### New Capabilities
<!-- None: this change widens an existing merge stage and its provenance. -->

### Modified Capabilities

- `slicing-and-recombination`: the two-stage merge is no longer IoU-only and
  class-aware-only. Each stage selects by score under a configurable match metric,
  with class-agnostic matching as the default, and may optionally union its matches
  under a bounded growth limit instead of discarding them. Absorbed classes are
  recorded.
- `results-provenance`: merge rows gain `merged_class_ids`;
  `merge_strategy_name` becomes a real strategy identifier; the scope-aware
  `recombination_hash` source gains the strategy, metric, and growth-bound
  parameters and bumps its version.
- `inference-evaluation-export`: the post-hoc recombination override surface
  includes the merge method, match metric, and growth bound, which move final
  recombination identity while compatibility identities stay fixed.

## Non-Goals

Deliberately excluded, and left unsupported so that configuring them fails
validation rather than reaching a stub (`DESIGN-DOC-v1.md:84`):

- **Non-greedy `nmm`.** Measured on the run above: connected components over the
  match graph are **identical to greedy clusters at every stage and at both IoS
  0.5 and 0.3** (max cluster size 2, max union inflation 1.07×). It buys nothing
  on this imagery and carries an unbounded-union failure mode greedy does not —
  a single mediocre detection spanning two adjacent objects fuses both. The name
  stays reserved for the non-greedy variant per `DESIGN-DOC-v2.md:247`.
- `lsnms`, Weighted Box Fusion, and cross-level box averaging (v2 §5 / §11).
- Favorability scoring, `near_edge_count`, and edge penalties (v2 §5). Stage 2
  still ranks by raw confidence.
- Per-stage divergent methods or metrics. One flat namespace now; v2's nested
  `within_level.*` / `cross_level.*` shape lands once `cross_level.method:
  favorability` needs a different value space, so the migration happens once.
- Exposing Ultralytics' per-slice NMS parameters. `backends.py:150` calls
  `predict()` without `iou` or `agnostic_nms`, so per-slice NMS silently runs at
  the library's `iou=0.7`, class-aware defaults — values absent from every config
  and every hash. That is a genuine provenance hole and it is what emits the
  16-pair failure mode upstream, but it is a separate change.

## Impact

- `src/sati/tiling.py`: add `box_ios` and a match-metric dispatch; generalize
  `greedy_nms` into a match/select pass returning `(winner, matched)`; union in
  `merge_within_level` and `merge_cross_level`; strategy registry replacing the
  two module constants; `merged_class_ids` on `LevelDetection` / `FinalDetection`.
- `src/sati/config_schemas/root.py`: `RecombinationConfig` gains `method` and
  `match_metric` literals, renames both thresholds, flips `class_agnostic`.
- `src/sati/config_defaults/recombination/default.yaml`; the three existing
  authored experiment configs (`configs/experiment/v1/{manual/baseline,
  combo1/default,sam2dirty/start_zoom}.yaml`) take the field rename and are pinned
  to pre-change behavior, so none of them changes output as a side effect.
- **New** `configs/experiment/v1/combo1b/default.yaml`: combo1 with only the
  recombination block changed, and the sole adopter of the new defaults. It
  requires **no retraining** — recombination is infer/eval-time policy, so combo1b
  reuses combo1's checkpoint — and it references combo1's
  `all_fullframes_split.toml` rather than copying it, so every compatibility
  identity is provably equal across the pair and `recombination_hash` is the only
  identity that moves.
- `src/sati/hashing.py`: `recombination_hash_full_source` params and version.
- `src/sati/manifests.py`: `merged_class_ids` column, `schema_version`.
- `src/sati/infer.py`: drops the constant import, resolves strategy per stage,
  writes the new column.
- `src/sati/evaluation.py` / `inspect.py`: read the widened schema; per-level
  diagnostics must stay meaningful when a final box unions across levels.
- Tests: `tests/test_recombination.py` (including the assertion at line 47 that
  currently *documents* the unfixed edge-fragment behavior), `test_infer.py:87`
  (`== "nms"`), `test_evaluation.py`, `test_inspect.py`.
- Docs: `DESIGN-DOC-v1.md` §6.2, §3 config table, §37 exclusion list;
  `DESIGN-DOC-v2.md` §5 and the §19 comparison table.
- No new runtime dependency. Preserves every v2 seam: the per-stage `Strategy`
  boundary, `recombination_hash` scoping, and room for `near_edge_count` on
  `level_merge` rows.
