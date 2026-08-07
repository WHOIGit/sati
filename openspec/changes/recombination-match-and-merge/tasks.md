## 1. Config surface

- [ ] 1.1 Add `method: Literal["nms", "greedy_nmm"] = "nms"` (D16) and
  `match_metric: Literal["intersection_over_union", "intersection_over_smaller"] =
  "intersection_over_smaller"` to `RecombinationConfig`; rename
  `within_level_iou` → `within_level_threshold` and `cross_level_iou` →
  `cross_level_threshold` with defaults `0.5` → `0.8` (D12); add
  `max_merge_growth: float = 4.0` with `ge=1.0` (D14); flip `class_agnostic` to
  `True`. No aliases, no validators accepting the old names, and no validator
  coupling threshold to metric.
- [ ] 1.2 Update `src/sati/config_defaults/recombination/default.yaml` to the seven
  fields: `method: nms`, `match_metric: intersection_over_smaller`,
  `within_level_threshold: 0.8`, `cross_level_threshold: 0.8`,
  `max_merge_growth: 4.0`, `confidence_threshold: 0.05`, `class_agnostic: true`.
- [ ] 1.3 Rename the recombination fields in all three **existing** experiment
  configs and pin each to pre-change behavior (`method: nms`,
  `match_metric: intersection_over_union`, thresholds `0.5`,
  `class_agnostic: false`) so no existing experiment changes output as a side
  effect of the default flip (D5).
- [ ] 1.4 Add `configs/experiment/v1/combo1b/default.yaml` (D13): a copy of
  `combo1/default.yaml` with `experiment.name: v1/combo1b/default` and only the
  recombination block changed to the new defaults. It SHALL reference
  `configs/experiment/v1/combo1/all_fullframes_split.toml` — do **not** copy the
  split file — and SHALL NOT alter data, model, slicing, or training, so every
  compatibility identity matches combo1 and only `recombination_hash` moves.
- [ ] 1.5 Add `configs/experiment/v1/combo1b/greedy_nmm.yaml` (D15): identical to
  `combo1b/default.yaml` except `experiment.name: v1/combo1b/greedy_nmm` and
  `method: greedy_nmm`. Same match metric, thresholds, and `class_agnostic` — so this
  pair isolates **merge vs suppress** as the single variable, exactly as combo1 vs
  combo1b/default isolates matching.
- [ ] 1.6 Add a test that a config using `within_level_iou` fails validation with
  a strict-schema error rather than silently defaulting.
- [ ] 1.7 Add a test asserting combo1 and combo1b resolve to equal `config_hash`
  inputs for every compatibility identity (dataset, prepared dataset, target
  schema, class mapping, model config, preprocessing) and differ only in
  recombination — the A/B/C is only controlled if this holds. Extend it to
  `combo1b/greedy_nmm.yaml`, which must additionally differ from
  `combo1b/default.yaml` in nothing but `method`.

## 2. Matching and merge in `sati.tiling`

- [ ] 2.1 Add `box_ios` and a metric dispatch keyed by the config literal; keep
  `box_iou` unchanged and keep the comparison strictly `>` (D8).
- [ ] 2.2 Generalize `greedy_nms` into a metric-parameterized match pass returning
  `(winner_index, matched_indices)`; preserve the `(-score, input_index)` ordering
  and ascending candidate iteration exactly (D7). Matching is evaluated against
  the original winner box, single pass (D6).
- [ ] 2.3 Add the strategy registry `{"nms": 1, "greedy_nmm": 1}` with a resolver
  from config, and **delete** `MERGE_STRATEGY_NAME` / `MERGE_STRATEGY_VERSION`
  and their import in `infer.py` — migration-free removal, no re-export.
- [ ] 2.4 Add `merged_class_ids` to `LevelDetection` and `FinalDetection`
  (winner's class first, remainder ascending, never empty) and populate it in both
  stages regardless of method or matching mode (D11).
- [ ] 2.5 Apply envelope merging in `merge_within_level` and `merge_cross_level`
  when the method is `greedy_nmm`; leave the winner's box untouched under `nms`.
  Both modes keep the winner's score and class and the existing sorted,
  de-duplicated contributor lists.
- [ ] 2.6 Enforce `max_merge_growth` (D14): take the envelope only when
  `area(envelope) <= max_merge_growth * area(winner_box)`, else keep the winner's
  box. Matched members are consumed either way — the bound must not change which
  clusters form or how many detections survive.
- [ ] 2.7 Count `max_merge_growth` cap activations per stage and expose the total so
  under-restoration is observable rather than invisible (open question 1); surface it
  through the `infer`/`eval` summary rather than a new results column. Zero under the
  default `nms` method.
- [ ] 2.8 Update the module docstring: it currently says recombination "is the
  two-stage NMS at the bottom of this module".

## 3. Provenance, hashing, and results schema

- [ ] 3.1 Update `recombination_hash_full_source` params per D10: add
  `match_metric` and `max_merge_growth`, use the renamed threshold keys, keep the scope rule, take
  `strategy`/`strategy_version` from the registry, and do **not** duplicate
  `method` into `params`.
- [ ] 3.2 Bump `RECOMBINATION_HASH_VERSION` `"3"` → `"4"`.
- [ ] 3.3 Add the `merged_class_ids` list column to the shared row schema in
  `manifests.py` and bump the results/sidecar `schema_version`. Merge rows always
  write a non-empty list; `frame_metadata` and `slice_prediction` rows write a
  typed null, never `[class_id]` (D11).
- [ ] 3.4 Update `infer.py` to resolve the per-stage strategy name/version from
  config and write `merged_class_ids` on both merge row types.
- [ ] 3.5 Update the `_metadata.json` sidecar writer/model so the stored
  `recombination_hash` `*_source` objects carry the v4 shape.

## 4. Evaluation and inspection

- [ ] 4.1 No change to `per_level_diagnostic` logic (audited: `evaluation.py:124`
  already counts a detection under every level in `source_levels`). Update its
  docstring to note that larger clusters make per-level counts sum to more than the
  detection count more often, and add a test pinning that a multi-level detection
  is counted once per contributing level.
- [ ] 4.2 Update `inspect.py` results summarization for the widened schema.

## 5. Tests

- [ ] 5.1 `test_recombination.py`: rewrite
  `test_within_level_merges_across_slice_boundary`, which currently asserts the
  *unfixed* behavior (two boundary fragments both surviving) as expected.
- [ ] 5.2 Add the reference-run geometry as a fixed case: nominal 1024, overlap
  0.2, 100×100 object at level x ∈ [1000,1100] — assert IoU ≈ 0.24 (no match) and
  IoS = 1.0 (match), and that `greedy_nmm` yields the whole box.
- [ ] 5.3 Add a fragment-outranks-whole case (fragment score above the whole box's)
  asserting the merged box is the whole object in `greedy_nmm` and the fragment
  under `nms` — the D6 payoff, pinned.
- [ ] 5.4 Add a class-agnostic case over coincident different-class boxes asserting
  one survivor, the higher-scoring class, and both classes in `merged_class_ids`.
- [ ] 5.5 Pin the threshold boundary: metric exactly equal to the threshold does
  not match (D8).
- [ ] 5.6 Pin determinism: identical input yields byte-identical output, and
  contributor lists are sorted and de-duplicated.
- [ ] 5.7 Assert `nms` and `greedy_nmm` form the *same* clusters and differ only
  in output geometry, so `nms` is an exact ablation of the merge decision.
- [ ] 5.8 Pin D14 with the measured frame-2 geometry: a 137x97 box at score 0.933
  matching a 2658x1206 box at score 0.510 (IoS 1.000, IoU 0.002) SHALL yield the
  small box, not a 248x-inflated envelope, and the large box SHALL still be
  consumed. Also assert two different `max_merge_growth` values give equal
  detection counts.
- [ ] 5.9 Pin that a fragment/whole pair at area ratio 0.31 still merges at
  `max_merge_growth: 4.0` (needs 3.2x) and would not at `2.0` — the boundary that
  sets the default.
- [ ] 5.10 Add an explicit non-idempotence note/test-comment where recombination is
  tested; do **not** assert idempotence (D6).
- [ ] 5.11 `test_infer.py:87` asserts `merge_strategy_name == "nms"`, which stays
  correct under the new default — confirm it now passes for the right reason (resolved
  from config, not from a deleted constant) rather than by coincidence. Update
  `test_evaluation.py` / `test_inspect.py` schema assertions and regenerate hash
  fixtures touching `recombination_hash`.

## 6. Documentation

- [ ] 6.1 `DESIGN-DOC-v1.md`: rewrite §6.2 (drop "NMS only"; document the two
  methods, the metric, bounded envelope semantics and why SATI departs from SAHI
  here, non-idempotence, and replace the
  now-known-inadequate `overlap_ratio` mitigation at line 493 with the real fix);
  update the config table at lines 152–155; remove NMM from the §37 exclusion
  list and from line 662.
- [ ] 6.2 `DESIGN-DOC-v2.md`: narrow §5 and the §19 comparison table to what
  remains deferred (non-greedy `nmm`, `lsnms`, favorability, `near_edge_count`,
  edge penalties, nested per-stage config); keep §11's WBF/cross-level-averaging
  entry.
- [ ] 6.3 Record the Ultralytics per-slice NMS provenance hole
  (`backends.py:150`: no `iou`, no `agnostic_nms`, therefore un-hashed library
  defaults) as a named follow-up, not a silent omission.

## 7. Verification and closeout

- [ ] 7.1 Run `pytest tests/` and `ruff check src tests` in the Nix dev shell.
- [ ] 7.2 Run `sati infer`/`eval` for both combo1b arms against combo1's
  existing checkpoint with `results.cache_slice_predictions: true`, and confirm on
  real Stage-1 input: duplicate-pair count drops to zero, and greedy clusters still
  equal connected components (the D2 deferral trigger for non-greedy `nmm`).
- [ ] 7.3 Resolve open question 1 against the **full split**: report
  `max_merge_growth` activation counts (task 2.7) and check whether slice fragments
  below a quarter of their object are common enough to loosen the `4.0` bound.
  Record the confirmation or revision in `design.md` D14.
- [ ] 7.4 Resolve open question 2: compare `v1/combo1b/greedy_nmm` against
  `v1/combo1b/default` on mAP and settle whether merging earns promotion to the
  default. Record the outcome in `design.md`; if it loses across the split, propose
  removing `greedy_nmm` and `max_merge_growth` rather than shipping an unused
  strategy.
- [ ] 7.5 Run `v1/combo1/default` (pinned to pre-change values) and confirm
  behavioral equivalence to the existing
  `runs/v1/combo1/default/2026-08-06_05-15-58_elated-salmon_infer/` output — same
  boxes, scores, and classes; digests differ by design. Then diff combo1 vs
  combo1b/default vs combo1b/greedy_nmm as the actual A/B/C result — matching isolated
  in the first pair, merging in the second.
- [ ] 7.6 Run `openspec validate --all --strict`, reconcile the main specs with
  these deltas, and leave the change ready for archive.
