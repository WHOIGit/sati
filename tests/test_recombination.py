"""Two-stage recombination (DESIGN-DOC-v1 §6.2) + recombination_hash scoping.

Several cases here carry measured geometry from two reference frames rather than
invented numbers, because the defaults they pin were chosen from those
measurements. Where that is so, the docstring says which frame.

Note on what is deliberately *not* asserted: recombination is not idempotent
under ``greedy_nmm``. Matching evaluates against the original winner box in a
single pass (design D6), so feeding ``final_merge`` boxes back through a stage can
merge further. Nothing in the pipeline does that, and no test should assert
round-trip stability.
"""

import pytest

from sati.config_schemas import RecombinationConfig, SlicingConfig
from sati.hashing import recombination_hash
from sati.tiling import (
    FinalDetection,
    SlicePrediction,
    box_ios,
    box_iou,
    match_clusters,
    merge_cross_level,
    merge_within_level,
    plan_frame,
    recombine,
    recombine_counted,
)

# packaged defaults: nms, intersection_over_smaller, 0.8/0.8, growth 8.0,
# conf 0.05, class-agnostic
RC = RecombinationConfig()
# the pre-change v1 reference the existing experiment configs are pinned to
RC_V1 = RecombinationConfig(
    match_metric="intersection_over_union",
    within_level_threshold=0.5,
    cross_level_threshold=0.5,
    class_agnostic=False,
)
RC_MERGE = RC.model_copy(update={"method": "greedy_nmm"})


def test_box_iou():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_box_ios_sees_containment_that_iou_cannot():
    """A nested box scores 1.0 on IoS and its area ratio on IoU.

    This is the whole reason the default metric changed: a slice-clipped fragment
    is a strict subset of its whole detection, so no IoU threshold can match them.
    """
    whole, nested = (0.0, 0.0, 100.0, 100.0), (0.0, 0.0, 30.0, 100.0)
    assert box_ios(whole, nested) == pytest.approx(1.0)
    assert box_iou(whole, nested) == pytest.approx(0.3)
    assert box_ios((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_match_clusters_class_aware_vs_agnostic():
    boxes = [(0, 0, 10, 10), (1, 1, 11, 11), (0, 0, 10, 10)]
    scores = [0.9, 0.8, 0.7]
    classes = [0, 0, 1]
    aware = match_clusters(boxes, scores, classes, 0.5, class_agnostic=False)
    assert aware == [(0, [1]), (2, [])]  # class 1 box survives
    agnostic = match_clusters(boxes, scores, classes, 0.5, class_agnostic=True)
    assert agnostic == [(0, [1, 2])]


def test_match_clusters_deterministic_tie_break():
    boxes = [(0, 0, 10, 10), (100, 100, 110, 110)]
    assert match_clusters(boxes, [0.5, 0.5], [0, 0], 0.5, False) == [(0, []), (1, [])]


def test_threshold_comparison_is_strict():
    """Metric exactly equal to the threshold does not match (design D8).

    A perfectly bisected fragment gives IoU exactly 0.5, so at threshold 0.5 the
    boundary decides whether it is deduplicated at all. Pinned rather than left to
    a ``>`` versus ``>=`` accident.
    """
    boxes = [(0.0, 0.0, 100.0, 100.0), (0.0, 0.0, 50.0, 100.0)]
    args = (boxes, [0.9, 0.8], [0, 0])
    assert match_clusters(*args, 0.5, True, "intersection_over_union") == [(0, []), (1, [])]
    assert match_clusters(*args, 0.49, True, "intersection_over_union") == [(0, [1])]
    # ...and IoS makes it moot: the nested box scores 1.0
    assert match_clusters(*args, 0.8, True, "intersection_over_smaller") == [(0, [1])]


def _plan_2048():
    # 2048×2048, nominal 1024, no overlap → level 0: 4 slices; level 1: whole-level
    slicing = SlicingConfig(nominal_slice_size=1024, overlap_ratio=0.0)
    return plan_frame((2048, 2048), slicing)


def test_touching_fragments_still_do_not_match():
    """Two abutting fragments share no area, so nothing can join them.

    Previously this asserted the same outcome as the *accepted v1 bias*. It is
    still the outcome, but for a narrower reason: with zero overlap neither IoU nor
    IoS can see a relationship, because the intersection itself is empty. The fix
    for edge fragments is overlap plus IoS (see the next test), not the metric
    alone — an important limit on what this change bought.
    """
    plan = _plan_2048()
    top_left = next(s for s in plan.slices if s.level == 0 and s.slice_xy == (0, 0))
    top_right = next(s for s in plan.slices if s.level == 0 and s.slice_xy == (1024, 0))
    preds = [
        SlicePrediction(top_left.slice_id, 0, (1000.0, 100.0, 1024.0, 200.0), 0.90, 3),
        SlicePrediction(top_right.slice_id, 0, (0.0, 98.0, 78.0, 202.0), 0.80, 3),
    ]
    for cfg in (RC, RC_MERGE, RC_V1):
        assert len(merge_within_level(preds, plan, cfg)) == 2


def test_clipped_fragment_matches_whole_detection_only_under_ios():
    """Measured frame-1 geometry: nominal 1024, overlap 0.2, a 100x100 object.

    Slice A (window 0..1024) sees it clipped to 24px wide; slice B (window
    820..1844) sees it whole. IoU is 0.24 and cannot reach any usable threshold;
    IoS is 1.0. This single case is the justification for the metric default.
    """
    slicing = SlicingConfig(nominal_slice_size=1024, overlap_ratio=0.2)
    plan = plan_frame((2048, 1024), slicing)
    a = next(s for s in plan.slices if s.level == 0 and s.slice_xy[0] == 0)
    b = next(s for s in plan.slices if s.level == 0 and s.slice_xy[0] == 820)

    clipped = (1000.0, 100.0, 1024.0, 200.0)  # level coords
    whole = (1000.0, 100.0, 1100.0, 200.0)
    assert box_iou(clipped, whole) == pytest.approx(0.24)
    assert box_ios(clipped, whole) == pytest.approx(1.0)

    preds = [
        SlicePrediction(a.slice_id, 0, clipped, 0.90, 3),
        SlicePrediction(b.slice_id, 0, (180.0, 100.0, 280.0, 200.0), 0.80, 3),
    ]
    # IoU at 0.5 leaves a duplicate; IoS at 0.8 collapses it
    assert len([d for d in merge_within_level(preds, plan, RC_V1) if d.level == 0]) == 2
    (det,) = [d for d in merge_within_level(preds, plan, RC) if d.level == 0]
    assert det.score == 0.90
    # under suppression the winner is the *fragment*, box unchanged
    assert det.bbox_level_xyxy == pytest.approx(clipped)


def test_default_growth_bound_admits_the_motivating_fragment():
    """The motivating fragment needs 4.167x, which the 8.0 default admits.

    Area ratio here is 0.24 (a 24px clip of a 100px object), so restoring the whole
    extent needs 1/0.24 = 4.167x. This case is *why* the default is 8.0 and not
    4.0: a 4.0 bound refused the very fragment that motivates the metric change,
    missing it by 4%. The measured range was wide — legitimate merges needed <=3.2x
    and the one pathological merge needed 248x — so 8.0 sits mid-gap, admitting
    fragments down to 12.5% of their object while still blocking the blob by 31x.
    """
    slicing = SlicingConfig(nominal_slice_size=1024, overlap_ratio=0.2)
    plan = plan_frame((2048, 1024), slicing)
    a = next(s for s in plan.slices if s.level == 0 and s.slice_xy[0] == 0)
    b = next(s for s in plan.slices if s.level == 0 and s.slice_xy[0] == 820)
    clipped = (1000.0, 100.0, 1024.0, 200.0)
    whole = (1000.0, 100.0, 1100.0, 200.0)
    preds = [
        SlicePrediction(a.slice_id, 0, clipped, 0.90, 3),  # fragment outranks whole
        SlicePrediction(b.slice_id, 0, (180.0, 100.0, 280.0, 200.0), 0.80, 3),
    ]
    assert RC_MERGE.max_merge_growth == 8.0
    result = recombine_counted(preds, plan, RC_MERGE)
    (det,) = [d for d in result.level_detections if d.level == 0]
    assert det.bbox_level_xyxy == pytest.approx(whole)  # restored at 8.0
    assert result.within_level_capped == 0

    # the rejected 4.0 default, kept as the regression that drove the change
    tight = RC_MERGE.model_copy(update={"max_merge_growth": 4.0})
    blocked = recombine_counted(preds, plan, tight)
    (unrestored,) = [d for d in blocked.level_detections if d.level == 0]
    assert unrestored.bbox_level_xyxy == pytest.approx(clipped)
    assert blocked.within_level_capped == 1  # and the cap says so out loud


def test_growth_bound_still_blocks_a_fragment_below_one_eighth():
    """8.0 moves the cliff rather than removing it (design D14's accepted cost).

    A fragment smaller than 1/max_merge_growth of its object is suppressed instead
    of restored. At 8.0 that boundary is 12.5%, so a 10% clip still falls back.
    """
    plan = _plan_2048()
    s0 = next(s for s in plan.slices if s.level == 0 and s.slice_xy == (0, 0))
    whole = (0.0, 0.0, 100.0, 100.0)
    tiny = (0.0, 0.0, 10.0, 100.0)  # area ratio 0.10 -> needs 10x
    preds = [
        SlicePrediction(s0.slice_id, 0, tiny, 0.9, 1),  # tiny fragment outranks
        SlicePrediction(s0.slice_id, 0, whole, 0.8, 1),
    ]
    result = recombine_counted(preds, plan, RC_MERGE)
    (det,) = [d for d in result.level_detections if d.level == 0]
    assert det.bbox_level_xyxy == pytest.approx(tiny)
    assert result.within_level_capped == 1


def test_fragment_outranking_whole_is_where_merging_pays():
    """Frame-2 measurement: the fragment outscored the whole box in 10 of 21 pairs.

    Suppression then keeps the half box and deletes the real object. Merging makes
    the result independent of which member won the score ordering — the one thing
    ``greedy_nmm`` demonstrably buys.
    """
    plan = _plan_2048()
    s0 = next(s for s in plan.slices if s.level == 0 and s.slice_xy == (0, 0))
    whole = (100.0, 100.0, 400.0, 400.0)
    fragment = (100.0, 100.0, 200.0, 400.0)
    preds = [
        SlicePrediction(s0.slice_id, 0, whole, 0.710, 1),
        SlicePrediction(s0.slice_id, 0, fragment, 0.752, 1),  # fragment wins
    ]
    (suppressed,) = [d for d in merge_within_level(preds, plan, RC) if d.level == 0]
    assert suppressed.bbox_level_xyxy == pytest.approx(fragment)
    (merged,) = [d for d in merge_within_level(preds, plan, RC_MERGE) if d.level == 0]
    assert merged.bbox_level_xyxy == pytest.approx(whole)
    assert merged.score == 0.752  # winner's score either way


def test_growth_bound_blocks_the_measured_blob_absorption():
    """Frame-2 measurement: a 2658x1206 false positive at 0.510 nesting a
    137x97 real detection at 0.933 (IoS 1.000, IoU 0.002).

    The small box outscores the blob, so an unbounded union would replace it with
    the blob — 248x inflation. The bound keeps the winner's box while still
    consuming the blob, so the duplicate is removed and only the geometry is
    protected.
    """
    plan = plan_frame((4096, 4096), SlicingConfig(nominal_slice_size=1024, overlap_ratio=0.0))
    s0 = next(s for s in plan.slices if s.level == 0 and s.slice_xy == (0, 0))
    small = (10.0, 10.0, 147.0, 107.0)  # 137x97
    blob = (0.0, 0.0, 1000.0, 800.0)
    preds = [
        SlicePrediction(s0.slice_id, 0, small, 0.933, 1),
        SlicePrediction(s0.slice_id, 0, blob, 0.510, 1),
    ]
    result = recombine_counted(preds, plan, RC_MERGE)
    (det,) = [d for d in result.level_detections if d.level == 0]
    assert det.bbox_level_xyxy == pytest.approx(small)  # not the 248x envelope
    assert det.source_slice_ids == (s0.slice_id,)  # the blob was still consumed
    assert result.within_level_capped == 1

    # unbounded, the pathology is real — this is what the bound prevents
    unbounded = RC_MERGE.model_copy(update={"max_merge_growth": 1e9})
    (inflated,) = [
        d for d in merge_within_level(preds, plan, unbounded) if d.level == 0
    ]
    assert inflated.bbox_level_xyxy == pytest.approx(blob)


def test_growth_bound_never_changes_the_detection_count():
    """The bound protects geometry only — clusters and survivors are unaffected."""
    plan = _plan_2048()
    s0 = next(s for s in plan.slices if s.level == 0 and s.slice_xy == (0, 0))
    preds = [
        SlicePrediction(s0.slice_id, 0, (10.0, 10.0, 60.0, 60.0), 0.9, 1),
        SlicePrediction(s0.slice_id, 0, (0.0, 0.0, 400.0, 400.0), 0.5, 1),
        SlicePrediction(s0.slice_id, 0, (700.0, 700.0, 800.0, 800.0), 0.8, 1),
    ]
    counts = {
        growth: len(recombine(preds, plan, RC_MERGE.model_copy(update={"max_merge_growth": growth}))[1])
        for growth in (1.0, 2.0, 4.0, 1e9)
    }
    assert len(set(counts.values())) == 1, counts


def test_measured_fragment_ratios_all_merge_at_the_default():
    """Frame-1 nested pairs sit at area ratio 0.31-0.48, needing 2.1x-3.2x.

    Comfortably inside 8.0. A 2.0 bound would block them, which is why the tight
    end of the measured gap was rejected.
    """
    plan = _plan_2048()
    s0 = next(s for s in plan.slices if s.level == 0 and s.slice_xy == (0, 0))
    whole = (0.0, 0.0, 100.0, 100.0)
    fragment = (0.0, 0.0, 31.0, 100.0)  # area ratio 0.31 → needs 3.2x
    preds = [
        SlicePrediction(s0.slice_id, 0, fragment, 0.9, 1),  # fragment wins
        SlicePrediction(s0.slice_id, 0, whole, 0.8, 1),
    ]
    at_four = merge_within_level(preds, plan, RC_MERGE)
    assert [d for d in at_four if d.level == 0][0].bbox_level_xyxy == pytest.approx(whole)
    tight = RC_MERGE.model_copy(update={"max_merge_growth": 2.0})
    assert [d for d in merge_within_level(preds, plan, tight) if d.level == 0][
        0
    ].bbox_level_xyxy == pytest.approx(fragment)


def test_class_agnostic_collapse_records_the_losing_class():
    """Frame-2: 347 of 369 duplicate pairs were cross-class.

    Class-agnostic matching collapses them; ``merged_class_ids`` is what keeps the
    disagreement recoverable afterwards.
    """
    plan = _plan_2048()
    s0 = next(s for s in plan.slices if s.level == 0 and s.slice_xy == (0, 0))
    box = (100.0, 100.0, 200.0, 200.0)
    preds = [
        SlicePrediction(s0.slice_id, 0, box, 0.970, 1),
        SlicePrediction(s0.slice_id, 0, box, 0.117, 2),  # the ghost
    ]
    (det,) = [d for d in merge_within_level(preds, plan, RC) if d.level == 0]
    assert det.class_id == 1  # most confident class wins
    assert det.merged_class_ids == (1, 2)  # winner first, absorbed after
    (final,) = recombine(preds, plan, RC)[1]
    assert final.merged_class_ids == (1, 2)

    # class-aware keeps both, and each records only its own class
    aware = [d for d in merge_within_level(preds, plan, RC_V1) if d.level == 0]
    assert len(aware) == 2
    assert sorted(d.merged_class_ids for d in aware) == [(1,), (2,)]


def test_single_member_cluster_records_one_class_not_null():
    """"Cluster of one" and "not a merge row" must stay distinguishable (D11)."""
    plan = _plan_2048()
    s0 = next(s for s in plan.slices if s.level == 0 and s.slice_xy == (0, 0))
    preds = [SlicePrediction(s0.slice_id, 0, (10.0, 10.0, 50.0, 50.0), 0.9, 3)]
    (det,) = merge_within_level(preds, plan, RC)
    assert det.merged_class_ids == (3,)


def test_methods_form_identical_clusters_and_differ_only_in_geometry():
    """``nms`` is an exact ablation of the merge decision.

    Measured on both reference frames: the two methods produce the same detection
    count. This pins that as a property, so a future change that makes merging
    also affect clustering cannot pass silently.
    """
    plan = _plan_2048()
    s0 = next(s for s in plan.slices if s.level == 0 and s.slice_xy == (0, 0))
    preds = [
        SlicePrediction(s0.slice_id, 0, (100.0, 100.0, 300.0, 300.0), 0.9, 1),
        SlicePrediction(s0.slice_id, 0, (120.0, 120.0, 260.0, 280.0), 0.7, 2),
        SlicePrediction(s0.slice_id, 0, (700.0, 700.0, 900.0, 900.0), 0.8, 1),
    ]
    a_lvl, a_fin = recombine(preds, plan, RC)
    b_lvl, b_fin = recombine(preds, plan, RC_MERGE)
    assert len(a_lvl) == len(b_lvl) and len(a_fin) == len(b_fin)
    assert [d.source_slice_ids for d in a_lvl] == [d.source_slice_ids for d in b_lvl]
    assert [d.merged_class_ids for d in a_fin] == [d.merged_class_ids for d in b_fin]
    assert [d.score for d in a_fin] == [d.score for d in b_fin]


def test_determinism_and_sorted_contributors():
    slicing = SlicingConfig(nominal_slice_size=1024, overlap_ratio=0.5)
    plan = plan_frame((2048, 1024), slicing)
    # windows at x = 0, 512, 1024; an object at level x∈[1000,1100] is visible in
    # all three, clipped by the outer two and whole in the middle one
    windows = sorted(
        (s for s in plan.slices if s.level == 0), key=lambda s: s.slice_xy[0]
    )
    assert [w.slice_xy[0] for w in windows] == [0, 512, 1024]

    def emitted(spec):
        """What that slice can physically report: the object clipped to its window."""
        wx1 = spec.slice_xy[0]
        wx2 = wx1 + spec.slice_wh[0]
        clipped_level = (max(1000.0, wx1), 100.0, min(1100.0, wx2), 200.0)
        return spec.affine_slice_to_level.invert().apply_box(clipped_level)

    # top score on the window that sees the object whole, so the growth bound is
    # not what this test is about; listed out of slice-id order on purpose
    preds = [
        SlicePrediction(windows[2].slice_id, 0, emitted(windows[2]), 0.5, 1),
        SlicePrediction(windows[1].slice_id, 0, emitted(windows[1]), 0.9, 1),
        SlicePrediction(windows[0].slice_id, 0, emitted(windows[0]), 0.7, 2),
    ]
    first = recombine(preds, plan, RC_MERGE)
    assert first == recombine(preds, plan, RC_MERGE)
    (det,) = [d for d in first[0] if d.level == 0]
    assert list(det.source_slice_ids) == sorted(set(det.source_slice_ids))
    assert list(det.merged_class_ids[1:]) == sorted(det.merged_class_ids[1:])


def test_growth_bound_is_inert_under_suppression():
    plan = _plan_2048()
    s0 = next(s for s in plan.slices if s.level == 0 and s.slice_xy == (0, 0))
    preds = [
        SlicePrediction(s0.slice_id, 0, (10.0, 10.0, 60.0, 60.0), 0.9, 1),
        SlicePrediction(s0.slice_id, 0, (0.0, 0.0, 400.0, 400.0), 0.5, 1),
    ]
    assert recombine_counted(preds, plan, RC).capped == 0


def test_within_level_merges_duplicates_in_overlap_strip():
    slicing = SlicingConfig(nominal_slice_size=1024, overlap_ratio=0.25)
    plan = plan_frame((1792, 1024), slicing)  # level 0: x tiles at 0 and 768; y stretched
    left = next(s for s in plan.slices if s.level == 0 and s.slice_xy[0] == 0)
    right = next(s for s in plan.slices if s.level == 0 and s.slice_xy[0] == 768)
    # object at level x∈[800,900] lives in both windows
    preds = [
        SlicePrediction(left.slice_id, 0, (800.0, 100.0, 900.0, 200.0), 0.85, 1),
        SlicePrediction(right.slice_id, 0, (32.0, 100.0, 132.0, 200.0), 0.95, 1),
    ]
    dets = merge_within_level(preds, plan, RC)
    (det,) = [d for d in dets if d.level == 0]
    assert det.score == 0.95
    assert det.source_slice_ids == tuple(sorted([left.slice_id, right.slice_id]))
    assert det.bbox_level_xyxy == pytest.approx((800.0, 100.0, 900.0, 200.0))


def test_confidence_threshold_applied_before_stage1():
    plan = _plan_2048()
    s0 = plan.slices[0]
    preds = [SlicePrediction(s0.slice_id, 0, (10.0, 10.0, 100.0, 100.0), 0.01, 0)]
    assert merge_within_level(preds, plan, RC) == ()


def test_stretched_slice_prediction_lifts_correctly():
    plan = _plan_2048()
    whole = next(s for s in plan.slices if s.level == 1)
    assert whole.is_whole_level and whole.stretched_axes == ()
    # 2048/2 = 1024 → level 1 not stretched; use a smaller frame for stretch
    slicing = SlicingConfig(nominal_slice_size=1024)
    plan_small = plan_frame((1500, 900), slicing)
    lvl1 = next(s for s in plan_small.slices if s.level == 1)
    assert lvl1.stretched_axes == ("x", "y")
    # emitted-pixel box maps back by slice_wh/nominal
    preds = [SlicePrediction(lvl1.slice_id, 1, (0.0, 0.0, 1024.0, 1024.0), 0.5, 0)]
    (det,) = merge_within_level(preds, plan_small, RC)
    level1 = plan_small.level(1)
    assert det.bbox_level_xyxy == pytest.approx((0.0, 0.0, level1.width, level1.height))


def test_cross_level_merge_and_source_levels():
    plan = _plan_2048()
    tl = next(s for s in plan.slices if s.level == 0 and s.slice_xy == (0, 0))
    whole = next(s for s in plan.slices if s.level == 1)
    # same physical object: fullres [100,100,400,400] → level0 same, level1 halved
    preds = [
        SlicePrediction(tl.slice_id, 0, (100.0, 100.0, 400.0, 400.0), 0.9, 2),
        SlicePrediction(whole.slice_id, 1, (50.0, 50.0, 200.0, 200.0), 0.7, 2),
    ]
    level_dets, final_dets = recombine(preds, plan, RC)
    assert len(level_dets) == 2  # one per level, no within-level competition
    (final,) = final_dets
    assert final.score == 0.9
    assert final.source_levels == (0, 1)
    assert final.bbox_fullres_xyxy == pytest.approx((100.0, 100.0, 400.0, 400.0))


def test_cross_level_keeps_distinct_objects():
    plan = _plan_2048()
    tl = next(s for s in plan.slices if s.level == 0 and s.slice_xy == (0, 0))
    preds = [
        SlicePrediction(tl.slice_id, 0, (0.0, 0.0, 50.0, 50.0), 0.9, 0),
        SlicePrediction(tl.slice_id, 0, (500.0, 500.0, 600.0, 600.0), 0.8, 0),
    ]
    _, final_dets = recombine(preds, plan, RC)
    assert len(final_dets) == 2
    assert all(isinstance(d, FinalDetection) for d in final_dets)


def test_merge_cross_level_empty():
    assert merge_cross_level([], _plan_2048(), RC) == ()


def test_recombination_hash_scoping():
    base = RecombinationConfig()
    cross_changed = RecombinationConfig(cross_level_threshold=0.9)
    within_changed = RecombinationConfig(within_level_threshold=0.9)
    # level_merge scope ignores cross-level params
    assert recombination_hash(base, "level_merge") == recombination_hash(
        cross_changed, "level_merge"
    )
    assert recombination_hash(base, "level_merge") != recombination_hash(
        within_changed, "level_merge"
    )
    # final_merge scope covers both
    assert recombination_hash(base, "final_merge") != recombination_hash(
        cross_changed, "final_merge"
    )
    # scopes never collide
    assert recombination_hash(base, "level_merge") != recombination_hash(base, "final_merge")


def test_recombination_hash_covers_the_new_params():
    """Method, metric, and growth bound all move the identity.

    ``max_merge_growth`` is included even under ``nms``, where it is inert, so a
    run's identity always records the bound that would have applied (D10).
    """
    base = RecombinationConfig()
    for update in (
        {"method": "greedy_nmm"},
        {"match_metric": "intersection_over_union"},
        {"max_merge_growth": 4.0},  # must differ from the 8.0 default
        {"class_agnostic": False},
    ):
        changed = base.model_copy(update=update)
        for scope in ("level_merge", "final_merge"):
            assert recombination_hash(base, scope) != recombination_hash(changed, scope), update
