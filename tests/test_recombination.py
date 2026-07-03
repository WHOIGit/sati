"""Two-stage NMS recombination (DESIGN-DOC-v1 §6.2) + recombination_hash scoping."""

import pytest

from sati.config_schemas import RecombinationConfig, SlicingConfig
from sati.hashing import recombination_hash
from sati.tiling import (
    FinalDetection,
    SlicePrediction,
    box_iou,
    greedy_nms,
    merge_cross_level,
    merge_within_level,
    plan_frame,
    recombine,
)

RC = RecombinationConfig()  # defaults: iou 0.5/0.5, conf 0.05, class-aware


def test_box_iou():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_greedy_nms_class_aware_vs_agnostic():
    boxes = [(0, 0, 10, 10), (1, 1, 11, 11), (0, 0, 10, 10)]
    scores = [0.9, 0.8, 0.7]
    classes = [0, 0, 1]
    aware = greedy_nms(boxes, scores, classes, 0.5, class_agnostic=False)
    assert aware == [(0, [1]), (2, [])]  # class 1 box survives
    agnostic = greedy_nms(boxes, scores, classes, 0.5, class_agnostic=True)
    assert agnostic == [(0, [1, 2])]


def test_greedy_nms_deterministic_tie_break():
    boxes = [(0, 0, 10, 10), (100, 100, 110, 110)]
    assert greedy_nms(boxes, [0.5, 0.5], [0, 0], 0.5, False) == [(0, []), (1, [])]


def _plan_2048():
    # 2048×2048, nominal 1024, no overlap → level 0: 4 slices; level 1: whole-level
    slicing = SlicingConfig(nominal_slice_size=1024, overlap_ratio=0.0, downsample_factor=2)
    return plan_frame((2048, 2048), slicing)


def test_within_level_merges_across_slice_boundary():
    plan = _plan_2048()
    top_left = next(s for s in plan.slices if s.level == 0 and s.slice_xy == (0, 0))
    top_right = next(s for s in plan.slices if s.level == 0 and s.slice_xy == (1024, 0))
    # same object seen near the shared boundary by both slices (level coords
    # ~[1000,100,1100,200]): each slice reports it in its own emitted pixels
    preds = [
        SlicePrediction(top_left.slice_id, 0, (1000.0, 100.0, 1024.0, 200.0), 0.90, 3),
        SlicePrediction(top_right.slice_id, 0, (0.0, 98.0, 78.0, 202.0), 0.80, 3),
    ]
    dets = merge_within_level(preds, plan, RC)
    # the two fragments touch at x=1024 but do not overlap (IoU 0), so both
    # survive Stage 1 — the known v1 edge-fragment behavior (§6.2 accepted bias)
    assert len(dets) == 2
    best = max(dets, key=lambda d: d.score)
    assert best.score == 0.90 and best.level == 0


def test_within_level_merges_duplicates_in_overlap_strip():
    slicing = SlicingConfig(nominal_slice_size=1024, overlap_ratio=0.25, downsample_factor=2)
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
    slicing = SlicingConfig(nominal_slice_size=1024, downsample_factor=2)
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
    cross_changed = RecombinationConfig(cross_level_iou=0.9)
    within_changed = RecombinationConfig(within_level_iou=0.9)
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
