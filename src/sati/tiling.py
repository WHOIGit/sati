"""Level stack, tiling + stretch, affines, projection, recombination (v1 §6).

Pure geometry with no I/O, deliberately — the property tests in
``tests/test_tiling.py`` are cheap because of it. Recombination (§6.2) is the
two stages at the bottom of this module: Stage 1 combines slice predictions
within each level, Stage 2 combines level detections across levels into
full-frame full-res detections — the only kind scored against ground truth.

Both stages cluster greedily by score under a configurable match metric
(``intersection_over_smaller`` by default, because IoU cannot match a
slice-clipped fragment to its whole detection) and then either discard the
non-winning members (``nms``, the default) or replace the winner's box with the
cluster envelope bounded by ``max_merge_growth`` (``greedy_nmm``).

Conventions (v1 §4): top-left origin, half-open ``xyxy`` boxes; slice origins
and dims are ints in *level* coordinates; transformed boxes are floats. Both
slice affines are ``[sx, sy, tx, ty]`` mapping **emitted-slice pixels**
(post-stretch) outward: ``x_level = sx·x_slice + tx``.

SAHI's ``overlap_ratio``/``min_area_ratio`` semantics are kept identical
(overlap in integer pixels: ``stride = nominal − int(nominal·overlap_ratio)``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sati.config_schemas import (
    AutoLevelPolicyConfig,
    RecombinationConfig,
    SlicingConfig,
    StartThenGeometricLevelPolicyConfig,
)

Box = tuple[float, float, float, float]

EDGE_LEFT = "left"
EDGE_RIGHT = "right"
EDGE_TOP = "top"
EDGE_BOTTOM = "bottom"


@dataclass(frozen=True)
class Affine:
    """Per-axis scale+translate: ``x' = sx·x + tx``, ``y' = sy·y + ty``."""

    sx: float
    sy: float
    tx: float
    ty: float

    def apply_box(self, box: Box) -> Box:
        x1, y1, x2, y2 = box
        return (
            self.sx * x1 + self.tx,
            self.sy * y1 + self.ty,
            self.sx * x2 + self.tx,
            self.sy * y2 + self.ty,
        )

    def invert(self) -> Affine:
        return Affine(
            sx=1.0 / self.sx,
            sy=1.0 / self.sy,
            tx=-self.tx / self.sx,
            ty=-self.ty / self.sy,
        )

    def compose(self, inner: Affine) -> Affine:
        """``self ∘ inner``: apply ``inner`` first, then ``self``."""
        return Affine(
            sx=self.sx * inner.sx,
            sy=self.sy * inner.sy,
            tx=self.sx * inner.tx + self.tx,
            ty=self.sy * inner.ty + self.ty,
        )

    def as_list(self) -> list[float]:
        return [self.sx, self.sy, self.tx, self.ty]


@dataclass(frozen=True)
class Level:
    """One entry in the downsample stack.

    Index 0 always means full-res (scale 1.0). A policy whose stack begins
    below full-res emits no level 0 — its levels are numbered from 1.
    """

    index: int
    scale_requested: float  # policy-defined; 1.0 ⇔ index 0
    width: int  # realized (post-floor) dims, min 1 px
    height: int

    @property
    def wh(self) -> tuple[int, int]:
        return (self.width, self.height)


@dataclass(frozen=True)
class SliceSpec:
    """Geometry of one emitted slice (the pixel-free half of a slices.parquet row)."""

    level: int
    slice_id: int  # unique across all levels of one fullframe
    slice_xy: tuple[int, int]  # window origin in level coords (pre-stretch)
    slice_wh: tuple[int, int]  # window size in level coords (pre-stretch)
    emitted_wh: tuple[int, int]  # always (nominal, nominal) in v1
    is_whole_level: bool
    stretched_axes: tuple[str, ...]  # subset of ("x", "y")
    affine_slice_to_level: Affine
    affine_slice_to_fullres: Affine


@dataclass(frozen=True)
class FramePlan:
    fullres_wh: tuple[int, int]
    levels: tuple[Level, ...]
    slices: tuple[SliceSpec, ...]

    def level(self, index: int) -> Level:
        # by index, not position: stacks that start below full-res begin at 1
        for level in self.levels:
            if level.index == index:
                return level
        raise KeyError(f"no level with index {index}")


@dataclass(frozen=True)
class ProjectedBox:
    """One annotation projected into one slice (§6.1); ``None`` if no overlap."""

    bbox_level_xyxy: Box
    bbox_slice_xyxy: Box  # clipped, in emitted (post-stretch) slice pixels
    visible_area_frac: float  # clipped area / level-box area, in level coords
    truncated_edges: tuple[str, ...]
    dropped_by_size_gate: bool
    dropped_by_min_area_ratio: bool

    @property
    def is_training_target(self) -> bool:
        return not (self.dropped_by_size_gate or self.dropped_by_min_area_ratio)


def _level_scale(position: int, slicing: SlicingConfig) -> float:
    """Scale of the stack entry at ``position`` (0-based position, not index)."""
    policy = slicing.level_policy
    if isinstance(policy, AutoLevelPolicyConfig):
        return 1.0 / policy.downsample_factor**position
    if isinstance(policy, StartThenGeometricLevelPolicyConfig):
        return policy.start_scale / policy.subsequent_downsample_factor**position
    raise TypeError(f"unsupported level policy: {policy!r}")


def _stop_at_nominal(slicing: SlicingConfig) -> bool:
    policy = slicing.level_policy
    if isinstance(policy, AutoLevelPolicyConfig):
        return True
    if isinstance(policy, StartThenGeometricLevelPolicyConfig):
        return policy.stop_when_long_side_lte_nominal
    raise TypeError(f"unsupported level policy: {policy!r}")


def build_level_stack(fullres_wh: tuple[int, int], slicing: SlicingConfig) -> tuple[Level, ...]:
    """Build the configured level stack.

    Auto policy preserves v1's original behavior: full-res level 0, each
    subsequent level divided by the policy's ``downsample_factor``.
    ``start_then_geometric`` starts at ``start_scale`` and divides by
    ``subsequent_downsample_factor``. Dims are always computed from full-res
    directly so floors do not accumulate.

    **Index convention:** level index 0 means full-res, always. A stack whose
    first entry is below full-res is numbered from 1, so the index keeps its
    meaning even when full-res is not among the slicing targets.
    """
    fullres_w, fullres_h = fullres_wh
    index_offset = 0 if _level_scale(0, slicing) == 1.0 else 1
    levels: list[Level] = []
    position = 0
    while True:
        scale = _level_scale(position, slicing)
        width = max(1, math.floor(fullres_w * scale))
        height = max(1, math.floor(fullres_h * scale))
        levels.append(
            Level(
                index=position + index_offset,
                scale_requested=scale,
                width=width,
                height=height,
            )
        )
        if _stop_at_nominal(slicing) and max(width, height) <= slicing.nominal_slice_size:
            break
        if slicing.max_levels is not None and len(levels) >= slicing.max_levels:
            break
        position += 1
    return tuple(levels)


@dataclass(frozen=True)
class AxisSpan:
    origin: int
    length: int
    stretched: bool


def tile_axis(axis_len: int, nominal: int, overlap_ratio: float) -> tuple[AxisSpan, ...]:
    """Tile one axis (§6.1): overlapping nominal tiles, last tile clamped flush.

    An axis shorter than nominal yields a single stretched span covering it.
    """
    if axis_len < nominal:
        return (AxisSpan(origin=0, length=axis_len, stretched=True),)
    stride = nominal - int(nominal * overlap_ratio)  # SAHI integer-pixel semantics
    origins: list[int] = []
    origin = 0
    while origin + nominal < axis_len:
        origins.append(origin)
        origin += stride
    origins.append(axis_len - nominal)  # last-tile rule: flush to the far edge
    return tuple(AxisSpan(origin=o, length=nominal, stretched=False) for o in origins)


def build_slices(
    fullres_wh: tuple[int, int], slicing: SlicingConfig
) -> tuple[tuple[Level, ...], tuple[SliceSpec, ...]]:
    """All slices of all levels for one frame; slice_id is frame-unique across levels."""
    fullres_w, fullres_h = fullres_wh
    nominal = slicing.nominal_slice_size
    levels = build_level_stack(fullres_wh, slicing)
    slices: list[SliceSpec] = []
    slice_id = 0
    for level in levels:
        level_to_fullres = Affine(
            sx=fullres_w / level.width,
            sy=fullres_h / level.height,
            tx=0.0,
            ty=0.0,
        )
        x_spans = tile_axis(level.width, nominal, slicing.overlap_ratio)
        y_spans = tile_axis(level.height, nominal, slicing.overlap_ratio)
        for y_span in y_spans:
            for x_span in x_spans:
                stretched = tuple(
                    axis for axis, span in (("x", x_span), ("y", y_span)) if span.stretched
                )
                slice_to_level = Affine(
                    sx=x_span.length / nominal,
                    sy=y_span.length / nominal,
                    tx=float(x_span.origin),
                    ty=float(y_span.origin),
                )
                slices.append(
                    SliceSpec(
                        level=level.index,
                        slice_id=slice_id,
                        slice_xy=(x_span.origin, y_span.origin),
                        slice_wh=(x_span.length, y_span.length),
                        emitted_wh=(nominal, nominal),
                        is_whole_level=(
                            (x_span.origin, y_span.origin) == (0, 0)
                            and (x_span.length, y_span.length) == level.wh
                        ),
                        stretched_axes=stretched,
                        affine_slice_to_level=slice_to_level,
                        affine_slice_to_fullres=level_to_fullres.compose(slice_to_level),
                    )
                )
                slice_id += 1
    return levels, tuple(slices)


def plan_frame(fullres_wh: tuple[int, int], slicing: SlicingConfig) -> FramePlan:
    levels, slices = build_slices(fullres_wh, slicing)
    return FramePlan(fullres_wh=fullres_wh, levels=levels, slices=slices)


def project_box(
    bbox_fullres_xyxy: Box,
    slice_spec: SliceSpec,
    level: Level,
    fullres_wh: tuple[int, int],
    slicing: SlicingConfig,
) -> ProjectedBox | None:
    """Project one full-res annotation into one slice (§6.1).

    Returns ``None`` when the box misses the slice window entirely; otherwise a
    row with both exclusion gates evaluated (recorded, never silently dropped).
    The size gate and ``visible_area_frac`` are evaluated in level coordinates —
    *before* stretch — on the clipped (visible) box.
    """
    fullres_w, fullres_h = fullres_wh
    fx1, fy1, fx2, fy2 = bbox_fullres_xyxy
    sx_lvl, sy_lvl = level.width / fullres_w, level.height / fullres_h
    lx1, ly1, lx2, ly2 = fx1 * sx_lvl, fy1 * sy_lvl, fx2 * sx_lvl, fy2 * sy_lvl

    wx1, wy1 = slice_spec.slice_xy
    wx2, wy2 = wx1 + slice_spec.slice_wh[0], wy1 + slice_spec.slice_wh[1]

    cx1, cy1 = max(lx1, wx1), max(ly1, wy1)
    cx2, cy2 = min(lx2, wx2), min(ly2, wy2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None

    level_area = (lx2 - lx1) * (ly2 - ly1)
    if level_area <= 0.0:
        return None
    visible_area_frac = ((cx2 - cx1) * (cy2 - cy1)) / level_area

    truncated = tuple(
        edge
        for edge, cut in (
            (EDGE_TOP, ly1 < wy1),
            (EDGE_BOTTOM, ly2 > wy2),
            (EDGE_LEFT, lx1 < wx1),
            (EDGE_RIGHT, lx2 > wx2),
        )
        if cut
    )

    bbox_slice = slice_spec.affine_slice_to_level.invert().apply_box((cx1, cy1, cx2, cy2))

    shorter_side_level = min(cx2 - cx1, cy2 - cy1)
    return ProjectedBox(
        bbox_level_xyxy=(lx1, ly1, lx2, ly2),
        bbox_slice_xyxy=bbox_slice,
        visible_area_frac=visible_area_frac,
        truncated_edges=truncated,
        dropped_by_size_gate=shorter_side_level < slicing.size_gate_px,
        dropped_by_min_area_ratio=visible_area_frac < slicing.min_area_ratio,
    )


# --- recombination (§6.2): two stages, configurable match + optional merge ---

# Per-strategy versions, not one global constant: adding `greedy_nmm` must not
# make `nms` look like it changed. Both start at 1.
MERGE_STRATEGY_VERSIONS: dict[str, int] = {"nms": 1, "greedy_nmm": 1}


def merge_strategy(recombination: "RecombinationConfig") -> tuple[str, int]:
    """Resolve ``(name, version)`` for the configured method."""
    name = recombination.method
    return name, MERGE_STRATEGY_VERSIONS[name]


@dataclass(frozen=True)
class SlicePrediction:
    """One raw detection in emitted-slice pixels (a ``slice_prediction`` row)."""

    slice_id: int
    level: int
    bbox_slice_xyxy: Box
    score: float
    class_id: int


@dataclass(frozen=True)
class LevelDetection:
    """One Stage-1 winner in level coordinates (a ``level_merge`` row)."""

    level: int
    bbox_level_xyxy: Box
    score: float
    class_id: int
    source_slice_ids: tuple[int, ...]  # winner's slice + every matched box's slice
    merged_class_ids: tuple[int, ...]  # winner's class first, then any absorbed


@dataclass(frozen=True)
class FinalDetection:
    """One Stage-2 winner in full-frame full-res coordinates (a ``final_merge`` row)."""

    bbox_fullres_xyxy: Box
    score: float
    class_id: int
    source_levels: tuple[int, ...]  # winner's level + every matched detection's level
    merged_class_ids: tuple[int, ...]


def box_area(b: Box) -> float:
    return (b[2] - b[0]) * (b[3] - b[1])


def _intersection(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def box_iou(a: Box, b: Box) -> float:
    inter = _intersection(a, b)
    if inter <= 0.0:
        return 0.0
    return inter / (box_area(a) + box_area(b) - inter)


def box_ios(a: Box, b: Box) -> float:
    """Intersection over the *smaller* box's area.

    The metric that can see a slice-clipped fragment. A fragment is a strict
    subset of its whole detection, so their IoU degenerates to the area ratio and
    can never clear a usable threshold; IoS is 1.0. This is why SAHI defaults to
    it, and it is SATI's default too.
    """
    inter = _intersection(a, b)
    if inter <= 0.0:
        return 0.0
    smaller = min(box_area(a), box_area(b))
    if smaller <= 0.0:
        return 0.0
    return inter / smaller


MATCH_METRICS = {
    "intersection_over_union": box_iou,
    "intersection_over_smaller": box_ios,
}


def envelope(boxes: list[Box]) -> Box:
    """Axis-aligned envelope (union extent) of one or more boxes."""
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def match_clusters(
    boxes: list[Box],
    scores: list[float],
    class_ids: list[int],
    threshold: float,
    class_agnostic: bool,
    match_metric: str = "intersection_over_union",
) -> list[tuple[int, list[int]]]:
    """Greedy score-ordered clustering; returns ``(winner, matched)`` pairs.

    Ranked by score, ties broken by input index for determinism. Matched
    candidates are visited in ascending index order. Matching uses the
    **original** winner box and runs in a single pass, so a box that later grows
    by merging never re-matches — SAHI's behavior, and what keeps this
    deterministic. A consequence documented rather than defended against:
    recombination is *not* idempotent.

    The comparison is strictly ``>``: a metric exactly equal to the threshold
    does not match. With ``intersection_over_smaller`` this is moot (a nested box
    scores 1.0), but a perfectly bisected fragment gives IoU exactly 0.5 and so
    would not match at threshold 0.5.

    Class-agnostic by default (see ``RecombinationConfig``): a box matches boxes
    of any class, and because the pass is score-ordered the most confident class
    wins with no extra tie-break logic.
    """
    metric = MATCH_METRICS[match_metric]
    order = sorted(range(len(boxes)), key=lambda i: (-scores[i], i))
    alive = set(order)
    clusters: list[tuple[int, list[int]]] = []
    for i in order:
        if i not in alive:
            continue
        alive.discard(i)
        matched: list[int] = []
        for j in sorted(alive):
            if not class_agnostic and class_ids[j] != class_ids[i]:
                continue
            if metric(boxes[i], boxes[j]) > threshold:
                matched.append(j)
                alive.discard(j)
        clusters.append((i, matched))
    return clusters


def merged_box(
    boxes: list[Box],
    winner: int,
    matched: list[int],
    recombination: "RecombinationConfig",
) -> tuple[Box, bool]:
    """The cluster's output box, plus whether the growth bound blocked a merge.

    Under ``nms`` the winner keeps its own box. Under ``greedy_nmm`` the box
    becomes the cluster envelope — but only when that envelope is at most
    ``max_merge_growth`` times the winner's own area.

    That bound is a deliberate departure from SAHI, which unions unconditionally.
    Measured motivation: a large false-positive blob nests around genuine small
    detections (IoS 1.0, IoU 0.002), and because the small box outscores it, an
    unbounded union replaced a 137x97 box with a 2658x1206 one — 248x inflation.
    Matched boxes are consumed either way, so a blocked merge degrades that one
    cluster to ``nms`` behavior and the detection count is never affected.
    """
    win = boxes[winner]
    if recombination.method == "nms" or not matched:
        return win, False
    box = envelope([boxes[k] for k in (winner, *matched)])
    if box_area(box) > recombination.max_merge_growth * box_area(win):
        return win, True
    return box, False


def cluster_class_ids(class_ids: list[int], winner: int, matched: list[int]) -> tuple[int, ...]:
    """Distinct classes in the cluster, winner's first, remainder ascending.

    Class-agnostic matching collapses a class disagreement into one detection;
    without this the losing class is unrecoverable, and near-ties do occur.
    """
    others = sorted({class_ids[k] for k in matched} - {class_ids[winner]})
    return (class_ids[winner], *others)


def merge_within_level(
    predictions: list[SlicePrediction],
    plan: FramePlan,
    recombination: "RecombinationConfig",
) -> tuple[LevelDetection, ...]:
    """Stage 1: per level — confidence filter, lift to level coords, match, merge."""
    detections, _ = merge_within_level_counted(predictions, plan, recombination)
    return detections


def merge_within_level_counted(
    predictions: list[SlicePrediction],
    plan: FramePlan,
    recombination: "RecombinationConfig",
) -> tuple[tuple[LevelDetection, ...], int]:
    """As ``merge_within_level``, also returning the growth-bound block count."""
    slices_by_id = {s.slice_id: s for s in plan.slices}
    detections: list[LevelDetection] = []
    capped = 0
    for level in plan.levels:
        lifted: list[SlicePrediction] = []
        boxes: list[Box] = []
        for pred in predictions:
            if pred.level != level.index:
                continue
            if pred.score < recombination.confidence_threshold:
                continue
            spec = slices_by_id[pred.slice_id]
            lifted.append(pred)
            boxes.append(spec.affine_slice_to_level.apply_box(pred.bbox_slice_xyxy))
        class_ids = [p.class_id for p in lifted]
        clusters = match_clusters(
            boxes,
            [p.score for p in lifted],
            class_ids,
            recombination.within_level_threshold,
            recombination.class_agnostic,
            recombination.match_metric,
        )
        for winner, matched in clusters:
            box, blocked = merged_box(boxes, winner, matched, recombination)
            capped += blocked
            contributors = sorted({lifted[k].slice_id for k in (winner, *matched)})
            detections.append(
                LevelDetection(
                    level=level.index,
                    bbox_level_xyxy=box,
                    score=lifted[winner].score,
                    class_id=lifted[winner].class_id,
                    source_slice_ids=tuple(contributors),
                    merged_class_ids=cluster_class_ids(class_ids, winner, matched),
                )
            )
    return tuple(detections), capped


def merge_cross_level(
    level_detections: list[LevelDetection] | tuple[LevelDetection, ...],
    plan: FramePlan,
    recombination: "RecombinationConfig",
) -> tuple[FinalDetection, ...]:
    """Stage 2: lift to full-res by realized scale, match ranked by raw confidence."""
    detections, _ = merge_cross_level_counted(level_detections, plan, recombination)
    return detections


def merge_cross_level_counted(
    level_detections: list[LevelDetection] | tuple[LevelDetection, ...],
    plan: FramePlan,
    recombination: "RecombinationConfig",
) -> tuple[tuple[FinalDetection, ...], int]:
    """As ``merge_cross_level``, also returning the growth-bound block count."""
    fullres_w, fullres_h = plan.fullres_wh
    dets = list(level_detections)
    boxes: list[Box] = []
    for det in dets:
        level = plan.level(det.level)
        lift = Affine(
            sx=fullres_w / level.width, sy=fullres_h / level.height, tx=0.0, ty=0.0
        )
        boxes.append(lift.apply_box(det.bbox_level_xyxy))
    class_ids = [d.class_id for d in dets]
    clusters = match_clusters(
        boxes,
        [d.score for d in dets],
        class_ids,
        recombination.cross_level_threshold,
        recombination.class_agnostic,
        recombination.match_metric,
    )
    finals: list[FinalDetection] = []
    capped = 0
    for winner, matched in clusters:
        box, blocked = merged_box(boxes, winner, matched, recombination)
        capped += blocked
        # every class already absorbed by any cluster member, not just its winner
        absorbed = {c for k in (winner, *matched) for c in dets[k].merged_class_ids}
        finals.append(
            FinalDetection(
                bbox_fullres_xyxy=box,
                score=dets[winner].score,
                class_id=dets[winner].class_id,
                source_levels=tuple(sorted({dets[k].level for k in (winner, *matched)})),
                merged_class_ids=(
                    dets[winner].class_id,
                    *sorted(absorbed - {dets[winner].class_id}),
                ),
            )
        )
    return tuple(finals), capped


@dataclass(frozen=True)
class Recombination:
    """Both stages' output for one frame, plus growth-bound diagnostics."""

    level_detections: tuple[LevelDetection, ...]
    final_detections: tuple[FinalDetection, ...]
    within_level_capped: int
    cross_level_capped: int

    @property
    def capped(self) -> int:
        return self.within_level_capped + self.cross_level_capped


def recombine(
    predictions: list[SlicePrediction],
    plan: FramePlan,
    recombination: "RecombinationConfig",
) -> tuple[tuple[LevelDetection, ...], tuple[FinalDetection, ...]]:
    """Both stages for one frame; returns (level_merge rows, final_merge rows)."""
    result = recombine_counted(predictions, plan, recombination)
    return result.level_detections, result.final_detections


def recombine_counted(
    predictions: list[SlicePrediction],
    plan: FramePlan,
    recombination: "RecombinationConfig",
) -> Recombination:
    """Both stages plus how often the growth bound blocked a merge.

    The count is what makes under-restoration observable: a fragment smaller than
    ``1 / max_merge_growth`` of its object falls back to suppression instead of
    being restored, and without a count that is invisible. Always zero under
    ``method: nms``.
    """
    level_detections, within = merge_within_level_counted(predictions, plan, recombination)
    finals, cross = merge_cross_level_counted(level_detections, plan, recombination)
    return Recombination(
        level_detections=level_detections,
        final_detections=finals,
        within_level_capped=within,
        cross_level_capped=cross,
    )
