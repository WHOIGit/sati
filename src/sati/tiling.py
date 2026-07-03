"""Level stack, tiling + stretch, affines, projection, recombination (v1 §6).

Pure geometry with no I/O, deliberately — the property tests in
``tests/test_tiling.py`` are cheap because of it. Recombination (§6.2) is the
two-stage NMS at the bottom of this module: Stage 1 merges slice predictions
within each level, Stage 2 merges level detections across levels into
full-frame full-res detections — the only kind scored against ground truth.

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

from sati.config_schemas import RecombinationConfig, SlicingConfig

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
    """One entry in the downsample stack. Level 0 is full-res."""

    index: int
    scale_requested: float  # 1 / factor**index
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
        return self.levels[index]


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


def build_level_stack(fullres_wh: tuple[int, int], slicing: SlicingConfig) -> tuple[Level, ...]:
    """Automatic-factor stack (§6.1): divide until the *longer* side fits nominal.

    Level *k* dims are ``floor(fullres / factor**k)`` per axis (min 1 px),
    computed from full-res directly so floors do not accumulate. The stack
    includes the first level whose ``max(w, h) ≤ nominal`` (the terminal
    whole-frame level) unless ``max_levels`` caps it first.
    """
    fullres_w, fullres_h = fullres_wh
    factor = slicing.downsample_factor
    levels: list[Level] = []
    k = 0
    while True:
        scale = 1.0 / factor**k
        width = max(1, math.floor(fullres_w * scale))
        height = max(1, math.floor(fullres_h * scale))
        levels.append(Level(index=k, scale_requested=scale, width=width, height=height))
        if max(width, height) <= slicing.nominal_slice_size:
            break
        if slicing.max_levels is not None and len(levels) >= slicing.max_levels:
            break
        k += 1
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


# --- recombination (§6.2): two stages, NMS only ----------------------------

MERGE_STRATEGY_NAME = "nms"
MERGE_STRATEGY_VERSION = 1


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
    source_slice_ids: tuple[int, ...]  # winner's slice + every suppressed box's slice


@dataclass(frozen=True)
class FinalDetection:
    """One Stage-2 winner in full-frame full-res coordinates (a ``final_merge`` row)."""

    bbox_fullres_xyxy: Box
    score: float
    class_id: int
    source_levels: tuple[int, ...]  # winner's level + every suppressed detection's level


def box_iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def greedy_nms(
    boxes: list[Box],
    scores: list[float],
    class_ids: list[int],
    iou_threshold: float,
    class_agnostic: bool,
) -> list[tuple[int, list[int]]]:
    """Greedy NMS; returns ``(winner_index, suppressed_indices)`` pairs.

    Ranked by score, ties broken by input index for determinism. Class-aware by
    default: a box only suppresses boxes of its own class unless
    ``class_agnostic``.
    """
    order = sorted(range(len(boxes)), key=lambda i: (-scores[i], i))
    alive = set(order)
    winners: list[tuple[int, list[int]]] = []
    for i in order:
        if i not in alive:
            continue
        alive.discard(i)
        suppressed: list[int] = []
        for j in sorted(alive):
            if not class_agnostic and class_ids[j] != class_ids[i]:
                continue
            if box_iou(boxes[i], boxes[j]) > iou_threshold:
                suppressed.append(j)
                alive.discard(j)
        winners.append((i, suppressed))
    return winners


def merge_within_level(
    predictions: list[SlicePrediction],
    plan: FramePlan,
    recombination: "RecombinationConfig",
) -> tuple[LevelDetection, ...]:
    """Stage 1: per level — confidence filter, lift to level coords, NMS."""
    slices_by_id = {s.slice_id: s for s in plan.slices}
    detections: list[LevelDetection] = []
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
        winners = greedy_nms(
            boxes,
            [p.score for p in lifted],
            [p.class_id for p in lifted],
            recombination.within_level_iou,
            recombination.class_agnostic,
        )
        for winner, suppressed in winners:
            contributors = sorted({lifted[k].slice_id for k in (winner, *suppressed)})
            detections.append(
                LevelDetection(
                    level=level.index,
                    bbox_level_xyxy=boxes[winner],
                    score=lifted[winner].score,
                    class_id=lifted[winner].class_id,
                    source_slice_ids=tuple(contributors),
                )
            )
    return tuple(detections)


def merge_cross_level(
    level_detections: list[LevelDetection] | tuple[LevelDetection, ...],
    plan: FramePlan,
    recombination: "RecombinationConfig",
) -> tuple[FinalDetection, ...]:
    """Stage 2: lift to full-res by realized scale, NMS ranked by raw confidence."""
    fullres_w, fullres_h = plan.fullres_wh
    dets = list(level_detections)
    boxes: list[Box] = []
    for det in dets:
        level = plan.level(det.level)
        lift = Affine(
            sx=fullres_w / level.width, sy=fullres_h / level.height, tx=0.0, ty=0.0
        )
        boxes.append(lift.apply_box(det.bbox_level_xyxy))
    winners = greedy_nms(
        boxes,
        [d.score for d in dets],
        [d.class_id for d in dets],
        recombination.cross_level_iou,
        recombination.class_agnostic,
    )
    return tuple(
        FinalDetection(
            bbox_fullres_xyxy=boxes[winner],
            score=dets[winner].score,
            class_id=dets[winner].class_id,
            source_levels=tuple(sorted({dets[k].level for k in (winner, *suppressed)})),
        )
        for winner, suppressed in winners
    )


def recombine(
    predictions: list[SlicePrediction],
    plan: FramePlan,
    recombination: "RecombinationConfig",
) -> tuple[tuple[LevelDetection, ...], tuple[FinalDetection, ...]]:
    """Both stages for one frame; returns (level_merge rows, final_merge rows)."""
    level_detections = merge_within_level(predictions, plan, recombination)
    return level_detections, merge_cross_level(level_detections, plan, recombination)
