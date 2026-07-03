"""Property tests for sati.tiling (DESIGN-DOC-v1 §12 testing spine).

Random frame sizes / factors / overlaps → slice → project → invert affine →
assert round-trip within float tolerance, and assert full coverage of every
level by its tiles. Seeded RNG, no hypothesis dependency.
"""

import math

import numpy as np
import pytest

from sati.config_schemas import SlicingConfig
from sati.tiling import (
    Affine,
    build_level_stack,
    plan_frame,
    project_box,
    tile_axis,
)

RNG = np.random.default_rng(20260702)


def random_configs(n):
    # Ranges bounded to keep slice counts sane (overlap 0.9 at nominal 64 over a
    # 12000px frame explodes into millions of slices); coverage of the extreme
    # overlap/nominal corners is exercised directly in test_axis_tiling_*.
    for _ in range(n):
        yield (
            (int(RNG.integers(40, 8000)), int(RNG.integers(40, 8000))),  # fullres_wh
            SlicingConfig(
                nominal_slice_size=int(RNG.integers(128, 2049)),
                overlap_ratio=float(RNG.uniform(0.0, 0.5)),
                downsample_factor=float(RNG.uniform(1.2, 4.0)),
            ),
        )


# --- affines ---------------------------------------------------------------


def test_affine_invert_compose_roundtrip():
    aff = Affine(sx=0.37, sy=2.5, tx=123.0, ty=-7.5)
    box = (10.0, 20.0, 110.0, 220.0)
    assert aff.invert().apply_box(aff.apply_box(box)) == pytest.approx(box)
    ident = aff.compose(aff.invert())
    assert ident.apply_box(box) == pytest.approx(box)


# --- level stack -----------------------------------------------------------


def test_level_stack_properties():
    for fullres_wh, cfg in random_configs(200):
        levels = build_level_stack(fullres_wh, cfg)
        assert levels[0].wh == fullres_wh  # level 0 is full-res
        # dims computed from full-res directly, floors never accumulate
        for level in levels:
            assert level.width == max(1, math.floor(fullres_wh[0] / cfg.downsample_factor**level.index))
            assert level.height == max(1, math.floor(fullres_wh[1] / cfg.downsample_factor**level.index))
        # terminal level fits nominal on the longer side; earlier levels don't
        assert max(levels[-1].wh) <= cfg.nominal_slice_size or (
            cfg.max_levels is not None and len(levels) == cfg.max_levels
        )
        for level in levels[:-1]:
            assert max(level.wh) > cfg.nominal_slice_size


def test_level_stack_max_levels_cap():
    cfg = SlicingConfig(nominal_slice_size=256, downsample_factor=2, max_levels=2)
    levels = build_level_stack((8192, 8192), cfg)
    assert len(levels) == 2


def test_single_level_when_frame_fits():
    cfg = SlicingConfig(nominal_slice_size=1024)
    levels = build_level_stack((800, 600), cfg)
    assert len(levels) == 1


# --- axis tiling -----------------------------------------------------------


def test_axis_tiling_coverage_and_last_tile_rule():
    for _ in range(500):
        nominal = int(RNG.integers(32, 2049))
        axis_len = int(RNG.integers(1, 20000))
        overlap = float(RNG.uniform(0.0, 0.95))
        spans = tile_axis(axis_len, nominal, overlap)
        if axis_len < nominal:
            assert spans == tile_axis(axis_len, nominal, overlap)  # deterministic
            assert len(spans) == 1 and spans[0].stretched and spans[0].length == axis_len
            continue
        # all tiles full nominal length, in bounds
        assert all(s.length == nominal and not s.stretched for s in spans)
        assert spans[0].origin == 0
        assert spans[-1].origin == axis_len - nominal  # flush to far edge
        # full coverage without gaps
        covered_to = 0
        for span in spans:
            assert span.origin <= covered_to  # no gap
            covered_to = max(covered_to, span.origin + span.length)
        assert covered_to == axis_len


# --- slices ----------------------------------------------------------------


def test_slice_properties():
    for fullres_wh, cfg in random_configs(50):
        plan = plan_frame(fullres_wh, cfg)
        ids = [s.slice_id for s in plan.slices]
        assert ids == sorted(ids) and len(set(ids)) == len(ids)  # frame-unique
        for s in plan.slices:
            level = plan.level(s.level)
            # window in level bounds
            assert 0 <= s.slice_xy[0] and s.slice_xy[0] + s.slice_wh[0] <= level.width
            assert 0 <= s.slice_xy[1] and s.slice_xy[1] + s.slice_wh[1] <= level.height
            assert s.emitted_wh == (cfg.nominal_slice_size, cfg.nominal_slice_size)
            # stretch bookkeeping matches window vs nominal
            assert ("x" in s.stretched_axes) == (s.slice_wh[0] < cfg.nominal_slice_size)
            assert ("y" in s.stretched_axes) == (s.slice_wh[1] < cfg.nominal_slice_size)
        # terminal whole-frame level is one whole-level slice (unless capped)
        last = plan.levels[-1]
        if max(last.wh) <= cfg.nominal_slice_size:
            last_slices = [s for s in plan.slices if s.level == last.index]
            assert len(last_slices) == 1 and last_slices[0].is_whole_level


def test_affine_roundtrip_through_slices():
    for fullres_wh, cfg in random_configs(30):
        plan = plan_frame(fullres_wh, cfg)
        for s in plan.slices:
            # emitted-slice corners map into the level window and back exactly
            emitted_box = (0.0, 0.0, float(s.emitted_wh[0]), float(s.emitted_wh[1]))
            level_box = s.affine_slice_to_level.apply_box(emitted_box)
            assert level_box == pytest.approx(
                (
                    s.slice_xy[0],
                    s.slice_xy[1],
                    s.slice_xy[0] + s.slice_wh[0],
                    s.slice_xy[1] + s.slice_wh[1],
                ),
                abs=1e-6,
            )
            back = s.affine_slice_to_level.invert().apply_box(level_box)
            assert back == pytest.approx(emitted_box, abs=1e-6)
            # slice→fullres composes level→fullres over slice→level
            level = plan.level(s.level)
            fullres_box = s.affine_slice_to_fullres.apply_box(emitted_box)
            expected = (
                level_box[0] * fullres_wh[0] / level.width,
                level_box[1] * fullres_wh[1] / level.height,
                level_box[2] * fullres_wh[0] / level.width,
                level_box[3] * fullres_wh[1] / level.height,
            )
            assert fullres_box == pytest.approx(expected, rel=1e-9, abs=1e-6)


# --- annotation projection -------------------------------------------------


def _random_box(fullres_wh):
    w, h = fullres_wh
    x1 = float(RNG.uniform(0, w - 2))
    y1 = float(RNG.uniform(0, h - 2))
    return (x1, y1, float(RNG.uniform(x1 + 1, w)), float(RNG.uniform(y1 + 1, h)))


def test_projection_roundtrip_and_gates():
    for fullres_wh, cfg in random_configs(20):
        plan = plan_frame(fullres_wh, cfg)
        for _ in range(10):
            box = _random_box(fullres_wh)
            seen_anywhere = False
            for s in plan.slices:
                level = plan.level(s.level)
                proj = project_box(box, s, level, fullres_wh, cfg)
                if proj is None:
                    continue
                seen_anywhere = True
                ex1, ey1, ex2, ey2 = proj.bbox_slice_xyxy
                # clipped slice box inside emitted domain
                assert -1e-6 <= ex1 <= ex2 <= s.emitted_wh[0] + 1e-6
                assert -1e-6 <= ey1 <= ey2 <= s.emitted_wh[1] + 1e-6
                assert 0.0 < proj.visible_area_frac <= 1.0 + 1e-9
                # slice box maps back to the clipped level box
                back = s.affine_slice_to_level.apply_box(proj.bbox_slice_xyxy)
                wx1, wy1 = s.slice_xy
                wx2, wy2 = wx1 + s.slice_wh[0], wy1 + s.slice_wh[1]
                lx1, ly1, lx2, ly2 = proj.bbox_level_xyxy
                expected = (
                    max(lx1, wx1),
                    max(ly1, wy1),
                    min(lx2, wx2),
                    min(ly2, wy2),
                )
                assert back == pytest.approx(expected, abs=1e-6)
                # untruncated ⇔ fully visible
                if not proj.truncated_edges:
                    assert proj.visible_area_frac == pytest.approx(1.0)
                    assert not proj.dropped_by_min_area_ratio or cfg.min_area_ratio > 1.0 - 1e-9
            # every box overlaps at least the whole-frame terminal level slice
            if max(plan.levels[-1].wh) <= cfg.nominal_slice_size:
                assert seen_anywhere


def test_projection_gate_flags():
    cfg = SlicingConfig(nominal_slice_size=512, size_gate_px=16, min_area_ratio=0.2)
    plan = plan_frame((512, 512), cfg)
    (s,) = plan.slices
    level = plan.level(0)
    tiny = project_box((10.0, 10.0, 20.0, 20.0), s, level, (512, 512), cfg)
    assert tiny is not None and tiny.dropped_by_size_gate and not tiny.dropped_by_min_area_ratio
    ok = project_box((10.0, 10.0, 60.0, 60.0), s, level, (512, 512), cfg)
    assert ok is not None and ok.is_training_target
    outside = project_box((600.0, 600.0, 700.0, 700.0), s, level, (512, 512), cfg)
    assert outside is None


def test_projection_min_area_ratio_flag():
    cfg = SlicingConfig(nominal_slice_size=256, overlap_ratio=0.0, min_area_ratio=0.5, size_gate_px=4)
    plan = plan_frame((512, 512), cfg)
    level0 = plan.level(0)
    first = next(s for s in plan.slices if s.level == 0 and s.slice_xy == (0, 0))
    # box mostly outside the first tile: visible sliver < 50%
    proj = project_box((200.0, 10.0, 400.0, 60.0), first, level0, (512, 512), cfg)
    assert proj is not None
    assert proj.visible_area_frac == pytest.approx(56 / 200)
    assert proj.dropped_by_min_area_ratio and not proj.is_training_target
    assert proj.truncated_edges == ("right",)
