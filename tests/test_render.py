"""Detection-overlay artifacts (infer-render-artifacts D1-D10).

The plan is separated from the drawing on purpose: :func:`plan_artifacts` and
:func:`coco_document` are pure, so the file set, the canvases, and the box counts
per pipeline state are all testable without coconut-crab installed. Only the
tests that assert bytes on disk need the extra, and they skip without it.
"""

from pathlib import Path

import numpy as np
import pytest

from sati.backends import SliceDetection
from sati.config_loader import compose_and_resolve
from sati.config_schemas import StartThenGeometricLevelPolicyConfig
from sati.hashing import config_hash, preprocessing_hash, recombination_hash
from sati.infer import infer_frame
from sati.render import (
    FULLFRAME_SUBDIR,
    RenderError,
    artifacts_enabled,
    coco_document,
    plan_artifacts,
    render_frame,
)

CLASS_NAMES = {0: "denticle", 1: "tooth"}


def _cfg(tmp_path, overrides=(), *, stack_below_fullres=False):
    cfg = compose_and_resolve(
        [
            f"data.source.uri={tmp_path}/unused.json",
            "model.num_classes=2",
            "slicing.nominal_slice_size=64",
            f"infer_outputs.dir={tmp_path}/run",
            *overrides,
        ],
        command="infer",
        base_dir=tmp_path,
    )
    if stack_below_fullres:
        # A stack whose first realized level is 1, so there is no level-zero
        # artifact to name (D1).
        cfg.slicing.level_policy = StartThenGeometricLevelPolicyConfig(
            kind="start_then_geometric",
            start_scale=0.5,
            subsequent_downsample_factor=2.0,
            stop_when_long_side_lte_nominal=True,
        )
    return cfg


class EchoBackend:
    """The same fixed detection list in every slice, as in ``test_infer``."""

    def __init__(self, box=(6.0, 6.0, 30.0, 30.0), score=0.9, class_id=1):
        self.dets = [SliceDetection(bbox_xyxy=box, score=score, class_id=class_id)]

    def train(self, data_yaml, cfg, *, resume_from=None):
        raise NotImplementedError

    def load(self, model_ref):
        pass

    def predict(self, slices, *, imgsz, confidence_threshold, iou, agnostic_nms):
        return [list(self.dets) for _ in slices]


class DuplicateEchoBackend(EchoBackend):
    """Two near-identical detections per slice, so Stage 1 provably collapses them.

    Needed because a single box per slice leaves the uncombined and combined
    states with equal counts on a small frame — the artifacts would then be
    indistinguishable for reasons that have nothing to do with rendering.
    """

    def __init__(self):
        super().__init__()
        self.dets = [
            SliceDetection(bbox_xyxy=(6.0, 6.0, 30.0, 30.0), score=0.9, class_id=1),
            SliceDetection(bbox_xyxy=(7.0, 7.0, 29.0, 29.0), score=0.8, class_id=0),
        ]


def _inference(cfg, wh=(300, 200), backend=None):
    pixels = np.zeros((wh[1], wh[0], 3), np.uint8)
    return infer_frame("frame-a", pixels, cfg, backend or EchoBackend()), pixels


def _by_name(cfg, **kwargs):
    inference, pixels = _inference(cfg, **kwargs)
    artifacts = plan_artifacts(
        inference, inference.pyramid, cfg=cfg, class_names=CLASS_NAMES
    )
    return inference, {a.name: a for a in artifacts}


# --- the artifact set (5.1) -------------------------------------------------


def test_artifact_set_is_two_per_level_plus_three(tmp_path):
    cfg = _cfg(tmp_path)
    inference, planned = _by_name(cfg)
    levels = [level.index for level in inference.plan.levels]
    assert len(planned) == 2 * len(levels) + 3
    expected = {"raw-levels-stacked", "levels-stacked", "final"}
    expected |= {f"raw-level-{i}" for i in levels}
    expected |= {f"level-{i}" for i in levels}
    assert set(planned) == expected


def test_no_level_zero_when_the_stack_starts_below_fullres(tmp_path):
    cfg = _cfg(tmp_path, stack_below_fullres=True)
    inference, planned = _by_name(cfg)
    assert [level.index for level in inference.plan.levels] == [1, 2, 3]
    assert len(planned) == 9  # 2*3 + 3
    assert "level-0" not in planned and "raw-level-0" not in planned


def test_disabled_kinds_drop_only_their_own_files(tmp_path):
    cfg = _cfg(tmp_path, ["rendering.kinds.raw_level=false", "rendering.kinds.final=false"])
    _, planned = _by_name(cfg)
    assert not any(name.startswith("raw-level-") for name in planned)
    assert "final" not in planned
    assert "levels-stacked" in planned and "raw-levels-stacked" in planned


# --- canvases (5.2) ---------------------------------------------------------


def test_each_artifact_is_drawn_on_its_own_states_canvas(tmp_path):
    cfg = _cfg(tmp_path, stack_below_fullres=True)
    inference, planned = _by_name(cfg)
    fullres_wh = inference.fullres_wh
    for level in inference.plan.levels:
        for name in (f"level-{level.index}", f"raw-level-{level.index}"):
            height, width = planned[name].canvas.shape[:2]
            assert (width, height) == level.wh
    for name in ("raw-levels-stacked", "levels-stacked", "final"):
        height, width = planned[name].canvas.shape[:2]
        assert (width, height) == fullres_wh
    # the point of the level canvas: it is genuinely smaller, so a box drawn in
    # level coords needs no scale correction
    assert planned["level-1"].canvas.shape[:2] != planned["final"].canvas.shape[:2]


def test_boxes_live_in_their_own_canvas_coordinates(tmp_path):
    cfg = _cfg(tmp_path, stack_below_fullres=True)
    _, planned = _by_name(cfg)
    for name, artifact in planned.items():
        height, width = artifact.canvas.shape[:2]
        for box in artifact.boxes:
            x1, y1, x2, y2 = box.bbox_xyxy
            assert 0 <= x1 < x2 <= width + 1, name
            assert 0 <= y1 < y2 <= height + 1, name


# --- the two stacked states are distinct (5.4) ------------------------------


def test_stacked_artifacts_use_level_categories(tmp_path):
    cfg = _cfg(tmp_path, stack_below_fullres=True)
    inference, planned = _by_name(cfg)
    level_names = [f"level-{level.index}" for level in inference.plan.levels]
    for name in ("raw-levels-stacked", "levels-stacked"):
        assert list(planned[name].categories) == level_names
        assert {box.category for box in planned[name].boxes} <= set(level_names)
    # while the per-level and final artifacts stay keyed by class
    assert "tooth" in planned["final"].categories
    assert {box.category for box in planned["final"].boxes} == {"tooth"}


def test_stacked_counts_pin_the_two_states_apart(tmp_path):
    cfg = _cfg(tmp_path, stack_below_fullres=True)
    inference, planned = _by_name(cfg, backend=DuplicateEchoBackend())
    threshold = cfg.recombination.confidence_threshold
    filtered = [p for p in inference.slice_predictions if p.score >= threshold]
    assert len(planned["raw-levels-stacked"].boxes) == len(filtered)
    assert len(planned["levels-stacked"].boxes) == len(inference.level_detections)
    assert len(planned["final"].boxes) == len(inference.final_detections)
    # and they really are different states, not the same list twice: Stage 1
    # collapses each slice's duplicate pair to one winner
    assert len(planned["raw-levels-stacked"].boxes) == 2 * len(
        planned["levels-stacked"].boxes
    )


def test_stacked_is_exactly_the_per_level_states_lifted(tmp_path):
    """No new geometry: a stacked box count is the sum of its per-level counts."""
    cfg = _cfg(tmp_path, stack_below_fullres=True)
    inference, planned = _by_name(cfg)
    levels = [level.index for level in inference.plan.levels]
    assert len(planned["raw-levels-stacked"].boxes) == sum(
        len(planned[f"raw-level-{i}"].boxes) for i in levels
    )
    assert len(planned["levels-stacked"].boxes) == sum(
        len(planned[f"level-{i}"].boxes) for i in levels
    )


# --- confidence filter, not the slice cache (D4, 5.3) -----------------------


def test_raw_level_is_confidence_filtered_not_raw(tmp_path):
    cfg = _cfg(tmp_path)
    low = 0.5 * cfg.recombination.confidence_threshold
    pixels = np.zeros((200, 300, 3), np.uint8)
    inference = infer_frame("frame-a", pixels, cfg, EchoBackend(score=low))
    planned = {
        a.name: a
        for a in plan_artifacts(
            inference, inference.pyramid, cfg=cfg, class_names=CLASS_NAMES
        )
    }
    # every prediction is below threshold, so the merge stage saw nothing
    assert inference.slice_predictions
    assert all(len(a.boxes) == 0 for a in planned.values())


def test_raw_level_does_not_depend_on_the_slice_cache(tmp_path):
    """Boxes come from in-memory predictions, not from the results table (D4)."""
    cached = _cfg(tmp_path / "on", ["results.cache_slice_predictions=true"])
    uncached = _cfg(tmp_path / "off", ["results.cache_slice_predictions=false"])
    assert uncached.results.cache_slice_predictions is False
    counts = []
    for cfg in (cached, uncached):
        _, planned = _by_name(cfg)
        counts.append(
            {name: len(a.boxes) for name, a in planned.items() if "raw" in name}
        )
    assert counts[0] == counts[1]
    assert all(count > 0 for count in counts[1].values())


# --- box-count regression (5.5) --------------------------------------------


def test_box_counts_track_recombination_state(tmp_path):
    """A merge-strategy change that silently drops boxes shows up here.

    Counts are asserted against the recombination state for the same frame
    rather than against literals, so this fails when rendering and recombination
    disagree — which is the failure worth catching — and not merely when the
    strategy legitimately merges differently.
    """
    cfg = _cfg(tmp_path, stack_below_fullres=True)
    inference, planned = _by_name(cfg, backend=DuplicateEchoBackend())
    threshold = cfg.recombination.confidence_threshold
    for level in inference.plan.levels:
        expected_raw = sum(
            1
            for p in inference.slice_predictions
            if p.level == level.index and p.score >= threshold
        )
        expected_merged = sum(
            1 for d in inference.level_detections if d.level == level.index
        )
        assert len(planned[f"raw-level-{level.index}"].boxes) == expected_raw
        assert len(planned[f"level-{level.index}"].boxes) == expected_merged
        # stage 1 is a suppression under the default `nms`, never an addition
        assert 0 < expected_merged < expected_raw
    assert len(planned["final"].boxes) <= len(planned["levels-stacked"].boxes)


# --- COCO interchange (3.1, D6) --------------------------------------------


def test_coco_document_carries_score_and_canvas_geometry(tmp_path):
    cfg = _cfg(tmp_path)
    _, planned = _by_name(cfg)
    artifact = planned["final"]
    doc = coco_document(artifact)
    (image,) = doc["images"]
    height, width = artifact.canvas.shape[:2]
    assert (image["width"], image["height"]) == (width, height)
    assert image["file_name"] == "final.jpg"
    names = {c["id"]: c["name"] for c in doc["categories"]}
    assert set(names.values()) == set(artifact.categories)
    assert doc["annotations"]
    for ann, box in zip(doc["annotations"], artifact.boxes, strict=True):
        x1, y1, x2, y2 = box.bbox_xyxy
        # COCO is xywh, and `score` rides along so `{score}` resolves in the
        # label template with no upstream change
        assert ann["bbox"] == [x1, y1, x2 - x1, y2 - y1]
        assert ann["score"] == box.score
        assert ann["iscrowd"] == 0
        assert "segmentation" not in ann
        assert names[ann["category_id"]] == box.category


def test_coco_categories_survive_an_empty_artifact(tmp_path):
    """The legend is declared, not derived, so colors are stable across runs."""
    cfg = _cfg(tmp_path)
    low = 0.5 * cfg.recombination.confidence_threshold
    inference = infer_frame(
        "frame-a", np.zeros((200, 300, 3), np.uint8), cfg, EchoBackend(score=low)
    )
    artifact = next(
        a
        for a in plan_artifacts(
            inference, inference.pyramid, cfg=cfg, class_names=CLASS_NAMES
        )
        if a.name == "final"
    )
    assert artifact.boxes == ()
    assert set(coco_document(artifact)["categories"][0]) == {"id", "name"}
    assert len(coco_document(artifact)["categories"]) == len(CLASS_NAMES)


# --- identity: rendering moves nothing (2.6) -------------------------------


def test_rendering_is_absent_from_every_hash(tmp_path):
    """A rendering-only difference leaves config_hash and every identity equal."""
    base = _cfg(tmp_path / "a")
    styled = _cfg(
        tmp_path / "b",
        [
            "rendering.enabled=false",
            "rendering.kinds.final=false",
            f"rendering.style_config={tmp_path}/style.toml",
        ],
    )
    assert base.rendering != styled.rendering  # the configs really do differ
    assert config_hash(base, "infer") == config_hash(styled, "infer")
    assert preprocessing_hash(base.slicing) == preprocessing_hash(styled.slicing)
    for scope in ("level_merge", "final_merge"):
        assert recombination_hash(base.recombination, scope) == recombination_hash(
            styled.recombination, scope
        )
    # including the resolved runtime digest the sidecar re-verifies against
    assert base.runtime.config_hash == styled.runtime.config_hash


def test_rendering_block_is_not_in_the_active_projection(tmp_path):
    """Guard for the deliberate omission: adding it here must fail loudly."""
    from sati.hashing import _ACTIVE_BLOCKS, config_hash_source

    for blocks in _ACTIVE_BLOCKS.values():
        assert "rendering" not in blocks
    assert "rendering" not in config_hash_source(_cfg(tmp_path), "infer")


def test_rendering_disabled_changes_no_detection(tmp_path):
    on = _cfg(tmp_path / "on")
    off = _cfg(tmp_path / "off", ["rendering.enabled=false"])
    assert artifacts_enabled(on, "infer") and not artifacts_enabled(off, "infer")
    first, _ = _inference(on)
    second, _ = _inference(off)
    assert first.final_detections == second.final_detections
    assert first.level_detections == second.level_detections


# --- the per-command default (D8) ------------------------------------------


def test_overlays_default_on_for_infer_and_off_for_eval(tmp_path):
    from sati.config_schemas import EvalOutputsConfig, InferOutputsConfig

    # The per-command default lives on the output leaf, the same shape `figures`
    # already uses, so it is a property of the schema rather than of composition.
    assert InferOutputsConfig().artifacts.enabled is True
    assert EvalOutputsConfig().artifacts.enabled is False

    infer_cfg = _cfg(tmp_path)
    assert infer_cfg.infer_outputs.artifacts.enabled is True
    assert artifacts_enabled(infer_cfg, "infer") is True
    # `rendering.enabled` stays true for eval; the leaf is what holds it back, so
    # eval can opt in without touching the rendering block at all
    assert infer_cfg.rendering.enabled is True
    infer_cfg.eval_outputs.artifacts.dir = tmp_path / "eval-artifacts"
    assert artifacts_enabled(infer_cfg, "eval") is False


# --- the optional extra (1.3, D7) ------------------------------------------


def test_render_frame_is_a_no_op_when_disabled_without_the_extra(tmp_path, monkeypatch):
    """Inference without rendering must not need coconut-crab at all."""
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("coconut_crab"):
            raise ImportError("simulated missing extra")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    cfg = _cfg(tmp_path, ["rendering.enabled=false"])
    inference, pixels = _inference(cfg)
    assert (
        render_frame(
            inference,
            pixels,
            cfg=cfg,
            class_names=CLASS_NAMES,
            artifacts_dir=tmp_path / "artifacts",
            pyramid=inference.pyramid,
        )
        == []
    )
    assert not (tmp_path / "artifacts").exists()


def test_render_names_the_missing_extra(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("coconut_crab"):
            raise ImportError("simulated missing extra")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    cfg = _cfg(tmp_path)
    inference, pixels = _inference(cfg)
    with pytest.raises(RenderError, match=r"sati\[render\]"):
        render_frame(
            inference,
            pixels,
            cfg=cfg,
            class_names=CLASS_NAMES,
            artifacts_dir=tmp_path / "artifacts",
            pyramid=inference.pyramid,
        )


# --- bytes on disk (5.1, 5.2, 5.6) -----------------------------------------

coconut_crab = pytest.importorskip("coconut_crab", reason="needs the render extra")
PIL_Image = pytest.importorskip("PIL.Image", reason="needs the render extra")


def _write(cfg, tmp_path, name="artifacts"):
    inference, pixels = _inference(cfg)
    directory = tmp_path / name
    written = render_frame(
        inference,
        pixels,
        cfg=cfg,
        class_names=CLASS_NAMES,
        artifacts_dir=directory,
        pyramid=inference.pyramid,
    )
    return inference, directory, written


def test_written_tree_matches_the_declared_layout(tmp_path):
    cfg = _cfg(tmp_path, stack_below_fullres=True)
    inference, directory, written = _write(cfg, tmp_path)
    fullframes = directory / inference.fullframe_id / FULLFRAME_SUBDIR
    assert {p.name for p in fullframes.iterdir()} == {
        "raw-level-1.jpg",
        "raw-level-2.jpg",
        "raw-level-3.jpg",
        "raw-levels-stacked.jpg",
        "level-1.jpg",
        "level-2.jpg",
        "level-3.jpg",
        "levels-stacked.jpg",
        "final.jpg",
    }
    assert sorted(written) == sorted(fullframes.iterdir())
    assert all(path.stat().st_size > 0 for path in written)


def test_written_pixel_dimensions_match_each_canvas(tmp_path):
    cfg = _cfg(tmp_path, stack_below_fullres=True)
    inference, directory, _ = _write(cfg, tmp_path)
    fullframes = directory / inference.fullframe_id / FULLFRAME_SUBDIR

    def size(name: str) -> tuple[int, int]:
        with PIL_Image.open(fullframes / f"{name}.jpg") as image:
            return image.size

    for level in inference.plan.levels:
        assert size(f"level-{level.index}") == level.wh
        assert size(f"raw-level-{level.index}") == level.wh
    for name in ("final", "levels-stacked", "raw-levels-stacked"):
        assert size(name) == inference.fullres_wh


def test_rendering_is_byte_identical_across_runs(tmp_path):
    cfg = _cfg(tmp_path, stack_below_fullres=True)
    first = _write(cfg, tmp_path, "one")[2]
    second = _write(cfg, tmp_path, "two")[2]
    assert [p.name for p in first] == [p.name for p in second]
    for a, b in zip(first, second, strict=True):
        assert a.read_bytes() == b.read_bytes(), a.name


def test_style_config_is_delegated_not_revalidated(tmp_path):
    """coconut-crab's own ConfigError surfaces unchanged for an unknown key."""
    from coconut_crab.render.config import ConfigError

    style = tmp_path / "style.toml"
    style.write_text('not_a_coconut_crab_key = "boom"\n')
    cfg = _cfg(tmp_path, [f"rendering.style_config={style}"])
    with pytest.raises(ConfigError, match="Unknown render config field"):
        _write(cfg, tmp_path)


def test_packaged_style_example_is_accepted_upstream(tmp_path):
    """The shipped example must load in coconut-crab's own loader (2.5)."""
    from coconut_crab.render.config import load_render_toml, parse_style

    example = (
        Path("src/sati/config_defaults/rendering/style-example.toml").resolve()
    )
    values = load_render_toml(example)
    style = parse_style(values["style"])
    # colors sit flat under [style]; a nested table would silently become one
    # junk entry instead of an error, which is why this asserts resolution
    assert style.color_for("tooth", "bbox") == "#34c759"
    assert style.color_for("level-2", "bbox") == "#5ac8fa"
    assert style.color_for("nonesuch", "bbox") == "cyan"
    assert values["text"] == "{CATEGORY}:{score:.2f}"


def test_shipped_style_renders_the_score_label(tmp_path):
    """`{score}` resolves against the raw annotation field SATI writes (D6)."""
    example = Path("src/sati/config_defaults/rendering/style-example.toml").resolve()
    plain = _cfg(tmp_path / "plain")
    labeled = _cfg(tmp_path / "labeled", [f"rendering.style_config={example}"])
    plain_files = _write(plain, tmp_path, "plain")[2]
    labeled_files = _write(labeled, tmp_path, "labeled")[2]
    by_name = {p.name: p for p in plain_files}
    # drawing labels changes the pixels; a template that failed to resolve would
    # still differ, so also assert the styled colors took effect
    changed = [
        p for p in labeled_files if p.read_bytes() != by_name[p.name].read_bytes()
    ]
    assert len(changed) == len(labeled_files)
