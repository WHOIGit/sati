"""Detection overlays: a visual trace of the two-stage recombination pipeline.

One image per pipeline state, so a box's appearance or disappearance is
attributable to a specific stage rather than inferred from Parquet queries::

       per level (level canvas)              stacked (full-res canvas)

       raw-level-N.jpg  ───────────────────▶ raw-levels-stacked.jpg
            │  Stage 1: within-level merge        │
            ▼                                     ▼
       level-N.jpg      ───────────────────▶ levels-stacked.jpg
                            lift + overlay        │
                                                  ▼  Stage 2: cross-level merge
                                             final.jpg

``2N + 3`` files under ``artifacts/<image-stem>/fullframes/``.

**Division of labor** (infer-render-artifacts D5). This module owns *structure* —
which images exist, their layout, naming, canvas, level selection, and the
confidence filter — and computes no new geometry: every box comes from
``slice_prediction``, ``level_merge``, or ``final_merge`` state that
:mod:`sati.tiling` already produced. *Style* — colors, line weight, label
template, text placement, encoding — belongs to coconut-crab's own TOML,
reached through ``rendering.style_config`` and not re-validated here.

**Nothing here moves an identity.** Rendering is downstream observation; it
cannot change a detection, a metric, or a result row, and ``rendering`` appears
in no ``hashing._ACTIVE_BLOCKS`` entry precisely so an overlay never invalidates
a result.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from sati.config_schemas import RootConfig
from sati.errors import SatiError
from sati.materialize import LevelPyramid, build_level_pyramid
from sati.tiling import Affine, Box, FramePlan, LevelDetection, SlicePrediction

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sati.infer import FrameInference

# The overlays live one level down from `<image-stem>/`, reserving
# `artifacts/<stem>/slices/` for the per-slice overlays DESIGN-DOC-v1 §9
# anticipates. Naming it now costs nothing and avoids a later move.
FULLFRAME_SUBDIR = "fullframes"

IMAGE_SUFFIX = ".jpg"

_MISSING_EXTRA = (
    "rendering is enabled but coconut-crab is not installed. Install the render "
    "extra (`pip install 'sati[render]'`), or set `rendering.enabled=false`."
)


class RenderError(SatiError):
    """Rendering was requested but cannot be performed."""


@dataclass(frozen=True)
class RenderBox:
    """One box to draw, already in its canvas's coordinate space."""

    bbox_xyxy: Box
    score: float
    category: str


@dataclass(frozen=True)
class Artifact:
    """One planned overlay: a name, a canvas, and the boxes on it."""

    name: str
    canvas: np.ndarray
    boxes: tuple[RenderBox, ...]
    # Declared rather than derived from `boxes` so an artifact with zero
    # detections still gets its full color legend, and so a category a strategy
    # happened to eliminate does not silently change color between two runs.
    categories: tuple[str, ...]


# --- geometry: reuse, never re-derive ---------------------------------------


def _level_lift(plan: FramePlan, level_index: int) -> Affine:
    """Level coords → full-res, by realized scale.

    Identical to the lift ``tiling.merge_cross_level_counted`` applies, so a
    stacked artifact draws a box exactly where Stage 2 matched it.
    """
    fullres_w, fullres_h = plan.fullres_wh
    level = plan.level(level_index)
    return Affine(
        sx=fullres_w / level.width, sy=fullres_h / level.height, tx=0.0, ty=0.0
    )


def filtered_predictions(
    predictions: Sequence[SlicePrediction],
    plan: FramePlan,
    level_index: int,
    confidence_threshold: float,
) -> list[tuple[SlicePrediction, Box]]:
    """One level's slice predictions at or above threshold, lifted to level coords.

    Confidence-filtered rather than raw-raw (D4): the filter is the first thing
    Stage 1 does, so an unfiltered image would attribute the filter's removals
    to merging. Reads in-memory predictions, so this does not depend on
    ``results.cache_slice_predictions``.
    """
    slices_by_id = {s.slice_id: s for s in plan.slices}
    out: list[tuple[SlicePrediction, Box]] = []
    for pred in predictions:
        if pred.level != level_index or pred.score < confidence_threshold:
            continue
        spec = slices_by_id[pred.slice_id]
        out.append((pred, spec.affine_slice_to_level.apply_box(pred.bbox_slice_xyxy)))
    return out


def class_name(class_id: int, class_names: dict[int, str]) -> str:
    return class_names.get(class_id, f"class-{class_id}")


def level_name(level_index: int) -> str:
    """The synthetic category name both stacked artifacts color by (D2)."""
    return f"level-{level_index}"


# --- the artifact plan ------------------------------------------------------


def plan_artifacts(
    inference: "FrameInference",
    pyramid: LevelPyramid,
    *,
    cfg: RootConfig,
    class_names: dict[int, str],
) -> list[Artifact]:
    """Every enabled overlay for one frame: name, canvas, boxes, legend.

    Pure — resolves no paths and writes nothing, so the file set and the box
    counts per artifact are testable without a rendering backend installed.
    """
    plan = inference.plan
    kinds = cfg.rendering.kinds
    threshold = cfg.recombination.confidence_threshold
    class_categories = tuple(
        class_name(class_id, class_names)
        for class_id in sorted({*class_names, *_observed_class_ids(inference)})
    )
    level_categories = tuple(level_name(level.index) for level in plan.levels)

    level_dets_by_index: dict[int, list[LevelDetection]] = {}
    for det in inference.level_detections:
        level_dets_by_index.setdefault(det.level, []).append(det)

    artifacts: list[Artifact] = []
    stacked_raw: list[RenderBox] = []
    stacked_combined: list[RenderBox] = []

    for level in plan.levels:
        lifted = filtered_predictions(
            inference.slice_predictions, plan, level.index, threshold
        )
        combined = level_dets_by_index.get(level.index, [])
        to_fullres = _level_lift(plan, level.index)

        if kinds.raw_level:
            artifacts.append(
                Artifact(
                    name=f"raw-level-{level.index}",
                    canvas=pyramid.pixels[level.index],
                    boxes=tuple(
                        RenderBox(box, pred.score, class_name(pred.class_id, class_names))
                        for pred, box in lifted
                    ),
                    categories=class_categories,
                )
            )
        if kinds.level:
            artifacts.append(
                Artifact(
                    name=f"level-{level.index}",
                    canvas=pyramid.pixels[level.index],
                    boxes=tuple(
                        RenderBox(
                            det.bbox_level_xyxy,
                            det.score,
                            class_name(det.class_id, class_names),
                        )
                        for det in combined
                    ),
                    categories=class_categories,
                )
            )
        # The stacked artifacts are the same two states lifted to full-res and
        # relabeled by level, not a separate computation.
        if kinds.raw_levels_stacked:
            stacked_raw.extend(
                RenderBox(to_fullres.apply_box(box), pred.score, level_name(level.index))
                for pred, box in lifted
            )
        if kinds.levels_stacked:
            stacked_combined.extend(
                RenderBox(
                    to_fullres.apply_box(det.bbox_level_xyxy),
                    det.score,
                    level_name(level.index),
                )
                for det in combined
            )

    if kinds.raw_levels_stacked:
        artifacts.append(
            Artifact(
                name="raw-levels-stacked",
                canvas=pyramid.fullres,
                boxes=tuple(stacked_raw),
                categories=level_categories,
            )
        )
    if kinds.levels_stacked:
        artifacts.append(
            Artifact(
                name="levels-stacked",
                canvas=pyramid.fullres,
                boxes=tuple(stacked_combined),
                categories=level_categories,
            )
        )
    if kinds.final:
        artifacts.append(
            Artifact(
                name="final",
                canvas=pyramid.fullres,
                boxes=tuple(
                    RenderBox(
                        det.bbox_fullres_xyxy,
                        det.score,
                        class_name(det.class_id, class_names),
                    )
                    for det in inference.final_detections
                ),
                categories=class_categories,
            )
        )
    return artifacts


def _observed_class_ids(inference: "FrameInference") -> set[int]:
    """Classes present in the frame, so an absent class map still legends."""
    return {
        *(p.class_id for p in inference.slice_predictions),
        *(d.class_id for d in inference.level_detections),
        *(d.class_id for d in inference.final_detections),
    }


# --- COCO interchange -------------------------------------------------------


def coco_document(artifact: Artifact) -> dict[str, Any]:
    """The in-memory COCO document for one artifact (D6).

    ``bbox`` is COCO's ``[x, y, w, h]`` in this canvas's own coordinate space,
    so no artifact needs a scale correction applied to its geometry. ``score``
    rides along as a raw annotation field, which coconut-crab's label templating
    resolves case-insensitively — that is why ``{score}`` works with no upstream
    change. ``iscrowd`` is 0 and ``segmentation`` is omitted: v1 is boxes-only.
    """
    height, width = artifact.canvas.shape[:2]
    category_ids = {name: index for index, name in enumerate(artifact.categories, start=1)}
    annotations = []
    for index, box in enumerate(artifact.boxes, start=1):
        x1, y1, x2, y2 = box.bbox_xyxy
        annotations.append(
            {
                "id": index,
                "image_id": 1,
                "category_id": category_ids[box.category],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
                "score": box.score,
            }
        )
    return {
        "images": [
            {
                "id": 1,
                "file_name": f"{artifact.name}{IMAGE_SUFFIX}",
                "width": width,
                "height": height,
            }
        ],
        "categories": [
            {"id": index, "name": name} for name, index in category_ids.items()
        ],
        "annotations": annotations,
    }


# --- the coconut-crab boundary ----------------------------------------------


def _load_backend() -> tuple[Any, Any, Any]:
    """Import coconut-crab at the point of use (D7).

    Deferred so ``prepare``, ``train``, and ``infer``-without-rendering keep
    working in an environment that lacks it, and so a missing install surfaces
    as a configuration refusal naming the extra rather than an ``ImportError``
    at startup.
    """
    try:
        from PIL import Image
        from coconut_crab.render import config as cc_config
        from coconut_crab.render import engine as cc_engine
    except ImportError as error:  # pragma: no cover - exercised via monkeypatch
        raise RenderError(_MISSING_EXTRA) from error
    return cc_config, cc_engine, Image


def _render_config(cc_config: Any, style_config: Path | None, coco_path: Path) -> Any:
    """Compose coconut-crab's config, overriding only the structural keys (D5).

    Everything else — style, label template, encoding — comes from the user's
    TOML or coconut-crab's own defaults. Its ``ConfigError`` is deliberately not
    caught: a bad style key is coconut-crab's to reject, not SATI's to
    re-validate.
    """
    values = dict(cc_config.DEFAULTS)
    if style_config is not None:
        values.update(cc_config.load_render_toml(style_config))
    # `image_root` and `output` are nominal: the canvas is handed over in memory
    # and the output path is chosen per artifact by the caller. They are still
    # supplied because `render_config_from_values` requires them.
    values["coco_file"] = coco_path
    values["image_root"] = coco_path.parent
    values["output"] = "{FILE_STEM}" + IMAGE_SUFFIX
    return cc_config.render_config_from_values(values)


def render_artifact(artifact: Artifact, out_path: Path, style_config: Path | None) -> Path:
    """Draw one artifact and write it.

    The canvas is passed to coconut-crab **in memory** via
    ``render_annotations_on_image``, so a 308 MP full-res frame is not written
    out and read back just to be re-encoded. Only the COCO document round-trips
    through a temporary file, because ``load_coco`` takes a path.
    """
    cc_config, cc_engine, Image = _load_backend()
    with tempfile.TemporaryDirectory(prefix="sati-render-") as scratch:
        coco_path = Path(scratch) / f"{artifact.name}.json"
        coco_path.write_text(json.dumps(coco_document(artifact)))
        config = _render_config(cc_config, style_config, coco_path)
        dataset = cc_engine.load_coco(coco_path)
        (image,) = dataset.images
        # Pixels are RGB uint8 HWC out of `io.read_image` and stay that way
        # through `build_level_pyramid`, so no channel swap is needed.
        base = Image.fromarray(artifact.canvas).convert("RGBA")
        cc_engine.render_annotations_on_image(base, dataset, image, config)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cc_engine.save_rendered_image(base, out_path)
    return out_path


# --- entry point ------------------------------------------------------------


def artifacts_enabled(cfg: RootConfig, command: str = "infer") -> bool:
    """Both gates: the global switch and the per-command output leaf (D8)."""
    if not cfg.rendering.enabled:
        return False
    leaf = getattr(cfg.output_for(command), "artifacts", None)
    return bool(leaf is not None and leaf.enabled and leaf.dir is not None)


def render_frame(
    inference: "FrameInference",
    pixels: np.ndarray,
    *,
    cfg: RootConfig,
    class_names: dict[int, str],
    artifacts_dir: Path,
    pyramid: LevelPyramid | None = None,
) -> list[Path]:
    """Write one frame's overlay tree; return the files written, in order.

    ``pyramid`` is accepted so ``infer`` can hand over the stack it already
    built rather than resizing every level a second time.
    """
    if not cfg.rendering.enabled:
        return []
    if pyramid is None:
        pyramid = build_level_pyramid(pixels, inference.plan.levels)
    destination = artifacts_dir / inference.fullframe_id / FULLFRAME_SUBDIR
    style_config = cfg.rendering.style_config
    return [
        render_artifact(
            artifact, destination / f"{artifact.name}{IMAGE_SUFFIX}", style_config
        )
        for artifact in plan_artifacts(
            inference, pyramid, cfg=cfg, class_names=class_names
        )
    ]
