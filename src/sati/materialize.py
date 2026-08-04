"""Slice/label emission to disk + generated ``data.yaml`` (v1 §4.8, §6.1).

The pixel source sits behind a narrow boundary (settled 2026-07-03):
:func:`build_level_pyramid` produces a :class:`LevelPyramid` (index-keyed level
arrays + the full-res frame) and :func:`emit_slice` cuts one emitted slice from
it, applying the sharp-source rule for stretched windows (§6.1). Everything
else consumes those two functions, so v2 can swap the in-memory pyramid for a
zarr-backed one without touching emission, labels, or manifests.

Frames are processed one at a time — one frame plus its level stack in memory,
never whole-dataset buffering (§5 memory note).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from sati.config_schemas import RootConfig, SlicingConfig
from sati.dataset_identity import dataset_identity_sources, load_source_dataset
from sati.hashing import (
    class_mapping_hash,
    class_mapping_hash_full_source,
    preprocessing_hash,
    preprocessing_hash_full_source,
    target_schema_hash,
    target_schema_hash_full_source,
)
from sati.io import IngestedAnnotation, IngestedDataset, IngestedFrame, IngestError, read_image
from sati.manifests import (
    ANNOTATIONS_SCHEMA,
    FULLFRAMES_SCHEMA,
    SCHEMA_VERSION,
    SLICE_ANNOTATIONS_SCHEMA,
    SLICES_SCHEMA,
    HashWithSource,
    ManifestMetadata,
    build_table,
    write_manifest,
    write_metadata_sidecar,
)
from sati.tiling import FramePlan, Level, SliceSpec, plan_frame, project_box

TRAINED_SPLITS = ("training", "validation")  # holdout materializes but stays out of data.yaml


# --- the pixel-source boundary ----------------------------------------------


@dataclass(frozen=True)
class LevelPyramid:
    """The pixel source behind the seam: per-level arrays keyed by level index,
    plus the full-res frame (which may not itself be a level, e.g. under
    ``start_then_geometric``)."""

    fullres: np.ndarray
    levels: dict[int, "Level"]  # level index → Level
    pixels: dict[int, np.ndarray]  # level index → array


def build_level_pyramid(frame: np.ndarray, levels: tuple[Level, ...]) -> LevelPyramid:
    """Per-level pixel arrays, each resized from full-res directly.

    A level whose realized dims equal the source frame reuses the original
    array. Other levels, including a configured half-scale first level, are
    resized from full-res directly (never cascaded) with ``INTER_AREA`` (§6.1).
    """
    frame_h, frame_w = frame.shape[:2]
    pixels: dict[int, np.ndarray] = {}
    for level in levels:
        if (level.width, level.height) == (frame_w, frame_h):
            pixels[level.index] = frame
        else:
            pixels[level.index] = cv2.resize(
                frame, (level.width, level.height), interpolation=cv2.INTER_AREA
            )
    return LevelPyramid(
        fullres=frame, levels={level.index: level for level in levels}, pixels=pixels
    )


def _stretch_source(spec: SliceSpec, pyramid: LevelPyramid) -> int | None:
    """Sharp-source rule (settled 2026-07-03): a stretched window is emitted
    from the smallest level whose mapped window is ≥ emitted size on both axes,
    so stretching never upscales already-downsampled pixels. Returns that
    level's index, or ``None`` meaning "use the full-res frame".
    """
    own = pyramid.levels[spec.level]
    ew, eh = spec.emitted_wh
    best: tuple[int, int] | None = None  # (area proxy: width, index)
    for level in sorted(pyramid.levels.values(), key=lambda lv: lv.width * lv.height):
        mapped_w = spec.slice_wh[0] * level.width / own.width
        mapped_h = spec.slice_wh[1] * level.height / own.height
        if mapped_w >= ew and mapped_h >= eh:
            best = (level.width, level.index)
            break
    if best is not None:
        return best[1]
    return None  # fall back to full-res, the sharpest thing we have


def emit_slice(spec: SliceSpec, pyramid: LevelPyramid) -> np.ndarray:
    """Cut one emitted slice.

    Unstretched slices crop their own level directly. Stretched slices source
    their window from the finest sufficient level (or full-res) and resize
    *down* to emitted size — never up from a coarser level — so whole-level and
    edge-stretched slices stop looking blurry.
    """
    x, y = spec.slice_xy
    w, h = spec.slice_wh
    if not spec.stretched_axes and (w, h) == spec.emitted_wh:
        return pyramid.pixels[spec.level][y : y + h, x : x + w]

    own = pyramid.levels[spec.level]
    source_index = _stretch_source(spec, pyramid)
    if source_index is not None:
        source = pyramid.pixels[source_index]
        src_level = pyramid.levels[source_index]
        sx, sy = src_level.width / own.width, src_level.height / own.height
    else:
        source = pyramid.fullres
        src_h, src_w = source.shape[:2]
        sx, sy = src_w / own.width, src_h / own.height

    x0 = round(x * sx)
    y0 = round(y * sy)
    x1 = min(source.shape[1], round((x + w) * sx))
    y1 = min(source.shape[0], round((y + h) * sy))
    crop = source[y0:y1, x0:x1]
    shrinking = crop.shape[1] >= spec.emitted_wh[0] and crop.shape[0] >= spec.emitted_wh[1]
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(crop, spec.emitted_wh, interpolation=interp)


# --- emission ---------------------------------------------------------------


def slice_stem(fullframe_id: str, spec: SliceSpec) -> str:
    return f"{fullframe_id}_L{spec.level}_S{spec.slice_id}"


@dataclass
class _FrameOutput:
    slice_rows: list[dict]
    slice_ann_rows: list[dict]


def _yolo_label_line(class_id: int, bbox_slice: tuple, emitted_wh: tuple[int, int]) -> str:
    x1, y1, x2, y2 = bbox_slice
    ew, eh = emitted_wh
    cx, cy = (x1 + x2) / 2 / ew, (y1 + y2) / 2 / eh
    w, h = (x2 - x1) / ew, (y2 - y1) / eh
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def materialize_frame(
    frame: IngestedFrame,
    annotations: list[IngestedAnnotation],
    cfg: RootConfig,
    dataset_root: Path,
    *,
    slice_ann_id_start: int = 0,
) -> _FrameOutput:
    """Slice one frame to disk and return its manifest rows.

    Writes ``yolo/images/<split>/<stem>.<ext>`` and a matching label ``.txt``
    for every slice — zero-target slices get empty label files (§4.8). Only
    non-dropped projections become label lines; every projection becomes a
    ``slice_annotations`` row with its drop flags recorded.
    """
    pixels = read_image(frame.source_uri)
    actual_hw = pixels.shape[:2]
    if actual_hw != (frame.height, frame.width):
        raise IngestError(
            f"frame '{frame.fullframe_id}': manifest says {frame.width}x{frame.height}, "
            f"decoded image is {actual_hw[1]}x{actual_hw[0]}"
        )

    fullres_wh = (frame.width, frame.height)
    plan: FramePlan = plan_frame(fullres_wh, cfg.slicing)
    pyramid = build_level_pyramid(pixels, plan.levels)

    ext = cfg.slicing.image_format
    images_dir = dataset_root / "yolo" / "images" / frame.dataset_split
    labels_dir = dataset_root / "yolo" / "labels" / frame.dataset_split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    slice_rows: list[dict] = []
    slice_ann_rows: list[dict] = []
    slice_ann_id = slice_ann_id_start
    for spec in plan.slices:
        level = plan.level(spec.level)
        emitted = emit_slice(spec, pyramid)
        stem = slice_stem(frame.fullframe_id, spec)
        image_path = images_dir / f"{stem}.{ext}"
        if not cv2.imwrite(str(image_path), cv2.cvtColor(emitted, cv2.COLOR_RGB2BGR)):
            raise IngestError(f"failed to write slice image: {image_path}")

        label_lines: list[str] = []
        for ann in annotations:
            proj = project_box(ann.bbox_xyxy, spec, level, fullres_wh, cfg.slicing)
            if proj is None:
                continue
            slice_ann_rows.append(
                {
                    "slice_ann_id": slice_ann_id,
                    "parent_ann_id": ann.ann_id,
                    "fullframe_id": frame.fullframe_id,
                    "level": spec.level,
                    "slice_id": spec.slice_id,
                    "bbox_fullres_xyxy": list(ann.bbox_xyxy),
                    "bbox_level_xyxy": list(proj.bbox_level_xyxy),
                    "bbox_slice_xyxy": list(proj.bbox_slice_xyxy),
                    "visible_area_frac": proj.visible_area_frac,
                    "truncated_edges": list(proj.truncated_edges),
                    "dropped_by_size_gate": proj.dropped_by_size_gate,
                    "dropped_by_min_area_ratio": proj.dropped_by_min_area_ratio,
                }
            )
            slice_ann_id += 1
            if proj.is_training_target:
                label_lines.append(
                    _yolo_label_line(ann.class_id, proj.bbox_slice_xyxy, spec.emitted_wh)
                )
        (labels_dir / f"{stem}.txt").write_text(
            "\n".join(label_lines) + ("\n" if label_lines else "")
        )

        slice_rows.append(
            {
                "fullframe_id": frame.fullframe_id,
                "level": spec.level,
                "slice_id": spec.slice_id,
                "dataset_split": frame.dataset_split,
                "scale": level.scale_requested,
                "level_wh": list(level.wh),
                "slice_xy": list(spec.slice_xy),
                "slice_wh": list(spec.slice_wh),
                "emitted_wh": list(spec.emitted_wh),
                "is_whole_level": spec.is_whole_level,
                "stretched_axes": list(spec.stretched_axes),
                "slice_uri": str(image_path.relative_to(dataset_root)),
                "affine_slice_to_level": spec.affine_slice_to_level.as_list(),
                "affine_slice_to_fullres": spec.affine_slice_to_fullres.as_list(),
            }
        )
    return _FrameOutput(slice_rows=slice_rows, slice_ann_rows=slice_ann_rows)


def write_data_yaml(
    dataset_root: Path,
    class_map: dict[int, str],
    slicing: SlicingConfig,
    *,
    dataset_hash_value: str | None = None,
    prepared_dataset_hash_value: str | None = None,
) -> Path:
    """The generated ``data.yaml`` ultralytics consumes verbatim; embeds the
    dataset-level ``preprocessing_hash`` so ``train`` can refuse on mismatch (§14)."""
    yolo_root = dataset_root / "yolo"
    yolo_root.mkdir(parents=True, exist_ok=True)
    spec = {
        "path": str(yolo_root),
        "train": "images/training",
        "val": "images/validation",
        "names": {int(k): v for k, v in class_map.items()},
        "sati_dataset_hash": dataset_hash_value,
        "sati_prepared_dataset_hash": prepared_dataset_hash_value,
        "sati_preprocessing_hash": preprocessing_hash(slicing),
    }
    path = yolo_root / "data.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


def materialize_dataset(
    dataset: IngestedDataset, cfg: RootConfig, dataset_root: Path | str
) -> Path:
    """The work of ``prepare`` (§5): slice every frame, write all four manifests,
    the ``_metadata.json`` sidecar, and the YOLO training tree. Returns the root."""
    dataset_root = Path(dataset_root)
    anns_by_frame: dict[str, list[IngestedAnnotation]] = {}
    for ann in dataset.annotations:
        anns_by_frame.setdefault(ann.fullframe_id, []).append(ann)

    fullframe_rows = [
        {
            "fullframe_id": f.fullframe_id,
            "source_uri": f.source_uri,
            "dataset_split": f.dataset_split,
            "fullres_wh": [f.width, f.height],
            "color_mode": f.color_mode,
        }
        for f in dataset.frames
    ]
    annotation_rows = []
    for ann in dataset.annotations:
        frame = next(f for f in dataset.frames if f.fullframe_id == ann.fullframe_id)
        x1, y1, x2, y2 = ann.bbox_xyxy
        w, h = x2 - x1, y2 - y1
        annotation_rows.append(
            {
                "fullframe_id": ann.fullframe_id,
                "ann_id": ann.ann_id,
                "class_id": ann.class_id,
                "class_name": ann.class_name,
                "bbox_xyxy": list(ann.bbox_xyxy),
                "bbox_w": w,
                "bbox_h": h,
                "bbox_rel_cxcywh": [
                    (x1 + x2) / 2 / frame.width,
                    (y1 + y2) / 2 / frame.height,
                    w / frame.width,
                    h / frame.height,
                ],
            }
        )

    slice_rows: list[dict] = []
    slice_ann_rows: list[dict] = []
    for frame in dataset.frames:  # sequential: one frame + stack in memory at a time
        output = materialize_frame(
            frame,
            anns_by_frame.get(frame.fullframe_id, []),
            cfg,
            dataset_root,
            slice_ann_id_start=len(slice_ann_rows),
        )
        slice_rows.extend(output.slice_rows)
        slice_ann_rows.extend(output.slice_ann_rows)

    manifests_dir = dataset_root / "manifests"
    write_manifest(build_table(FULLFRAMES_SCHEMA, fullframe_rows), manifests_dir / "fullframes.parquet")
    write_manifest(build_table(ANNOTATIONS_SCHEMA, annotation_rows), manifests_dir / "annotations.parquet")
    write_manifest(build_table(SLICES_SCHEMA, slice_rows), manifests_dir / "slices.parquet")
    write_manifest(
        build_table(SLICE_ANNOTATIONS_SCHEMA, slice_ann_rows),
        manifests_dir / "slice_annotations.parquet",
    )
    (
        logical_hash,
        logical_source,
        prepared_hash,
        prepared_source,
        prepared_id,
    ) = dataset_identity_sources(dataset, cfg)
    cfg.data.prepared.dataset_hash = logical_hash
    cfg.data.prepared.prepared_dataset_hash = prepared_hash
    cfg.data.prepared.prepared_dataset_id = prepared_id
    write_data_yaml(
        dataset_root,
        dataset.class_map,
        cfg.slicing,
        dataset_hash_value=logical_hash,
        prepared_dataset_hash_value=prepared_hash,
    )
    metadata = ManifestMetadata(
        schema_version=SCHEMA_VERSION,
        class_mapping={str(k): v for k, v in dataset.class_map.items()},
        hashes={
            "class_mapping_hash": HashWithSource(
                value=class_mapping_hash(dataset.class_map),
                source=class_mapping_hash_full_source(dataset.class_map),
            ),
            "dataset_hash": HashWithSource(
                value=logical_hash,
                source=logical_source,
            ),
            "prepared_dataset_hash": HashWithSource(
                value=prepared_hash,
                source=prepared_source,
            ),
            "target_schema_hash": HashWithSource(
                value=target_schema_hash(cfg),
                source=target_schema_hash_full_source(cfg),
            ),
            "preprocessing_hash": HashWithSource(
                value=preprocessing_hash(cfg.slicing),
                source=preprocessing_hash_full_source(cfg.slicing),
            ),
        },
    )
    write_metadata_sidecar(metadata, manifests_dir)
    return dataset_root


def execute_prepare(cfg: RootConfig) -> tuple[Path, dict[int, int]]:
    """``sati prepare`` (§5): ingest → resolve splits → materialize into the run
    directory. Returns ``(dataset_root, per-level slice counts)`` — the counts
    are the §8 scale-starvation diagnostic."""
    from sati.manifests import read_manifest

    dataset = load_source_dataset(cfg)
    if cfg.prepare_outputs.dir is None:
        raise IngestError("prepare_outputs.dir was not resolved")
    root = materialize_dataset(dataset, cfg, cfg.prepare_outputs.dir)
    slices = read_manifest(root / "manifests" / "slices.parquet", SLICES_SCHEMA)
    per_level: dict[int, int] = {}
    for level in slices.column("level").to_pylist():
        per_level[level] = per_level.get(level, 0) + 1
    return root, dict(sorted(per_level.items()))
