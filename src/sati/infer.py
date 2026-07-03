"""Sliced inference engine (v1 §9). ``eval`` is this plus scoring.

Per frame: build the level stack in memory → emit nominal×nominal slices (the
same `materialize` pixel seam, minus disk writes) → ``predict`` per slice →
recombine (§6.2) → ``validation_results`` rows: ``final_merge`` +
``frame_metadata`` + ``level_merge``, plus ``slice_prediction`` only when
``results.cache_slice_predictions`` is on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sati.backends import Backend, UltralyticsBackend
from sati.config_schemas import RootConfig
from sati.hashing import (
    checkpoint_hash,
    config_hash,
    config_hash_full_source,
    model_config_hash,
    model_config_hash_full_source,
    preprocessing_hash,
    preprocessing_hash_full_source,
    recombination_hash,
)
from sati.io import IngestError, read_image
from sati.manifests import (
    SCHEMA_VERSION,
    VALIDATION_RESULTS_SCHEMA,
    HashWithSource,
    ManifestMetadata,
    build_table,
    write_manifest,
    write_metadata_sidecar,
)
from sati.materialize import build_level_pyramid, emit_slice
from sati.tiling import (
    MERGE_STRATEGY_NAME,
    FinalDetection,
    FramePlan,
    LevelDetection,
    SlicePrediction,
    plan_frame,
    recombine,
)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class FrameInference:
    fullframe_id: str
    fullres_wh: tuple[int, int]
    plan: FramePlan
    slice_predictions: tuple[SlicePrediction, ...]
    level_detections: tuple[LevelDetection, ...]
    final_detections: tuple[FinalDetection, ...]


@dataclass(frozen=True)
class RunHashes:
    """The per-run constant hash family carried on every results row (§4.5)."""

    config_hash: str
    model_config_hash: str
    checkpoint_hash: str | None = None
    dataset_hash: str | None = None
    class_map_hash: str | None = None
    preprocessing_hash: str | None = None


def run_hashes_for(cfg: RootConfig) -> RunHashes:
    weights = cfg.model.weights
    return RunHashes(
        config_hash=config_hash(cfg),
        model_config_hash=model_config_hash(cfg.model),
        checkpoint_hash=(
            checkpoint_hash(weights) if weights and Path(weights).is_file() else None
        ),
        preprocessing_hash=preprocessing_hash(cfg.slicing),
    )


def infer_frame(
    fullframe_id: str,
    pixels: np.ndarray,
    cfg: RootConfig,
    backend: Backend,
) -> FrameInference:
    """One frame through slice → predict → recombine. Pixels stay in memory."""
    fullres_wh = (pixels.shape[1], pixels.shape[0])
    plan = plan_frame(fullres_wh, cfg.slicing)
    pyramid = build_level_pyramid(pixels, plan.levels)

    predictions: list[SlicePrediction] = []
    batch = max(1, cfg.training.batch)
    specs = list(plan.slices)
    for start in range(0, len(specs), batch):
        chunk = specs[start : start + batch]
        emitted = [emit_slice(pyramid[s.level], s) for s in chunk]
        per_slice = backend.predict(
            emitted,
            imgsz=cfg.model.imgsz,
            confidence_threshold=cfg.recombination.confidence_threshold,
        )
        for spec, detections in zip(chunk, per_slice):
            predictions.extend(
                SlicePrediction(
                    slice_id=spec.slice_id,
                    level=spec.level,
                    bbox_slice_xyxy=det.bbox_xyxy,
                    score=det.score,
                    class_id=det.class_id,
                )
                for det in detections
            )

    level_detections, final_detections = recombine(predictions, plan, cfg.recombination)
    return FrameInference(
        fullframe_id=fullframe_id,
        fullres_wh=fullres_wh,
        plan=plan,
        slice_predictions=tuple(predictions),
        level_detections=level_detections,
        final_detections=final_detections,
    )


def rows_for_frame(
    inference: FrameInference,
    *,
    cfg: RootConfig,
    hashes: RunHashes,
    dataset_split: str,
    source_uri: str,
) -> list[dict]:
    """``validation_results`` rows for one frame, ready for ``build_table``."""
    base = {
        "fullframe_id": inference.fullframe_id,
        "dataset_split": dataset_split,
        "run_id": cfg.run_id,
        "schema_version": SCHEMA_VERSION,
        "config_hash": hashes.config_hash,
        "model_config_hash": hashes.model_config_hash,
        "checkpoint_hash": hashes.checkpoint_hash,
        "dataset_hash": hashes.dataset_hash,
        "class_map_hash": hashes.class_map_hash,
        "preprocessing_hash": hashes.preprocessing_hash,
    }
    level_hash = recombination_hash(cfg.recombination, "level_merge")
    final_hash = recombination_hash(cfg.recombination, "final_merge")

    rows: list[dict] = [
        {
            **base,
            "record_type": "frame_metadata",
            "source_uri": source_uri,
            "fullres_wh": list(inference.fullres_wh),
        }
    ]
    if cfg.results.cache_slice_predictions:
        rows.extend(
            {
                **base,
                "record_type": "slice_prediction",
                "level": p.level,
                "slice_id": p.slice_id,
                "bbox_slice_xyxy": list(p.bbox_slice_xyxy),
                "score": p.score,
                "class_id": p.class_id,
            }
            for p in inference.slice_predictions
        )
    rows.extend(
        {
            **base,
            "record_type": "level_merge",
            "level": d.level,
            "bbox_level_xyxy": list(d.bbox_level_xyxy),
            "score": d.score,
            "class_id": d.class_id,
            "source_slice_ids": list(d.source_slice_ids),
            "merge_strategy_name": MERGE_STRATEGY_NAME,
            "recombination_hash": level_hash,
        }
        for d in inference.level_detections
    )
    rows.extend(
        {
            **base,
            "record_type": "final_merge",
            "bbox_fullres_xyxy": list(d.bbox_fullres_xyxy),
            "score": d.score,
            "class_id": d.class_id,
            "source_levels": list(d.source_levels),
            "merge_strategy_name": MERGE_STRATEGY_NAME,
            "recombination_hash": final_hash,
        }
        for d in inference.final_detections
    )
    return rows


def _discover_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        images = sorted(
            p for p in source.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES
        )
        if images:
            return images
    raise IngestError(f"no input imagery at {source}")


def execute_infer(cfg: RootConfig, backend: Backend | None = None) -> Path:
    """SAHI-like sliced inference over new imagery (§9): ``data.source_path`` is
    an image file or a directory of images; ``model.weights`` is the checkpoint.
    Writes ``validation_results.parquet`` into the run dir; returns its path."""
    if not cfg.model.weights:
        raise IngestError("infer requires model.weights=<checkpoint or hub asset>")
    backend = backend or UltralyticsBackend()
    backend.load(cfg.model.weights)
    hashes = run_hashes_for(cfg)

    rows: list[dict] = []
    for image_path in _discover_images(cfg.data.source_path):
        pixels = read_image(image_path)  # one frame + stack in memory at a time
        inference = infer_frame(image_path.stem, pixels, cfg, backend)
        rows.extend(
            rows_for_frame(
                inference,
                cfg=cfg,
                hashes=hashes,
                dataset_split="holdout",  # new imagery has no split; scored never
                source_uri=str(image_path.resolve()),
            )
        )

    out = Path(cfg.output.run_dir) / "validation_results.parquet"
    write_manifest(build_table(VALIDATION_RESULTS_SCHEMA, rows), out)
    write_results_sidecar(cfg, class_map={}, directory=out.parent)
    return out


def write_results_sidecar(
    cfg: RootConfig, *, class_map: dict[str, str], directory: Path
) -> Path:
    """§4.5 sidecar for a results table: class map + each hash paired with its
    verifiable source. Infer over new imagery has no class map (empty); eval
    over a prepared split passes the prepared manifest's."""
    return write_metadata_sidecar(
        ManifestMetadata(
            schema_version=SCHEMA_VERSION,
            class_map=class_map,
            hashes={
                "config_hash": HashWithSource(
                    value=config_hash(cfg), source=config_hash_full_source(cfg)
                ),
                "model_config_hash": HashWithSource(
                    value=model_config_hash(cfg.model),
                    source=model_config_hash_full_source(cfg.model),
                ),
                "preprocessing_hash": HashWithSource(
                    value=preprocessing_hash(cfg.slicing),
                    source=preprocessing_hash_full_source(cfg.slicing),
                ),
            },
        ),
        directory,
    )
