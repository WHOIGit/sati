"""Read-only source preflight and logical/prepared dataset identities."""

from __future__ import annotations

from typing import Any

from sati.config_schemas import RootConfig
from sati.hashing import (
    dataset_hash,
    dataset_hash_full_source,
    prepared_dataset_hash,
    prepared_dataset_hash_full_source,
    prepared_dataset_id,
)
from sati.io import IngestedDataset, apply_split_override, ingest_coco, ingest_yolo, load_split_toml


def load_source_dataset(cfg: RootConfig) -> IngestedDataset:
    source = cfg.data.source
    if source.format == "coco":
        dataset = ingest_coco(source.uri, image_root=source.image_root)
    else:
        dataset = ingest_yolo(source.uri)
    if source.split_file is None:
        return dataset
    frames = apply_split_override(dataset.frames, load_split_toml(source.split_file))
    kept = {frame.fullframe_id for frame in frames}
    return IngestedDataset(
        frames=frames,
        annotations=tuple(
            annotation
            for annotation in dataset.annotations
            if annotation.fullframe_id in kept
        ),
        class_map=dataset.class_map,
    )


def logical_rows(dataset: IngestedDataset) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fullframes = [
        {
            "fullframe_id": frame.fullframe_id,
            "source_uri": frame.source_uri,
            "dataset_split": frame.dataset_split,
            "fullres_wh": [frame.width, frame.height],
            "color_mode": frame.color_mode,
        }
        for frame in dataset.frames
    ]
    annotations = [
        {
            "fullframe_id": annotation.fullframe_id,
            "ann_id": annotation.ann_id,
            "class_id": annotation.class_id,
            "bbox_xyxy": list(annotation.bbox_xyxy),
        }
        for annotation in dataset.annotations
    ]
    return fullframes, annotations


def dataset_identity_sources(
    dataset: IngestedDataset,
    cfg: RootConfig,
) -> tuple[str, dict[str, Any], str, dict[str, Any], str]:
    fullframes, annotations = logical_rows(dataset)
    logical_source = dataset_hash_full_source(
        fullframes,
        annotations,
        dataset.class_map,
    )
    logical_hash = dataset_hash(fullframes, annotations, dataset.class_map)
    prepared_source = prepared_dataset_hash_full_source(logical_hash, cfg.slicing)
    prepared_hash = prepared_dataset_hash(logical_hash, cfg.slicing)
    return (
        logical_hash,
        logical_source,
        prepared_hash,
        prepared_source,
        prepared_dataset_id(prepared_hash),
    )


def populate_source_identities(cfg: RootConfig) -> IngestedDataset:
    dataset = load_source_dataset(cfg)
    logical_hash, _, prepared_hash, _, prepared_id = dataset_identity_sources(dataset, cfg)
    cfg.data.prepared.dataset_hash = logical_hash
    cfg.data.prepared.prepared_dataset_hash = prepared_hash
    cfg.data.prepared.prepared_dataset_id = prepared_id
    return dataset
