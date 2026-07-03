"""``sati convert`` (v1 §5): COCO/YOLO ↔ canonical manifests, both directions.

A standalone utility, not a pipeline stage: no slicing, no level building, no
YOLO *training* tree. Use ``prepare`` to start a run; ``convert`` is for
interchange.

Manifests → COCO writes one ``<split>.json`` per split present (mirroring what
``ingest_coco`` accepts as a directory). Manifests → YOLO writes labels from
the stored ``bbox_rel_cxcywh`` and symlinks images rather than copying.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from sati.config_schemas import RootConfig
from sati.hashing import (
    class_map_hash,
    class_map_hash_full_source,
    dataset_hash,
    dataset_hash_full_source,
)
from sati.io import IngestedDataset, IngestError, ingest_coco, ingest_yolo
from sati.manifests import (
    ANNOTATIONS_SCHEMA,
    FULLFRAMES_SCHEMA,
    SCHEMA_VERSION,
    HashWithSource,
    ManifestMetadata,
    build_table,
    read_manifest,
    read_metadata_sidecar,
    write_manifest,
    write_metadata_sidecar,
)

_SPLIT_TO_YOLO_KEY = {"training": "train", "validation": "val", "holdout": "test"}


def _find_manifests_dir(source: Path) -> Path:
    for candidate in (source, source / "manifests"):
        if (candidate / "fullframes.parquet").is_file():
            return candidate
    raise IngestError(f"no manifests at {source} (expected fullframes.parquet)")


def dataset_to_manifests(dataset: IngestedDataset, out_dir: Path) -> Path:
    """Write fullframes + annotations manifests + sidecar. No slice tables —
    convert never slices."""
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
    dims = {f.fullframe_id: (f.width, f.height) for f in dataset.frames}
    annotation_rows = []
    for ann in dataset.annotations:
        w, h = dims[ann.fullframe_id]
        x1, y1, x2, y2 = ann.bbox_xyxy
        annotation_rows.append(
            {
                "fullframe_id": ann.fullframe_id,
                "ann_id": ann.ann_id,
                "class_id": ann.class_id,
                "class_name": ann.class_name,
                "bbox_xyxy": list(ann.bbox_xyxy),
                "bbox_w": x2 - x1,
                "bbox_h": y2 - y1,
                "bbox_rel_cxcywh": [
                    (x1 + x2) / 2 / w,
                    (y1 + y2) / 2 / h,
                    (x2 - x1) / w,
                    (y2 - y1) / h,
                ],
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(build_table(FULLFRAMES_SCHEMA, fullframe_rows), out_dir / "fullframes.parquet")
    write_manifest(
        build_table(ANNOTATIONS_SCHEMA, annotation_rows), out_dir / "annotations.parquet"
    )
    files = sorted(
        (p.name, p.stat().st_size)
        for p in out_dir.glob("*.parquet")
    )
    write_metadata_sidecar(
        ManifestMetadata(
            schema_version=SCHEMA_VERSION,
            class_map={str(k): v for k, v in dataset.class_map.items()},
            hashes={
                "class_map_hash": HashWithSource(
                    value=class_map_hash(dataset.class_map),
                    source=class_map_hash_full_source(dataset.class_map),
                ),
                "dataset_hash": HashWithSource(
                    value=dataset_hash(files), source=dataset_hash_full_source(files)
                ),
            },
        ),
        out_dir,
    )
    return out_dir


def manifests_to_coco(manifests_dir: Path, out_dir: Path) -> list[Path]:
    """One ``<split>.json`` per split present; boxes back to COCO xywh."""
    fullframes = read_manifest(manifests_dir / "fullframes.parquet", FULLFRAMES_SCHEMA)
    annotations = read_manifest(manifests_dir / "annotations.parquet", ANNOTATIONS_SCHEMA)
    class_map = read_metadata_sidecar(manifests_dir).class_map
    categories = [{"id": int(k), "name": v} for k, v in class_map.items()]

    frames_by_split: dict[str, list[dict]] = {}
    for frame in fullframes.to_pylist():
        frames_by_split.setdefault(frame["dataset_split"], []).append(frame)
    anns_by_frame: dict[str, list[dict]] = {}
    for ann in annotations.to_pylist():
        anns_by_frame.setdefault(ann["fullframe_id"], []).append(ann)

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for split, frames in sorted(frames_by_split.items()):
        images, coco_anns = [], []
        for frame in frames:
            images.append(
                {
                    "id": frame["fullframe_id"],
                    "file_name": frame["source_uri"],
                    "width": frame["fullres_wh"][0],
                    "height": frame["fullres_wh"][1],
                }
            )
            for ann in anns_by_frame.get(frame["fullframe_id"], []):
                x1, y1, x2, y2 = ann["bbox_xyxy"]
                coco_anns.append(
                    {
                        "id": ann["ann_id"],
                        "image_id": ann["fullframe_id"],
                        "category_id": ann["class_id"],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1),
                        "iscrowd": 0,
                    }
                )
        path = out_dir / f"{split}.json"
        path.write_text(
            json.dumps(
                {"images": images, "annotations": coco_anns, "categories": categories}
            )
        )
        written.append(path)
    return written


def manifests_to_yolo(manifests_dir: Path, out_dir: Path) -> Path:
    """YOLO interchange tree: symlinked images, labels from ``bbox_rel_cxcywh``,
    ``data.yaml`` keyed by the splits present."""
    fullframes = read_manifest(manifests_dir / "fullframes.parquet", FULLFRAMES_SCHEMA)
    annotations = read_manifest(manifests_dir / "annotations.parquet", ANNOTATIONS_SCHEMA)
    class_map = read_metadata_sidecar(manifests_dir).class_map

    anns_by_frame: dict[str, list[dict]] = {}
    for ann in annotations.to_pylist():
        anns_by_frame.setdefault(ann["fullframe_id"], []).append(ann)

    splits_present: set[str] = set()
    for frame in fullframes.to_pylist():
        split = frame["dataset_split"]
        splits_present.add(split)
        images_dir = out_dir / "images" / split
        labels_dir = out_dir / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        source = Path(frame["source_uri"])
        link = images_dir / f"{frame['fullframe_id']}{source.suffix}"
        if not link.exists():
            link.symlink_to(source)
        lines = [
            f"{ann['class_id']} " + " ".join(f"{v:.6f}" for v in ann["bbox_rel_cxcywh"])
            for ann in anns_by_frame.get(frame["fullframe_id"], [])
        ]
        (labels_dir / f"{frame['fullframe_id']}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )

    spec: dict = {"path": str(out_dir)}
    for split in ("training", "validation", "holdout"):
        if split in splits_present:
            spec[_SPLIT_TO_YOLO_KEY[split]] = f"images/{split}"
    spec["names"] = {int(k): v for k, v in class_map.items()}
    (out_dir / "data.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    return out_dir


def execute_convert(cfg: RootConfig, *, to: str) -> Path:
    """Dispatch: ``to=manifests`` ingests ``data.*``; ``to=coco|yolo`` reads
    manifests at ``data.source_path``. Output lands in the run directory."""
    if to not in ("manifests", "coco", "yolo"):
        raise IngestError(f"unknown convert target '{to}' (manifests|coco|yolo)")
    run_dir = Path(cfg.output.run_dir)
    if to == "manifests":
        if cfg.data.source_format == "coco":
            dataset = ingest_coco(cfg.data.source_path, image_root=cfg.data.image_root)
        else:
            dataset = ingest_yolo(cfg.data.source_path)
        return dataset_to_manifests(dataset, run_dir / "manifests")
    manifests_dir = _find_manifests_dir(cfg.data.source_path)
    if to == "coco":
        manifests_to_coco(manifests_dir, run_dir / "coco")
        return run_dir / "coco"
    return manifests_to_yolo(manifests_dir, run_dir / "yolo_export")
