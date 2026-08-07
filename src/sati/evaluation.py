"""``sati eval`` (v1 §11): infer over a prepared split + scoring.

Metrics are computed on **final_merge predictions vs full-frame full-res
ground truth** — never per-slice. pycocotools mAP/PR overall and per-class,
written alongside ``validation_results.parquet``. The per-level breakdown is a
*diagnostic* (counts + confidence distribution per contributing level, GT
recall binned by long side), not a per-level mAP — that attribution problem is
deferred to v2.

v1 evaluates the ``validation`` split, fixed (no eval split knob in §3.3;
absent = unsupported).
"""

from __future__ import annotations

import contextlib
import io as _io
import json
from pathlib import Path

import numpy as np

from sati.backends import Backend, UltralyticsBackend
from sati.config_schemas import RootConfig
from sati.infer import infer_frame, rows_for_frame, run_hashes_for, write_results_sidecar
from sati.io import IngestError, read_image
from sati.manifests import (
    ANNOTATIONS_SCHEMA,
    FULLFRAMES_SCHEMA,
    VALIDATION_RESULTS_SCHEMA,
    HashWithSource,
    build_table,
    read_manifest,
    read_metadata_sidecar,
    write_manifest,
)
from sati.tiling import box_iou

EVAL_SPLIT = "validation"

# GT long-side bins (px) for the recall diagnostic — log2 ladder, SATI's scale axis
LONGSIDE_BIN_EDGES = (0, 16, 32, 64, 128, 256, 512, 1024, float("inf"))


def _longside_bin(box: list[float]) -> str:
    long_side = max(box[2] - box[0], box[3] - box[1])
    for lo, hi in zip(LONGSIDE_BIN_EDGES, LONGSIDE_BIN_EDGES[1:]):
        if lo <= long_side < hi:
            return f"{lo}-{'inf' if hi == float('inf') else int(hi)}"
    raise AssertionError("unreachable")


def coco_map_scores(
    gt_by_frame: dict[str, list[dict]],
    dt_by_frame: dict[str, list[dict]],
    class_map: dict[int, str],
) -> dict:
    """pycocotools mAP on in-memory GT/DT. GT/DT boxes arrive as xyxy full-res."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    frame_ids = {fid: i + 1 for i, fid in enumerate(sorted(gt_by_frame))}
    images = [{"id": i} for i in frame_ids.values()]
    categories = [{"id": cid, "name": name} for cid, name in class_map.items()]
    gt_anns = []
    for fid, anns in gt_by_frame.items():
        for ann in anns:
            x1, y1, x2, y2 = ann["bbox_xyxy"]
            gt_anns.append(
                {
                    "id": len(gt_anns) + 1,
                    "image_id": frame_ids[fid],
                    "category_id": ann["class_id"],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
    detections = []
    for fid, dets in dt_by_frame.items():
        for det in dets:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            detections.append(
                {
                    "image_id": frame_ids[fid],
                    "category_id": det["class_id"],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": det["score"],
                }
            )

    empty = {
        "mAP50_95": 0.0,
        "mAP50": 0.0,
        "mAP75": 0.0,
        "per_class_ap50_95": {name: 0.0 for name in class_map.values()},
    }
    if not gt_anns or not detections:
        return empty

    with contextlib.redirect_stdout(_io.StringIO()):  # pycocotools prints unconditionally
        gt = COCO()
        gt.dataset = {"images": images, "annotations": gt_anns, "categories": categories}
        gt.createIndex()
        dt = gt.loadRes(detections)
        ev = COCOeval(gt, dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    precision = ev.eval["precision"]  # [iou_thrs, recall, class, area, max_dets]
    per_class = {}
    for k, (cid, name) in enumerate(sorted(class_map.items())):
        p = precision[:, :, k, 0, -1]
        per_class[name] = float(p[p > -1].mean()) if (p > -1).any() else 0.0
    return {
        "mAP50_95": float(ev.stats[0]),
        "mAP50": float(ev.stats[1]),
        "mAP75": float(ev.stats[2]),
        "per_class_ap50_95": per_class,
    }


def per_level_diagnostic(final_rows: list[dict]) -> dict[str, dict]:
    """§11: per level, count + confidence distribution of final_merge rows whose
    source_levels include it.

    A detection is counted under *every* contributing level, so these counts sum
    to more than the detection count whenever clusters span levels. That is the
    intended semantic, not double counting — but note it gets more pronounced
    under class-agnostic matching at intersection-over-smaller, which forms
    larger clusters than class-aware IoU did.
    """
    by_level: dict[int, list[float]] = {}
    for row in final_rows:
        for level in row["source_levels"]:
            by_level.setdefault(level, []).append(row["score"])
    return {
        str(level): {
            "final_merge_count": len(scores),
            "confidence_mean": float(np.mean(scores)),
            "confidence_p50": float(np.median(scores)),
        }
        for level, scores in sorted(by_level.items())
    }


def gt_recall_by_size(
    gt_by_frame: dict[str, list[dict]],
    dt_by_frame: dict[str, list[dict]],
    iou_threshold: float = 0.5,
) -> dict[str, dict]:
    """§11: recall of GT boxes binned by long side (greedy class-aware IoU match)."""
    bins: dict[str, dict[str, int]] = {}
    for fid, gt_anns in gt_by_frame.items():
        dets = sorted(dt_by_frame.get(fid, []), key=lambda d: -d["score"])
        matched_gt: set[int] = set()
        for det in dets:
            best, best_iou = None, iou_threshold
            for gi, ann in enumerate(gt_anns):
                if gi in matched_gt or ann["class_id"] != det["class_id"]:
                    continue
                iou = box_iou(tuple(det["bbox_xyxy"]), tuple(ann["bbox_xyxy"]))
                if iou >= best_iou:
                    best, best_iou = gi, iou
            if best is not None:
                matched_gt.add(best)
        for gi, ann in enumerate(gt_anns):
            key = _longside_bin(ann["bbox_xyxy"])
            entry = bins.setdefault(key, {"n_gt": 0, "n_matched": 0})
            entry["n_gt"] += 1
            entry["n_matched"] += gi in matched_gt
    return {
        key: {
            "n_gt": entry["n_gt"],
            "recall50": entry["n_matched"] / entry["n_gt"],
        }
        for key, entry in sorted(bins.items(), key=lambda kv: float(kv[0].split("-")[0]))
    }


def execute_eval(
    cfg: RootConfig,
    *,
    checkpoint: Path,
    backend: Backend | None = None,
) -> tuple[Path, dict]:
    """Infer over the prepared validation split + score."""
    root = cfg.data.prepared.uri
    if root is None:
        raise IngestError("data.prepared.uri was not resolved")
    manifests_dir = root / "manifests"
    if not manifests_dir.is_dir():
        raise IngestError(f"no prepared dataset at {root} (run `sati prepare` first)")

    fullframes = read_manifest(manifests_dir / "fullframes.parquet", FULLFRAMES_SCHEMA)
    annotations = read_manifest(manifests_dir / "annotations.parquet", ANNOTATIONS_SCHEMA)
    prepared_meta = read_metadata_sidecar(manifests_dir)
    class_mapping = {int(k): v for k, v in prepared_meta.class_mapping.items()}

    frames = [r for r in fullframes.to_pylist() if r["dataset_split"] == EVAL_SPLIT]
    if not frames:
        raise IngestError(f"prepared dataset has no '{EVAL_SPLIT}' frames")

    gt_by_frame: dict[str, list[dict]] = {r["fullframe_id"]: [] for r in frames}
    for ann in annotations.to_pylist():
        if ann["fullframe_id"] in gt_by_frame:
            gt_by_frame[ann["fullframe_id"]].append(
                {"class_id": ann["class_id"], "bbox_xyxy": ann["bbox_xyxy"]}
            )

    backend = backend or UltralyticsBackend()
    backend.load(checkpoint)
    hashes = run_hashes_for(
        cfg,
        checkpoint=checkpoint,
        dataset_hash_value=prepared_meta.hashes["dataset_hash"].value,
        prepared_dataset_hash_value=prepared_meta.hashes["prepared_dataset_hash"].value,
        class_mapping=class_mapping,
    )

    rows: list[dict] = []
    dt_by_frame: dict[str, list[dict]] = {}
    for frame in frames:  # sequential: one frame + stack in memory
        pixels = read_image(frame["source_uri"])
        inference = infer_frame(frame["fullframe_id"], pixels, cfg, backend)
        rows.extend(
            rows_for_frame(
                inference,
                cfg=cfg,
                hashes=hashes,
                dataset_split=EVAL_SPLIT,
                source_uri=frame["source_uri"],
            )
        )
        dt_by_frame[frame["fullframe_id"]] = [
            {"bbox_xyxy": list(d.bbox_fullres_xyxy), "score": d.score, "class_id": d.class_id}
            for d in inference.final_detections
        ]

    if cfg.eval_outputs.results.dir is None or cfg.eval_outputs.metrics.dir is None:
        raise IngestError("eval output directories were not resolved")
    results_path = cfg.eval_outputs.results.dir / "validation_results.parquet"
    write_manifest(build_table(VALIDATION_RESULTS_SCHEMA, rows), results_path)
    write_results_sidecar(
        cfg,
        class_mapping=prepared_meta.class_mapping,
        checkpoint=checkpoint,
        directory=cfg.eval_outputs.results.dir,
        extra_hashes={
            "dataset_hash": HashWithSource(
                value=prepared_meta.hashes["dataset_hash"].value,
                source=prepared_meta.hashes["dataset_hash"].source,
            ),
            "prepared_dataset_hash": HashWithSource(
                value=prepared_meta.hashes["prepared_dataset_hash"].value,
                source=prepared_meta.hashes["prepared_dataset_hash"].source,
            ),
        },
    )

    final_rows = [r for r in rows if r["record_type"] == "final_merge"]
    metrics = {
        **coco_map_scores(gt_by_frame, dt_by_frame, class_mapping),
        "per_level": per_level_diagnostic(final_rows),
        "gt_recall_by_longside": gt_recall_by_size(gt_by_frame, dt_by_frame),
        "n_frames": len(frames),
        "n_final_merge": len(final_rows),
    }
    cfg.eval_outputs.metrics.dir.mkdir(parents=True, exist_ok=True)
    (cfg.eval_outputs.metrics.dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=False)
    )
    return results_path, metrics
