"""``sati inspect`` (v1 §5.1): read-only artifact introspection.

Reads manifests, sidecars, and file headers — never a model forward pass,
never a pixel decode, never a write to the inspected store. Each summary
function returns the dict that ``--json`` dumps; the CLI renders rich tables
from the same dict, so human and machine views can't drift. Hash previews and
re-verification share ``sati.hashing`` with the writers for the same reason.

``--json`` is a best-effort dump for now (§5.1 NOTE: versioned schema deferred
until the first automation consumer appears).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
import yaml
from rich.console import Console
from rich.table import Table

from sati.config_schemas import RootConfig
from sati.hashing import (
    checkpoint_hash,
    config_hash,
    model_config_hash,
    preprocessing_hash,
)
from sati.manifests import (
    ANNOTATIONS_SCHEMA,
    FULLFRAMES_SCHEMA,
    SLICE_ANNOTATIONS_SCHEMA,
    SLICES_SCHEMA,
    VALIDATION_RESULTS_SCHEMA,
    read_manifest,
    read_metadata_sidecar,
    verify_metadata_hashes,
)

inspect_app = typer.Typer(name="inspect", no_args_is_help=True)
console = Console()

JsonFlag = Annotated[bool, typer.Option("--json", help="machine-readable dump")]


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, f"{dotted}."))
        else:
            out[dotted] = value
    return out


def _resolve_config_path(path: Path) -> Path:
    return path / "resolved.yaml" if path.is_dir() else path


def summarize_config(path: Path, diff: Path | None = None) -> dict:
    """Validation status, derived values, and the hash family the config would
    produce — via the same ``sati.hashing`` code path the writers use."""
    config_path = _resolve_config_path(path)
    container = yaml.safe_load(config_path.read_text())
    summary: dict[str, Any] = {"path": str(config_path)}
    try:
        cfg = RootConfig.model_validate(container)
    except Exception as err:  # report, don't raise: inspect is diagnostic
        summary.update(valid=False, errors=str(err))
        return summary
    summary.update(
        valid=True,
        run_id=cfg.run_id,
        derived={"model.imgsz": cfg.model.imgsz},
        hashes={
            "config_hash": config_hash(cfg),
            "model_config_hash": model_config_hash(cfg.model),
            "preprocessing_hash": preprocessing_hash(cfg.slicing),
            "dataset_hash": None,  # needs a materialized dataset, not a config
        },
    )
    if diff is not None:
        other = yaml.safe_load(_resolve_config_path(diff).read_text())
        flat_a, flat_b = _flatten(container), _flatten(other)
        summary["diff"] = {
            key: {"this": flat_a.get(key), "other": flat_b.get(key)}
            for key in sorted(set(flat_a) | set(flat_b))
            if flat_a.get(key) != flat_b.get(key)
        }
    return summary


def summarize_dataset(root: Path) -> dict:
    manifests_dir = root / "manifests" if (root / "manifests").is_dir() else root
    fullframes = read_manifest(manifests_dir / "fullframes.parquet", FULLFRAMES_SCHEMA).to_pylist()
    annotations = read_manifest(
        manifests_dir / "annotations.parquet", ANNOTATIONS_SCHEMA
    ).to_pylist()
    sidecar = read_metadata_sidecar(manifests_dir)

    per_split: dict[str, int] = {}
    for frame in fullframes:
        per_split[frame["dataset_split"]] = per_split.get(frame["dataset_split"], 0) + 1
    per_class: dict[str, int] = {}
    for ann in annotations:
        per_class[ann["class_name"]] = per_class.get(ann["class_name"], 0) + 1
    frames_with_anns = {a["fullframe_id"] for a in annotations}
    widths = [f["fullres_wh"][0] for f in fullframes]
    heights = [f["fullres_wh"][1] for f in fullframes]
    long_sides = [max(a["bbox_w"], a["bbox_h"]) for a in annotations]

    return {
        "n_frames": len(fullframes),
        "n_annotations": len(annotations),
        "per_split": per_split,
        "class_map": sidecar.class_map,
        "per_class_counts": dict(sorted(per_class.items())),
        "zero_annotation_frames": len(fullframes) - len(frames_with_anns),
        "frame_wh": {
            "w_min": min(widths), "w_max": max(widths),
            "h_min": min(heights), "h_max": max(heights),
        },
        "box_long_side": (
            {
                "min": float(np.min(long_sides)),
                "p50": float(np.median(long_sides)),
                "max": float(np.max(long_sides)),
            }
            if long_sides
            else None
        ),
    }


def summarize_slices(root: Path, spot_check: int = 100) -> dict:
    manifests_dir = root / "manifests"
    slices = read_manifest(manifests_dir / "slices.parquet", SLICES_SCHEMA).to_pylist()
    slice_anns = read_manifest(
        manifests_dir / "slice_annotations.parquet", SLICE_ANNOTATIONS_SCHEMA
    ).to_pylist()

    per_level: dict[int, dict[str, int]] = {}
    stretched = {"x": 0, "y": 0, "none": 0}
    for row in slices:
        entry = per_level.setdefault(row["level"], {"slices": 0, "objects": 0})
        entry["slices"] += 1
        axes = row["stretched_axes"]
        if not axes:
            stretched["none"] += 1
        for axis in axes:
            stretched[axis] += 1
    drops = {"dropped_by_size_gate": 0, "dropped_by_min_area_ratio": 0}
    for row in slice_anns:
        per_level.setdefault(row["level"], {"slices": 0, "objects": 0})["objects"] += 1
        for key in drops:
            drops[key] += row[key]

    yolo_root = root / "yolo"
    disk_bytes = sum(p.stat().st_size for p in yolo_root.rglob("*") if p.is_file())
    missing = [
        row["slice_uri"]
        for row in slices[:spot_check]
        if not (root / row["slice_uri"]).is_file()
    ]
    return {
        "per_level": {str(k): v for k, v in sorted(per_level.items())},
        "drop_tallies": drops,
        "stretched_axis_frequency": stretched,
        "yolo_tree_bytes": disk_bytes,
        "spot_check": {"checked": min(spot_check, len(slices)), "missing": missing},
    }


def summarize_results(parquet_path: Path) -> dict:
    table = read_manifest(parquet_path, VALIDATION_RESULTS_SCHEMA)
    rows = table.to_pylist()
    by_type: dict[str, int] = {}
    run_ids: set[str] = set()
    hash_families: dict[str, set] = {}
    for row in rows:
        by_type[row["record_type"]] = by_type.get(row["record_type"], 0) + 1
        run_ids.add(row["run_id"])
        for key in ("config_hash", "model_config_hash", "checkpoint_hash", "dataset_hash"):
            if row[key] is not None:
                hash_families.setdefault(key, set()).add(row[key])
    sidecar = read_metadata_sidecar(parquet_path.parent)
    metrics_path = parquet_path.parent / "metrics.json"
    return {
        "n_rows": len(rows),
        "record_types": by_type,
        "run_ids": sorted(run_ids),
        "hash_families": {k: sorted(v) for k, v in sorted(hash_families.items())},
        "sidecar": {
            "schema_version": sidecar.schema_version,
            "hash_verification": verify_metadata_hashes(sidecar),
        },
        "metrics": json.loads(metrics_path.read_text()) if metrics_path.is_file() else None,
    }


def summarize_model(ckpt: Path) -> dict:
    summary: dict[str, Any] = {"checkpoint_hash": checkpoint_hash(ckpt)}
    try:  # best-effort metadata; never a forward pass
        import torch

        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        args = payload.get("train_args", {}) if isinstance(payload, dict) else {}
        summary["ultralytics"] = {
            "arch": str(args.get("model", "")),
            "imgsz": args.get("imgsz"),
            "epoch": payload.get("epoch") if isinstance(payload, dict) else None,
        }
    except Exception as err:
        summary["ultralytics"] = None
        summary["metadata_note"] = f"could not read checkpoint metadata: {err}"
    sibling = ckpt.parent / "inference_config.yaml"
    summary["inference_config"] = (
        yaml.safe_load(sibling.read_text()) if sibling.is_file() else None
    )
    return summary


def _emit(summary: dict, as_json: bool, title: str) -> None:
    if as_json:
        console.print_json(json.dumps(summary, default=str))
        return
    table = Table(title=title)
    table.add_column("field")
    table.add_column("value")
    for key, value in _flatten(summary).items():
        table.add_row(key, str(value))
    console.print(table)


@inspect_app.command("config")
def inspect_config(
    path: Path,
    diff: Annotated[Path | None, typer.Option("--diff")] = None,
    as_json: JsonFlag = False,
) -> None:
    _emit(summarize_config(path, diff), as_json, "inspect config")


@inspect_app.command("dataset")
def inspect_dataset(path: Path, as_json: JsonFlag = False) -> None:
    _emit(summarize_dataset(path), as_json, "inspect dataset")


@inspect_app.command("slices")
def inspect_slices(path: Path, as_json: JsonFlag = False) -> None:
    _emit(summarize_slices(path), as_json, "inspect slices")


@inspect_app.command("results")
def inspect_results(path: Path, as_json: JsonFlag = False) -> None:
    _emit(summarize_results(path), as_json, "inspect results")


@inspect_app.command("model")
def inspect_model(path: Path, as_json: JsonFlag = False) -> None:
    _emit(summarize_model(path), as_json, "inspect model")
