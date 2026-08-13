"""Read-only config and artifact introspection."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

from sati.checkpoints import (
    GEOMETRY_SLICING_FIELDS,
    PINNED_SLICING_FIELDS,
    read_checkpoint_metadata,
)
from sati.config_loader import resolve_configuration
from sati.config_schemas import CommandName, RootConfig
from sati.hashing import (
    checkpoint_hash,
    config_hash,
    config_hash_source,
    model_config_hash,
    preprocessing_hash,
    target_schema_hash,
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

OutputFormat = Annotated[
    Literal["text", "json"],
    typer.Option("--format", help="text or json"),
]


def _flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        dotted = f"{prefix}{key}"
        if isinstance(item, dict):
            result.update(_flatten(item, f"{dotted}."))
        else:
            result[dotted] = item
    return result


def _emit(summary: dict[str, Any], fmt: str, title: str) -> None:
    if fmt == "json":
        console.print_json(json.dumps(summary, default=str))
        return
    table = Table(title=title)
    table.add_column("field")
    table.add_column("value")
    for key, value in _flatten(summary).items():
        table.add_row(key, str(value))
    console.print(table)


def summarize_config(cfg: RootConfig) -> dict[str, Any]:
    return {
        "valid": True,
        "command": cfg.runtime.command,
        "run_id": cfg.runtime.run_id,
        "config_id": cfg.runtime.config_id,
        "active_output": str(cfg.output_for(cfg.runtime.command).dir),
        "hashes": {
            "config_hash": config_hash(cfg),
            "target_schema_hash": target_schema_hash(cfg),
            "model_config_hash": model_config_hash(cfg.model),
            "preprocessing_hash": preprocessing_hash(cfg.slicing),
        },
        "resolved": cfg.model_dump(mode="json", exclude_none=False),
    }


def _diff(left: Any, right: Any, prefix: str = "") -> dict[str, dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        result: dict[str, dict[str, Any]] = {}
        for key in sorted(set(left) | set(right)):
            result.update(_diff(left.get(key), right.get(key), f"{prefix}{key}."))
        return result
    if left == right:
        return {}
    return {prefix.removesuffix("."): {"left": left, "right": right}}


# The slicing subfields ``preprocessing_hash`` reads (see
# hashing.preprocessing_hash_full_source), split by whether a checkpoint pins them.
# The rest of ``slicing`` (target gates) feeds other identities, not preprocessing.
#
# Pinned fields are model-coupled: a difference here means the configs are not
# mutually runnable against one checkpoint. Geometry fields are overridable at
# infer/eval time, so a difference is a deliberate re-slice rather than a
# compatibility break, and grouping it apart keeps the two readable at a glance.
_PINNED_PREPROCESSING_FIELDS = frozenset(PINNED_SLICING_FIELDS)
_GEOMETRY_PREPROCESSING_FIELDS = frozenset(GEOMETRY_SLICING_FIELDS)
_PREPROCESSING_SLICING_FIELDS = _PINNED_PREPROCESSING_FIELDS | _GEOMETRY_PREPROCESSING_FIELDS


def _leaf_paths(value: Any, prefix: str = "") -> set[str]:
    """Dotted leaf paths of a nested mapping (lists count as leaves)."""
    if isinstance(value, dict) and value:
        paths: set[str] = set()
        for key, child in value.items():
            paths |= _leaf_paths(child, f"{prefix}{key}.")
        return paths
    return {prefix[:-1]} if prefix else set()


def _covering_hashes(path: str, config_paths: frozenset[str]) -> set[str]:
    """Behavioral hashes whose source reads this full-config dotted path."""
    parts = tuple(path.split("."))
    top = parts[0]
    covers: set[str] = set()
    if top == "task":
        covers.add("target_schema_hash")
    if top == "model" and (
        parts[1:2] == ("architecture",) or parts[1:] in {("num_classes",), ("imgsz",)}
    ):
        covers.add("model_config_hash")
    if top == "slicing" and len(parts) >= 2:
        if parts[1] in _PINNED_PREPROCESSING_FIELDS:
            covers.add("preprocessing_hash")
        elif parts[1] in _GEOMETRY_PREPROCESSING_FIELDS:
            # Reported apart from `preprocessing_hash`: re-slicing moves the
            # effective identity, not the checkpoint-pinned compatibility one.
            covers.add("effective_preprocessing_hash")
    # `config_hash_source` is now a bare, self-excluding projection: its leaf
    # paths are exactly the config paths it reads (dojo's idiom), so membership
    # decides coverage — generated/portable-path exclusion is already applied.
    if path in config_paths:
        covers.add("config_hash")
    return covers


def summarize_config_compare(left: RootConfig, right: RootConfig) -> dict[str, Any]:
    """Group every config difference under the hash source(s) that read it.

    Differences no behavioral hash reads — portable paths, generated identities,
    fields inactive for the command — fall to ``other``. Everything is keyed by
    full-config dotted path, so ``other`` reads as residual config differences.
    """
    config_paths: frozenset[str] = frozenset()
    try:
        config_paths = frozenset(
            _leaf_paths(config_hash_source(left)) | _leaf_paths(config_hash_source(right))
        )
    except ValueError:
        pass
    grouped: dict[str, dict[str, Any]] = {
        "config_hash": {},
        "target_schema_hash": {},
        "model_config_hash": {},
        "preprocessing_hash": {},
        "effective_preprocessing_hash": {},
        "other": {},
    }
    all_diffs = _diff(left.model_dump(mode="json"), right.model_dump(mode="json"))
    for path, delta in all_diffs.items():
        covers = _covering_hashes(path, config_paths)
        if covers:
            for name in covers:
                grouped[name][path] = delta
        else:
            grouped["other"][path] = delta
    return grouped


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
    }


def summarize_dataset(root: Path) -> dict[str, Any]:
    manifests = root / "manifests" if (root / "manifests").is_dir() else root
    fullframes = read_manifest(manifests / "fullframes.parquet", FULLFRAMES_SCHEMA).to_pylist()
    annotations = read_manifest(manifests / "annotations.parquet", ANNOTATIONS_SCHEMA).to_pylist()
    metadata = read_metadata_sidecar(manifests)
    per_split: dict[str, int] = {}
    per_class: dict[str, int] = {}
    for frame in fullframes:
        per_split[frame["dataset_split"]] = per_split.get(frame["dataset_split"], 0) + 1
    for annotation in annotations:
        name = annotation["class_name"]
        per_class[name] = per_class.get(name, 0) + 1
    frames_with_annotations = {annotation["fullframe_id"] for annotation in annotations}
    return {
        "n_frames": len(fullframes),
        "n_annotations": len(annotations),
        "zero_annotation_frames": len(fullframes) - len(frames_with_annotations),
        "per_split": per_split,
        "class_mapping": metadata.class_mapping,
        "per_class_counts": per_class,
        "image_width": _distribution([frame["fullres_wh"][0] for frame in fullframes]),
        "image_height": _distribution([frame["fullres_wh"][1] for frame in fullframes]),
        "box_long_side": _distribution(
            [max(a["bbox_w"], a["bbox_h"]) for a in annotations]
        ),
        "hashes": {
            name: entry.value for name, entry in metadata.hashes.items()
        },
        "hash_verification": verify_metadata_hashes(metadata),
    }


def summarize_slices(root: Path, spot_check: int = 100) -> dict[str, Any]:
    manifests = root / "manifests"
    slices = read_manifest(manifests / "slices.parquet", SLICES_SCHEMA).to_pylist()
    annotations = read_manifest(
        manifests / "slice_annotations.parquet",
        SLICE_ANNOTATIONS_SCHEMA,
    ).to_pylist()
    per_level: dict[int, dict[str, int]] = {}
    for row in slices:
        per_level.setdefault(row["level"], {"slices": 0, "objects": 0})["slices"] += 1
    for row in annotations:
        per_level.setdefault(row["level"], {"slices": 0, "objects": 0})["objects"] += 1
    stretched = {"x": 0, "y": 0, "both": 0, "none": 0}
    for row in slices:
        axes = set(row["stretched_axes"] or [])
        if axes >= {"x", "y"}:
            stretched["both"] += 1
        elif "x" in axes:
            stretched["x"] += 1
        elif "y" in axes:
            stretched["y"] += 1
        else:
            stretched["none"] += 1
    dropped = {
        "size_gate": sum(1 for row in annotations if row["dropped_by_size_gate"]),
        "min_area_ratio": sum(1 for row in annotations if row["dropped_by_min_area_ratio"]),
    }
    yolo_tree = root / "yolo"
    yolo_tree_bytes = (
        sum(f.stat().st_size for f in yolo_tree.rglob("*") if f.is_file())
        if yolo_tree.is_dir()
        else 0
    )
    missing = [
        row["slice_uri"]
        for row in slices[:spot_check]
        if not (root / row["slice_uri"]).is_file()
    ]
    return {
        "per_level": {str(key): value for key, value in sorted(per_level.items())},
        "dropped_targets": dropped,
        "stretched_axes": stretched,
        "yolo_tree_bytes": yolo_tree_bytes,
        "spot_check": {"checked": min(spot_check, len(slices)), "missing": missing},
    }


def _find_results(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = (
        path / "results" / "validation_results.parquet",
        path / "results" / "predictions.parquet",
        path / "validation_results.parquet",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f"no canonical result table under {path}")


def summarize_results(path: Path) -> dict[str, Any]:
    parquet = _find_results(path)
    rows = read_manifest(parquet, VALIDATION_RESULTS_SCHEMA).to_pylist()
    metadata = read_metadata_sidecar(parquet.parent)
    by_type: dict[str, int] = {}
    for row in rows:
        by_type[row["record_type"]] = by_type.get(row["record_type"], 0) + 1
    run_ids = sorted({row["run_id"] for row in rows if row.get("run_id")})
    hash_columns = (
        "config_hash",
        "model_config_hash",
        "checkpoint_hash",
        "dataset_hash",
        "prepared_dataset_hash",
        "target_schema_hash",
        "class_mapping_hash",
        "preprocessing_hash",
        "effective_preprocessing_hash",
    )
    hashes_present = {
        column: present
        for column in hash_columns
        if (present := sorted({row[column] for row in rows if row.get(column)}))
    }
    # The one comparison that says whether this table came off the pyramid its
    # checkpoint was trained on. Reported explicitly rather than left for a reader
    # to spot by diffing two opaque digests.
    re_sliced = hashes_present.get("preprocessing_hash") != hashes_present.get(
        "effective_preprocessing_hash"
    )
    merge_strategies = sorted(
        {row["merge_strategy_name"] for row in rows if row.get("merge_strategy_name")}
    )
    # how often class-agnostic matching actually collapsed a disagreement: a
    # single-element list means the cluster agreed, so counting the longer ones
    # is what makes the collapse visible rather than implied
    class_collapses = sum(
        1 for row in rows if row.get("merged_class_ids") and len(row["merged_class_ids"]) > 1
    )
    run_root = parquet.parent.parent if parquet.parent.name == "results" else parquet.parent
    metrics_path = run_root / "metrics" / "metrics.json"
    lifecycle_path = run_root / "run_manifest.json"
    return {
        "path": str(parquet),
        "n_rows": len(rows),
        "record_types": by_type,
        "run_ids": run_ids,
        "merge_strategies": merge_strategies,
        "class_collapsed_merges": class_collapses,
        "hashes_present": hashes_present,
        "re_sliced": re_sliced,
        "hash_verification": verify_metadata_hashes(metadata),
        "metrics": json.loads(metrics_path.read_text()) if metrics_path.is_file() else None,
        "lifecycle": (
            json.loads(lifecycle_path.read_text()) if lifecycle_path.is_file() else None
        ),
    }


def summarize_model(path: Path) -> dict[str, Any]:
    metadata_path = (
        path / "checkpoint_metadata.json"
        if path.is_dir()
        else path.parent / "checkpoint_metadata.json"
    )
    if metadata_path.is_file():
        metadata = read_checkpoint_metadata(metadata_path)
        summary = {"checkpoint_metadata": metadata}
        if path.is_file():
            summary["checkpoint_hash"] = checkpoint_hash(path)
        return summary
    sidecar = path / "inference_config.yaml" if path.is_dir() else path.parent / "inference_config.yaml"
    if sidecar.is_file():
        return {"inference_config": yaml.safe_load(sidecar.read_text())}
    raise ValueError(
        "no checkpoint_metadata.json or inference_config.yaml found; "
        "inspect canonical SATI artifacts rather than loading checkpoint tensors"
    )


def _load_resolved(path: Path) -> RootConfig:
    candidate = path / "config" / "resolved.yaml" if path.is_dir() else path
    return RootConfig.model_validate(yaml.safe_load(candidate.read_text()))


@inspect_app.command("config")
def inspect_config(
    overrides: Annotated[
        Optional[list[str]],
        typer.Argument(help="Hydra overrides"),
    ] = None,
    config_dir: Annotated[
        Optional[list[Path]],
        typer.Option("--config-dir"),
    ] = None,
    config: Annotated[Optional[Path], typer.Option("--config")] = None,
    resolved_config: Annotated[Optional[Path], typer.Option("--resolved-config")] = None,
    command: Annotated[Optional[CommandName], typer.Option("--command")] = None,
    fmt: OutputFormat = "text",
) -> None:
    selected_command: CommandName = command or "train"
    if resolved_config is not None and command is None:
        candidate = (
            resolved_config / "config" / "resolved.yaml"
            if resolved_config.is_dir()
            else resolved_config
        )
        payload = yaml.safe_load(candidate.read_text())
        selected_command = payload.get("runtime", {}).get("command")
        if selected_command is None:
            raise typer.BadParameter(
                "resolved config is missing runtime.command",
                param_hint="--resolved-config",
            )
    resolution = resolve_configuration(
        overrides or (),
        command=selected_command,
        config_dirs=config_dir or (),
        config_file=config,
        resolved_config_file=resolved_config,
        for_inspection=True,
    )
    _emit(summarize_config(resolution.config), fmt, "inspect config")


@inspect_app.command("config-compare")
def inspect_config_compare(
    left: Path,
    right: Path,
    fmt: OutputFormat = "text",
) -> None:
    _emit(
        summarize_config_compare(_load_resolved(left), _load_resolved(right)),
        fmt,
        "inspect config-compare",
    )


@inspect_app.command("dataset")
def inspect_dataset(path: Path, fmt: OutputFormat = "text") -> None:
    _emit(summarize_dataset(path), fmt, "inspect dataset")


@inspect_app.command("slices")
def inspect_slices(path: Path, fmt: OutputFormat = "text") -> None:
    _emit(summarize_slices(path), fmt, "inspect slices")


@inspect_app.command("results")
def inspect_results(path: Path, fmt: OutputFormat = "text") -> None:
    _emit(summarize_results(path), fmt, "inspect results")


@inspect_app.command("model")
def inspect_model(path: Path, fmt: OutputFormat = "text") -> None:
    _emit(summarize_model(path), fmt, "inspect model")
