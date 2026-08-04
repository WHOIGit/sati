"""Read-only config and artifact introspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

from sati.checkpoints import read_checkpoint_metadata
from sati.config_loader import resolve_configuration
from sati.config_schemas import (
    CommandName,
    RootConfig,
    SatiPathMarker,
    annotated_field_paths,
)
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


# Runtime identities that are generated, not authored, so ``config_hash``
# excludes them from its source projection (see hashing.config_hash_source).
_GENERATED_RUNTIME_PATHS = {
    ("runtime", "run_id"),
    ("runtime", "config_id"),
    ("runtime", "config_hash"),
    ("runtime", "command"),
}
# The slicing subfields ``preprocessing_hash`` reads (see
# hashing.preprocessing_hash_full_source); the rest of ``slicing`` (target
# gates) feeds other identities, not preprocessing.
_PREPROCESSING_SLICING_FIELDS = frozenset(
    {"nominal_slice_size", "overlap_ratio", "level_policy", "max_levels", "image_format"}
)
# Portable paths are replaced by ``<path>`` before hashing, so they never move a
# behavioral hash — a path-only difference must land in ``other``.
_SATIPATH_PATHS = frozenset(annotated_field_paths(RootConfig, SatiPathMarker))


def _covering_hashes(path: str, active_blocks: frozenset[str]) -> set[str]:
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
    if top == "slicing" and len(parts) >= 2 and parts[1] in _PREPROCESSING_SLICING_FIELDS:
        covers.add("preprocessing_hash")
    if (
        top in active_blocks
        and parts not in _GENERATED_RUNTIME_PATHS
        and parts not in _SATIPATH_PATHS
    ):
        covers.add("config_hash")
    return covers


def summarize_config_compare(left: RootConfig, right: RootConfig) -> dict[str, Any]:
    """Group every config difference under the hash source(s) that read it.

    Differences no behavioral hash reads — portable paths, generated identities,
    fields inactive for the command — fall to ``other``. Everything is keyed by
    full-config dotted path, so ``other`` reads as residual config differences.
    """
    try:
        active_blocks = frozenset(config_hash_source(left).get("config", {}))
    except ValueError:
        active_blocks = frozenset()
    grouped: dict[str, dict[str, Any]] = {
        "config_hash": {},
        "target_schema_hash": {},
        "model_config_hash": {},
        "preprocessing_hash": {},
        "other": {},
    }
    all_diffs = _diff(left.model_dump(mode="json"), right.model_dump(mode="json"))
    for path, delta in all_diffs.items():
        covers = _covering_hashes(path, active_blocks)
        if covers:
            for name in covers:
                grouped[name][path] = delta
        else:
            grouped["other"][path] = delta
    return grouped


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
    return {
        "n_frames": len(fullframes),
        "n_annotations": len(annotations),
        "per_split": per_split,
        "class_mapping": metadata.class_mapping,
        "per_class_counts": per_class,
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
    missing = [
        row["slice_uri"]
        for row in slices[:spot_check]
        if not (root / row["slice_uri"]).is_file()
    ]
    return {
        "per_level": {str(key): value for key, value in sorted(per_level.items())},
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
    run_root = parquet.parent.parent if parquet.parent.name == "results" else parquet.parent
    metrics_path = run_root / "metrics" / "metrics.json"
    lifecycle_path = run_root / "run_manifest.json"
    return {
        "path": str(parquet),
        "n_rows": len(rows),
        "record_types": by_type,
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
