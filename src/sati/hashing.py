"""Versioned canonical identity and compatibility hashes for SATI."""

from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from typing import Any, Literal

import coolname

from sati.config_schemas import (
    AutoLevelPolicyConfig,
    CommandName,
    ModelConfig,
    RecombinationConfig,
    RootConfig,
    SlicingConfig,
    StartThenGeometricLevelPolicyConfig,
)

PATH_SENTINEL = "<path>"

CONFIG_HASH_RECIPE = "sati.config_hash/2"
DATASET_HASH_RECIPE = "sati.dataset_hash/2"
PREPARED_DATASET_HASH_RECIPE = "sati.prepared_dataset_hash/1"
TARGET_SCHEMA_HASH_RECIPE = "sati.target_schema_hash/1"
CLASS_MAPPING_HASH_RECIPE = "sati.class_mapping_hash/1"
MODEL_CONFIG_HASH_RECIPE = "sati.model_config_hash/2"
PREPROCESSING_HASH_RECIPE = "sati.preprocessing_hash/3"
RECOMBINATION_HASH_RECIPE = "sati.recombination_hash/2"
CHECKPOINT_HASH_RECIPE = "sati.checkpoint_hash/1"
MODEL_HASH_RECIPE = "sati.model_hash/1"
STRETCH_POLICY = "stretch-to-nominal-sharp-source/v2"


def canonical_json(obj: Any) -> str:
    """Serialize a canonical JSON scalar/tree with stable float handling."""
    if obj is None:
        return "null"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        if math.isnan(obj):
            return '"NaN"'
        if math.isinf(obj):
            return '"Infinity"' if obj > 0 else '"-Infinity"'
        return f"{obj:.12g}"
    if isinstance(obj, str):
        escaped = obj.replace("\\", "\\\\").replace('"', '\\"')
        escaped = "".join(
            char if char >= " " else f"\\u{ord(char):04x}" for char in escaped
        )
        return f'"{escaped}"'
    if isinstance(obj, dict):
        if not all(isinstance(key, str) for key in obj):
            raise TypeError("canonical_json requires string dict keys")
        return "{" + ",".join(
            f"{canonical_json(key)}:{canonical_json(value)}"
            for key, value in sorted(obj.items())
        ) + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(canonical_json(value) for value in obj) + "]"
    raise TypeError(f"canonical_json cannot serialize {type(obj).__name__}")


def sha256_of(canonical: str) -> str:
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hash_source(source: dict[str, Any]) -> str:
    return sha256_of(canonical_json(source))


def file_hash(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def checkpoint_hash(path: Path | str) -> str:
    return file_hash(path)


def model_hash(path: Path | str) -> str:
    return file_hash(path)


def _seeded_slug(domain: str, value: str) -> str:
    digest = hashlib.sha256(f"{domain}:{value}".encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    generator = coolname.generate.__self__
    previous_random = generator.random
    try:
        coolname.replace_random(rng)
        return coolname.generate_slug(2)
    finally:
        coolname.replace_random(previous_random)


def deterministic_id(domain: str, digest: str) -> str:
    """Two-word deterministic human ID, domain separated like dojo templates."""
    return _seeded_slug(domain, digest)


def fresh_run_id() -> str:
    return coolname.generate_slug(2)


def config_id(config_digest: str) -> str:
    return deterministic_id("config_hash", config_digest)


def prepared_dataset_id(prepared_digest: str) -> str:
    return deterministic_id("prepared_dataset_hash", prepared_digest)


def _portable_path(value: Path | str | None) -> str | None:
    return None if value is None else PATH_SENTINEL


def _weights_source(model: ModelConfig) -> dict[str, Any]:
    source = model.weights.model_dump(mode="json", exclude_none=True)
    uri = model.weights.uri
    if model.weights.source == "checkpoint" and uri and Path(uri).is_file():
        source["uri"] = checkpoint_hash(uri)
    return source


_ACTIVE_BLOCKS: dict[CommandName, tuple[str, ...]] = {
    "prepare": ("experiment", "task", "runtime", "data", "slicing"),
    "convert": ("experiment", "task", "runtime", "data"),
    "train": ("experiment", "task", "runtime", "data", "slicing", "model", "training"),
    "eval": (
        "experiment",
        "task",
        "runtime",
        "data",
        "slicing",
        "recombination",
        "results",
    ),
    "infer": ("experiment", "task", "runtime", "slicing", "recombination", "results"),
    "export": ("experiment", "task", "runtime"),
}


def config_hash_source(cfg: RootConfig, command: CommandName | None = None) -> dict[str, Any]:
    """Project the active command's behavior from the resolved config."""
    active_command = command or cfg.runtime.command
    if active_command not in _ACTIVE_BLOCKS:
        raise ValueError("config_hash requires an active command")
    dump = cfg.model_dump(mode="json", exclude_none=True)
    source = {key: dump[key] for key in _ACTIVE_BLOCKS[active_command]}
    runtime = source.get("runtime", {})
    for key in ("run_id", "config_id", "config_hash", "command"):
        runtime.pop(key, None)

    data = source.get("data")
    if isinstance(data, dict):
        raw = data.get("source")
        if isinstance(raw, dict):
            for key in ("uri", "split_file", "image_root"):
                if key in raw:
                    raw[key] = _portable_path(raw[key])
        prepared = data.get("prepared")
        if isinstance(prepared, dict) and "uri" in prepared:
            prepared["uri"] = _portable_path(prepared["uri"])

    model = source.get("model")
    if isinstance(model, dict):
        model["weights"] = _weights_source(cfg.model)
    return {
        "recipe": CONFIG_HASH_RECIPE,
        "command": active_command,
        "config": source,
    }


def config_hash_full_source(
    cfg: RootConfig, command: CommandName | None = None
) -> dict[str, Any]:
    return config_hash_source(cfg, command)


def config_hash(cfg: RootConfig, command: CommandName | None = None) -> str:
    return hash_source(config_hash_source(cfg, command))


def target_schema_hash_full_source(cfg: RootConfig) -> dict[str, Any]:
    return {
        "recipe": TARGET_SCHEMA_HASH_RECIPE,
        "task": cfg.task.model_dump(mode="json"),
        "target": {
            "structure": "class_id+bbox",
            "bbox_dtype": "float64",
        },
    }


def target_schema_hash(cfg: RootConfig) -> str:
    return hash_source(target_schema_hash_full_source(cfg))


def class_mapping_hash_full_source(class_mapping: dict[int, str]) -> dict[str, Any]:
    return {
        "recipe": CLASS_MAPPING_HASH_RECIPE,
        "classes": [
            {"index": int(index), "name": name}
            for index, name in class_mapping.items()
        ],
    }


def class_mapping_hash(class_mapping: dict[int, str]) -> str:
    return hash_source(class_mapping_hash_full_source(class_mapping))


def model_config_hash_full_source(model: ModelConfig) -> dict[str, Any]:
    return {
        "recipe": MODEL_CONFIG_HASH_RECIPE,
        "model": {
            "architecture": model.architecture.model_dump(mode="json"),
            "input_shape": [model.imgsz, model.imgsz],
            "num_classes": model.num_classes,
        },
    }


def model_config_hash(model: ModelConfig) -> str:
    return hash_source(model_config_hash_full_source(model))


def level_policy_hash_source(slicing: SlicingConfig) -> dict[str, Any]:
    policy = slicing.level_policy
    if isinstance(policy, AutoLevelPolicyConfig):
        return {"kind": "auto", "downsample_factor": policy.downsample_factor}
    if isinstance(policy, StartThenGeometricLevelPolicyConfig):
        return {
            "kind": "start_then_geometric",
            "start_scale": policy.start_scale,
            "subsequent_downsample_factor": policy.subsequent_downsample_factor,
            "stop_when_long_side_lte_nominal": policy.stop_when_long_side_lte_nominal,
        }
    raise TypeError(f"unsupported level policy: {policy!r}")


def preprocessing_hash_full_source(
    slicing: SlicingConfig,
    level_stack_wh: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    return {
        "recipe": PREPROCESSING_HASH_RECIPE,
        "nominal_slice_size": slicing.nominal_slice_size,
        "overlap_ratio": slicing.overlap_ratio,
        "level_policy": level_policy_hash_source(slicing),
        "max_levels": slicing.max_levels,
        "stretch_policy": STRETCH_POLICY,
        "image_format": slicing.image_format,
        "level_stack": (
            None
            if level_stack_wh is None
            else [{"w": width, "h": height} for width, height in level_stack_wh]
        ),
    }


def preprocessing_hash(
    slicing: SlicingConfig,
    level_stack_wh: list[tuple[int, int]] | None = None,
) -> str:
    return hash_source(preprocessing_hash_full_source(slicing, level_stack_wh))


def recombination_hash_full_source(
    recombination: RecombinationConfig,
    scope: Literal["level_merge", "final_merge"],
    *,
    strategy_name: str = "nms",
    strategy_version: int = 1,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "confidence_threshold": recombination.confidence_threshold,
        "class_agnostic": recombination.class_agnostic,
        "within_level_iou": recombination.within_level_iou,
    }
    if scope == "final_merge":
        params["cross_level_iou"] = recombination.cross_level_iou
    return {
        "recipe": RECOMBINATION_HASH_RECIPE,
        "strategy": strategy_name,
        "version": strategy_version,
        "scope": scope,
        "params": params,
    }


def recombination_hash(
    recombination: RecombinationConfig,
    scope: Literal["level_merge", "final_merge"],
    *,
    strategy_name: str = "nms",
    strategy_version: int = 1,
) -> str:
    return hash_source(
        recombination_hash_full_source(
            recombination,
            scope,
            strategy_name=strategy_name,
            strategy_version=strategy_version,
        )
    )


def dataset_hash_full_source(
    fullframes: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    class_mapping: dict[int, str],
) -> dict[str, Any]:
    """Logical full-frame/annotation identity; no pixels or slice artifacts."""
    canonical_frames = [
        {
            "fullframe_id": row["fullframe_id"],
            "source_identity": Path(row["source_uri"]).name,
            "dataset_split": row["dataset_split"],
            "fullres_wh": list(row["fullres_wh"]),
            "color_mode": row["color_mode"],
        }
        for row in sorted(fullframes, key=lambda item: item["fullframe_id"])
    ]
    canonical_annotations = [
        {
            "fullframe_id": row["fullframe_id"],
            "ann_id": int(row["ann_id"]),
            "class_id": int(row["class_id"]),
            "bbox_xyxy": list(row["bbox_xyxy"]),
        }
        for row in sorted(
            annotations,
            key=lambda item: (item["fullframe_id"], int(item["ann_id"])),
        )
    ]
    return {
        "recipe": DATASET_HASH_RECIPE,
        "fullframes": canonical_frames,
        "annotations": canonical_annotations,
        "class_indices": list(class_mapping),
    }


def dataset_hash(
    fullframes: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    class_mapping: dict[int, str],
) -> str:
    return hash_source(dataset_hash_full_source(fullframes, annotations, class_mapping))


def prepared_dataset_hash_full_source(
    logical_dataset_hash: str,
    slicing: SlicingConfig,
) -> dict[str, Any]:
    return {
        "recipe": PREPARED_DATASET_HASH_RECIPE,
        "dataset_hash": logical_dataset_hash,
        "preprocessing": preprocessing_hash_full_source(slicing),
        "target_gates": {
            "min_area_ratio": slicing.min_area_ratio,
            "size_gate_px": slicing.size_gate_px,
        },
        "materialization_schema_version": 2,
        "mode": "materialized_yolo",
    }


def prepared_dataset_hash(logical_dataset_hash: str, slicing: SlicingConfig) -> str:
    return hash_source(prepared_dataset_hash_full_source(logical_dataset_hash, slicing))


# Removed public names deliberately retained only as import-time errors through absence:
# class_map_hash, class_map_hash_full_source, dataset file-list hashing.
