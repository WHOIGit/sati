"""Versioned canonical identity and compatibility hashes for SATI.

Mechanically aligned with the sibling ``image-classifier-dojo`` project
(``align-dojo-hash-mechanics``): sources project in the config's own bare key
namespace, carry a ``version`` field (no inline recipe string; ``config_hash``
carries no version token), and serialize through the shared ``_canonicalize`` +
``json.dumps`` canonical-JSON path. Portable-path and generated-identity
exclusion is type-driven via ``SatiPathMarker`` / ``HashDerivedMarker``.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Literal

import coolname

from sati.config_schemas import (
    AutoLevelPolicyConfig,
    CommandName,
    HashDerivedMarker,
    ModelConfig,
    RecombinationConfig,
    RootConfig,
    SatiPathMarker,
    SlicingConfig,
    StartThenGeometricLevelPolicyConfig,
    annotated_field_paths,
)
from sati.tiling import MERGE_STRATEGY_VERSIONS, merge_strategy

# Source versions (dojo-style `version` field). Bumped on the mechanics change.
# `config_hash` intentionally carries no version token (its identity is content).
DATASET_HASH_VERSION = "3"
PREPARED_DATASET_HASH_VERSION = "2"
TARGET_SCHEMA_HASH_VERSION = "2"
CLASS_MAPPING_HASH_VERSION = "2"
MODEL_CONFIG_HASH_VERSION = "3"
PREPROCESSING_HASH_VERSION = "4"
RECOMBINATION_HASH_VERSION = "4"
CHECKPOINT_HASH_VERSION = "1"
MODEL_HASH_VERSION = "1"
STRETCH_POLICY = "stretch-to-nominal-sharp-source/v2"


def _canonicalize(value: Any) -> Any:
    """Normalize a value for canonical JSON, matching the sibling project."""
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical_json requires string dict keys")
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return float(f"{value:.12g}")
    return value


def canonical_json(obj: Any) -> str:
    """Serialize a canonical JSON string with stable key order and floats."""
    return json.dumps(
        _canonicalize(obj),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


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

# Type-driven exclusion: portable paths (SatiPathMarker) and generated/derived
# identities (HashDerivedMarker) never contribute to a behavioral config source.
_EXCLUDED_SOURCE_PATHS: tuple[tuple[str, ...], ...] = (
    *annotated_field_paths(RootConfig, SatiPathMarker),
    *annotated_field_paths(RootConfig, HashDerivedMarker),
)


def _remove_dotted(node: Any, path: tuple[str, ...]) -> None:
    for part in path[:-1]:
        if not isinstance(node, dict) or part not in node:
            return
        node = node[part]
    if isinstance(node, dict):
        node.pop(path[-1], None)


def config_hash_source(cfg: RootConfig, command: CommandName | None = None) -> dict[str, Any]:
    """Bare, command-aware projection of the resolved config's behavior.

    Top-level keys are config field names (no ``recipe``/``command``/``config``
    envelope). Portable paths and generated/derived identities are excluded by
    walking their markers; the ``model.weights`` byte-substitution exception is
    applied last so a local initialization checkpoint contributes its bytes.
    """
    active_command = command or cfg.runtime.command
    if active_command not in _ACTIVE_BLOCKS:
        raise ValueError("config_hash requires an active command")
    dump = cfg.model_dump(mode="json", exclude_none=True)
    active = _ACTIVE_BLOCKS[active_command]
    source = {key: dump[key] for key in active if key in dump}
    for path in _EXCLUDED_SOURCE_PATHS:
        if path[0] in active:
            _remove_dotted(source, path)
    if "model" in source:
        source["model"]["weights"] = _weights_source(cfg.model)
    return source


def config_hash_full_source(
    cfg: RootConfig, command: CommandName | None = None
) -> dict[str, Any]:
    return config_hash_source(cfg, command)


def config_hash(cfg: RootConfig, command: CommandName | None = None) -> str:
    return hash_source(config_hash_source(cfg, command))


def target_schema_hash_full_source(cfg: RootConfig) -> dict[str, Any]:
    # The coordinate convention is a versioned constant of the target contract,
    # not authored task configuration (`task` carries only `type`).
    return {
        "version": TARGET_SCHEMA_HASH_VERSION,
        "task": {"type": cfg.task.type},
        "target": {
            "structure": "class_id+bbox",
            "bbox_dtype": "float64",
            "box_format": "xyxy",
            "box_bounds": "half_open",
            "origin": "top_left",
        },
    }


def target_schema_hash(cfg: RootConfig) -> str:
    return hash_source(target_schema_hash_full_source(cfg))


def class_mapping_hash_full_source(class_mapping: dict[int, str]) -> dict[str, Any]:
    return {
        "version": CLASS_MAPPING_HASH_VERSION,
        "classes": [
            {"index": int(index), "name": name}
            for index, name in class_mapping.items()
        ],
    }


def class_mapping_hash(class_mapping: dict[int, str]) -> str:
    return hash_source(class_mapping_hash_full_source(class_mapping))


def model_config_hash_full_source(model: ModelConfig) -> dict[str, Any]:
    return {
        "version": MODEL_CONFIG_HASH_VERSION,
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
        "version": PREPROCESSING_HASH_VERSION,
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
    strategy_name: str | None = None,
    strategy_version: int | None = None,
) -> dict[str, Any]:
    """Scope-aware source: within-level params on ``level_merge``, both on ``final_merge``.

    ``method`` is deliberately not duplicated into ``params`` — it is already the
    ``strategy`` field. ``max_merge_growth`` is included in both scopes even under
    ``method: nms``, where it is inert, so a run's identity always records the
    bound that would have applied rather than making the param set
    method-dependent.
    """
    if strategy_name is None:
        strategy_name, resolved_version = merge_strategy(recombination)
        if strategy_version is None:
            strategy_version = resolved_version
    elif strategy_version is None:
        strategy_version = MERGE_STRATEGY_VERSIONS[strategy_name]
    params: dict[str, Any] = {
        "confidence_threshold": recombination.confidence_threshold,
        "class_agnostic": recombination.class_agnostic,
        "match_metric": recombination.match_metric,
        "max_merge_growth": recombination.max_merge_growth,
        "within_level_threshold": recombination.within_level_threshold,
    }
    if scope == "final_merge":
        params["cross_level_threshold"] = recombination.cross_level_threshold
    return {
        "version": RECOMBINATION_HASH_VERSION,
        "strategy": strategy_name,
        "strategy_version": strategy_version,
        "scope": scope,
        "params": params,
    }


def recombination_hash(
    recombination: RecombinationConfig,
    scope: Literal["level_merge", "final_merge"],
    *,
    strategy_name: str | None = None,
    strategy_version: int | None = None,
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
        "version": DATASET_HASH_VERSION,
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
        "version": PREPARED_DATASET_HASH_VERSION,
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
