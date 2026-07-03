"""Canonical JSON and the compatibility-hash family (DESIGN-DOC-v1 §4.7).

All hashes are SHA-256 over canonical JSON of a deliberately chosen source object:
sorted keys, no whitespace, floats ``%.12g``, non-finite floats tokenized as strings,
``sha256:`` prefix, versioned source recipe.

Only the config-side hashes live here for now (``config_hash``, ``model_config_hash``);
dataset/preprocessing/checkpoint/recombination hashes arrive with their owning modules
and reuse :func:`canonical_json` / :func:`sha256_of`.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from typing import Literal

from sati.config_schemas import (
    HashDerivedMarker,
    ModelConfig,
    RecombinationConfig,
    RootConfig,
    SatiPathMarker,
    SlicingConfig,
    annotated_field_paths,
)

PATH_SENTINEL = "<path>"

CONFIG_HASH_RECIPE = "sati.config_hash/1"
MODEL_CONFIG_HASH_RECIPE = "sati.model_config_hash/1"
RECOMBINATION_HASH_RECIPE = "sati.recombination_hash/1"
DATASET_HASH_RECIPE = "sati.dataset_hash/1"
CLASS_MAP_HASH_RECIPE = "sati.class_map_hash/1"
PREPROCESSING_HASH_RECIPE = "sati.preprocessing_hash/1"


def checkpoint_hash(path: Path | str) -> str:
    """SHA-256 over the raw bytes of a weights/checkpoint file (v1 §4.7)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _weights_hash_value(weights: str | None) -> str | None:
    """The value ``model.weights`` contributes to config hashes.

    Settled (Sidney, 2026-07-02): a file-backed weights entry contributes its
    content hash (== ``checkpoint_hash``), not its path string, so config_hash
    stays deterministic across machines; a hub asset name ("yolo26n.pt" that is
    not a local file) contributes the string verbatim.
    """
    if weights is not None and Path(weights).is_file():
        return checkpoint_hash(weights)
    return weights


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, %.12g floats, tokenized non-finites."""
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
        # json.dumps-equivalent escaping without importing the full encoder
        out = obj.replace("\\", "\\\\").replace('"', '\\"')
        out = "".join(c if c >= " " else f"\\u{ord(c):04x}" for c in out)
        return f'"{out}"'
    if isinstance(obj, dict):
        if not all(isinstance(k, str) for k in obj):
            raise TypeError("canonical_json requires string dict keys")
        items = (f"{canonical_json(k)}:{canonical_json(v)}" for k, v in sorted(obj.items()))
        return "{" + ",".join(items) + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(canonical_json(v) for v in obj) + "]"
    raise TypeError(f"canonical_json cannot serialize {type(obj).__name__}")


def sha256_of(canonical: str) -> str:
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _set_nested(d: dict, path: tuple[str, ...], value: Any) -> None:
    for key in path[:-1]:
        d = d[key]
    d[path[-1]] = value


def _del_nested(d: dict, path: tuple[str, ...]) -> None:
    for key in path[:-1]:
        d = d[key]
    d.pop(path[-1], None)


def config_hash_source(cfg: RootConfig) -> dict:
    """Resolved RootConfig minus path identities and derived-only fields.

    Type-driven (§4.7 decision): every ``SatiPath``-annotated value becomes the
    ``"<path>"`` sentinel (``None`` stays ``None`` — presence/absence of an optional
    path is behavioral); every ``HashDerived``-annotated field is dropped.
    """
    dump = cfg.model_dump(mode="json")
    for path in annotated_field_paths(RootConfig, SatiPathMarker):
        node = dump
        for key in path[:-1]:
            node = node[key]
        if node[path[-1]] is not None:
            _set_nested(dump, path, PATH_SENTINEL)
    for path in annotated_field_paths(RootConfig, HashDerivedMarker):
        _del_nested(dump, path)
    dump["model"]["weights"] = _weights_hash_value(cfg.model.weights)
    return dump


def config_hash_full_source(cfg: RootConfig) -> dict:
    """Recipe-wrapped source — the exact object the sidecar stores as ``*_source``."""
    return {"recipe": CONFIG_HASH_RECIPE, "config": config_hash_source(cfg)}


def config_hash(cfg: RootConfig) -> str:
    return sha256_of(canonical_json(config_hash_full_source(cfg)))


def model_config_hash_full_source(model: ModelConfig) -> dict:
    dump = model.model_dump(mode="json")
    dump["weights"] = _weights_hash_value(model.weights)
    return {"recipe": MODEL_CONFIG_HASH_RECIPE, "model": dump}


def model_config_hash(model: ModelConfig) -> str:
    return sha256_of(canonical_json(model_config_hash_full_source(model)))


def recombination_hash_full_source(
    recombination: RecombinationConfig,
    scope: Literal["level_merge", "final_merge"],
    *,
    strategy_name: str = "nms",
    strategy_version: int = 1,
) -> dict:
    """Strategy name + version + merge params, scoped per record_type (§4.7).

    ``level_merge`` digests within-level config only; ``final_merge`` digests
    within-level **and** cross-level config.
    """
    params: dict = {
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
    return sha256_of(
        canonical_json(
            recombination_hash_full_source(
                recombination,
                scope,
                strategy_name=strategy_name,
                strategy_version=strategy_version,
            )
        )
    )


def dataset_hash_full_source(files: list[tuple[str, int]]) -> dict:
    """Manifest file identities: relative path + size — no pixels, no mtime (§4.7).

    Entries are sorted by relative path so the hash is order-insensitive.
    """
    return {
        "recipe": DATASET_HASH_RECIPE,
        "files": [
            {"path": rel_path, "size": size}
            for rel_path, size in sorted(files)
        ],
    }


def dataset_hash(files: list[tuple[str, int]]) -> str:
    return sha256_of(canonical_json(dataset_hash_full_source(files)))


def class_map_hash_full_source(class_map: dict[int, str]) -> dict:
    """Ordered class-index→name map (§4.7); order is part of the identity."""
    return {
        "recipe": CLASS_MAP_HASH_RECIPE,
        "classes": [{"index": index, "name": name} for index, name in class_map.items()],
    }


def class_map_hash(class_map: dict[int, str]) -> str:
    return sha256_of(canonical_json(class_map_hash_full_source(class_map)))


def preprocessing_hash_full_source(
    slicing: "SlicingConfig", level_stack_wh: list[tuple[int, int]] | None = None
) -> dict:
    """Nominal slice, stack, overlap, stretch policy (§4.7).

    Gates ensembling of raw slice predictions only, never metric comparison.
    With ``level_stack_wh`` the hash is frame-level identity (realized stack);
    without it, dataset-level identity (the automatic-factor stack *rule*) —
    the form embedded in the generated ``data.yaml``.
    """
    return {
        "recipe": PREPROCESSING_HASH_RECIPE,
        "nominal_slice_size": slicing.nominal_slice_size,
        "overlap_ratio": slicing.overlap_ratio,
        "downsample_factor": slicing.downsample_factor,
        "max_levels": slicing.max_levels,
        "stretch_policy": "stretch-to-nominal/v1",
        "level_stack": (
            None
            if level_stack_wh is None
            else [{"w": w, "h": h} for w, h in level_stack_wh]
        ),
    }


def preprocessing_hash(
    slicing: "SlicingConfig", level_stack_wh: list[tuple[int, int]] | None = None
) -> str:
    return sha256_of(canonical_json(preprocessing_hash_full_source(slicing, level_stack_wh)))
