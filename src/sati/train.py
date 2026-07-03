"""``sati train`` orchestration (v1 §8).

compose_and_resolve → preprocessing-hash gate → stock ultralytics train →
hash the produced checkpoint. That's the whole command.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from sati.backends import Backend, UltralyticsBackend
from sati.config_schemas import RootConfig
from sati.hashing import checkpoint_hash, preprocessing_hash


class TrainRefusedError(RuntimeError):
    """A hard gate failed before any model was loaded."""


def locate_data_yaml(cfg: RootConfig) -> Path:
    """``data.source_path`` for train points at a prepared dataset: either the
    prepare run's root (containing ``yolo/data.yaml``) or a ``data.yaml`` itself."""
    source = cfg.data.source_path
    if source.is_file() and source.name == "data.yaml":
        return source
    candidate = source / "yolo" / "data.yaml"
    if candidate.is_file():
        return candidate
    raise TrainRefusedError(
        f"no prepared dataset at {source}: expected <root>/yolo/data.yaml "
        "(run `sati prepare` first) or a direct path to a data.yaml"
    )


def check_preprocessing_gate(data_yaml: Path, cfg: RootConfig) -> None:
    """§14: the YOLO tree is derived; refuse to train when the composed slicing
    config doesn't match the hash embedded at prepare time."""
    spec = yaml.safe_load(data_yaml.read_text())
    embedded = spec.get("sati_preprocessing_hash")
    if embedded is None:
        raise TrainRefusedError(
            f"{data_yaml} carries no sati_preprocessing_hash; not a SATI-prepared tree"
        )
    composed = preprocessing_hash(cfg.slicing)
    if embedded != composed:
        raise TrainRefusedError(
            "preprocessing_hash mismatch — the materialized tree was prepared with "
            f"different slicing config (tree: {embedded}, composed: {composed}). "
            "Re-run `sati prepare` or compose the matching slicing options."
        )


def execute_train(cfg: RootConfig, backend: Backend | None = None) -> tuple[Path, str]:
    """Returns ``(checkpoint_path, checkpoint_hash)``."""
    backend = backend or UltralyticsBackend()
    data_yaml = locate_data_yaml(cfg)
    check_preprocessing_gate(data_yaml, cfg)
    checkpoint = backend.train(data_yaml, cfg)
    return checkpoint, checkpoint_hash(checkpoint)
