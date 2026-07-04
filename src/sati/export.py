"""``sati export`` (v1 §10): deployment artifact + ``inference_config.yaml``.

Wraps the backend's native export and writes the sidecar that lets deployment
reproduce the exact multi-resolution sampling with no other artifact: slicing
recipe, stretch policy, recombination params, class map, and the hash family.
``checkpoint_hash`` digests the source ``.pt``; export artifacts don't change it.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from sati.backends import Backend, UltralyticsBackend
from sati.config_schemas import RootConfig
from sati.hashing import (
    STRETCH_POLICY,
    checkpoint_hash,
    config_hash,
    level_policy_hash_source,
    model_config_hash,
    preprocessing_hash,
)
from sati.io import IngestError
from sati.manifests import read_metadata_sidecar

INFERENCE_CONFIG_NAME = "inference_config.yaml"


def _class_map_for(cfg: RootConfig) -> dict[str, str]:
    """Prepared root at ``data.source_path`` → its class map; else empty."""
    manifests_dir = cfg.data.source_path / "manifests"
    if manifests_dir.is_dir():
        return read_metadata_sidecar(manifests_dir).class_map
    return {}


def write_inference_config(cfg: RootConfig, checkpoint: Path, directory: Path) -> Path:
    sidecar = {
        "slicing": {
            "nominal_slice_size": cfg.slicing.nominal_slice_size,
            "overlap_ratio": cfg.slicing.overlap_ratio,
            "level_policy": level_policy_hash_source(cfg.slicing),
            "max_levels": cfg.slicing.max_levels,
            "stretch_policy": STRETCH_POLICY,
        },
        "recombination": cfg.recombination.model_dump(mode="json"),
        "model": {"arch": cfg.model.arch, "num_classes": cfg.model.num_classes,
                  "imgsz": cfg.model.imgsz},
        "class_map": _class_map_for(cfg),
        "hashes": {
            "checkpoint_hash": checkpoint_hash(checkpoint),
            "config_hash": config_hash(cfg),
            "model_config_hash": model_config_hash(cfg.model),
            "preprocessing_hash": preprocessing_hash(cfg.slicing),
        },
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / INFERENCE_CONFIG_NAME
    path.write_text(yaml.safe_dump(sidecar, sort_keys=False))
    return path


def execute_export(
    cfg: RootConfig, *, fmt: str = "onnx", backend: Backend | None = None
) -> tuple[Path, Path]:
    """Returns ``(artifact_path, inference_config_path)``."""
    weights = cfg.model.weights
    if not weights or not Path(weights).is_file():
        raise IngestError("export requires model.weights=<local checkpoint file>")
    backend = backend or UltralyticsBackend()
    artifact = backend.export(Path(weights), fmt=fmt, imgsz=cfg.model.imgsz)
    sidecar = write_inference_config(cfg, Path(weights), Path(cfg.output.run_dir))
    return artifact, sidecar
