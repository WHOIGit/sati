"""Canonical model export and portable inference sidecar."""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from sati.backends import Backend, UltralyticsBackend
from sati.checkpoints import InferenceContract, extract_inference_contract
from sati.config_schemas import RootConfig
from sati.hashing import checkpoint_hash, model_hash
from sati.io import IngestError

INFERENCE_CONFIG_NAME = "inference_config.yaml"


def write_inference_config(
    contract: InferenceContract,
    *,
    checkpoint: Path,
    artifact: Path,
    fmt: str,
    directory: Path,
) -> Path:
    sidecar = {
        **contract.model_dump(mode="json"),
        "export": {
            "format": fmt,
            "artifact": artifact.name,
            "checkpoint_hash": checkpoint_hash(checkpoint),
            "model_hash": model_hash(artifact),
        },
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / INFERENCE_CONFIG_NAME
    path.write_text(yaml.safe_dump(sidecar, sort_keys=False))
    return path


def execute_export(
    cfg: RootConfig,
    *,
    checkpoint: Path,
    fmt: str = "onnx",
    backend: Backend | None = None,
    contract: InferenceContract | None = None,
) -> tuple[Path, Path]:
    if not checkpoint.is_file():
        raise IngestError(f"export checkpoint does not exist: {checkpoint}")
    if cfg.export_outputs.models.dir is None:
        raise IngestError("export_outputs.models.dir was not resolved")
    contract = contract or extract_inference_contract(checkpoint)
    backend = backend or UltralyticsBackend()
    native_artifact = backend.export(checkpoint, fmt=fmt, imgsz=cfg.model.imgsz)
    cfg.export_outputs.models.dir.mkdir(parents=True, exist_ok=True)
    artifact = cfg.export_outputs.models.dir / native_artifact.name
    if native_artifact.resolve() != artifact.resolve():
        shutil.move(str(native_artifact), artifact)
    sidecar = write_inference_config(
        contract,
        checkpoint=checkpoint,
        artifact=artifact,
        fmt=fmt,
        directory=cfg.export_outputs.models.dir,
    )
    return artifact, sidecar
