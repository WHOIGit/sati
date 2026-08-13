"""Portable checkpoint inference contracts and canonical finalization."""

from __future__ import annotations

import json
import os
import shutil
import warnings
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from sati.config_schemas import RootConfig
from sati.errors import PreprocessingOverrideRefused
from sati.hashing import (
    checkpoint_hash,
    class_mapping_hash,
    class_mapping_hash_full_source,
    config_hash,
    model_config_hash,
    model_config_hash_full_source,
    preprocessing_hash,
    preprocessing_hash_full_source,
    recombination_hash_full_source,
    target_schema_hash,
    target_schema_hash_full_source,
)

CONTRACT_KEY = "sati_inference_contract"
CHECKPOINT_METADATA_NAME = "checkpoint_metadata.json"


class InferenceContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    sati_version: str
    source_run_id: str
    config_hash: str
    model: dict[str, Any]
    slicing: dict[str, Any]
    recombination: dict[str, Any]
    class_mapping: dict[str, str]
    target_schema: dict[str, Any]
    hashes: dict[str, str]
    hash_sources: dict[str, Any]


def build_inference_contract(
    cfg: RootConfig,
    *,
    class_mapping: dict[int, str],
) -> InferenceContract:
    return InferenceContract(
        schema_version=2,
        sati_version="0.1.0",
        source_run_id=cfg.runtime.run_id,
        config_hash=config_hash(cfg),
        model={
            "architecture": cfg.model.architecture.model_dump(mode="json"),
            "num_classes": cfg.model.num_classes,
            "imgsz": cfg.model.imgsz,
        },
        slicing=cfg.slicing.model_dump(mode="json"),
        recombination=cfg.recombination.model_dump(mode="json"),
        class_mapping={str(index): name for index, name in class_mapping.items()},
        target_schema=cfg.task.model_dump(mode="json"),
        hashes={
            "target_schema_hash": target_schema_hash(cfg),
            "class_mapping_hash": class_mapping_hash(class_mapping),
            "model_config_hash": model_config_hash(cfg.model),
            "preprocessing_hash": preprocessing_hash(cfg.slicing),
        },
        hash_sources={
            "target_schema_source": target_schema_hash_full_source(cfg),
            "class_mapping_source": class_mapping_hash_full_source(class_mapping),
            "model_config_source": model_config_hash_full_source(cfg.model),
            "preprocessing_source": preprocessing_hash_full_source(cfg.slicing),
            "level_merge_recombination_source": recombination_hash_full_source(
                cfg.recombination,
                "level_merge",
            ),
            "final_merge_recombination_source": recombination_hash_full_source(
                cfg.recombination,
                "final_merge",
            ),
        },
    )


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise RuntimeError(
            "checkpoint finalization requires torch; install sati[ultralytics]"
        ) from error
    return torch


def embed_inference_contract(path: Path, contract: InferenceContract) -> None:
    torch = _torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Ultralytics checkpoint root must be a mapping")
    payload[CONTRACT_KEY] = contract.model_dump(mode="json")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def extract_inference_contract(path: Path | str) -> InferenceContract:
    torch = _torch()
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or CONTRACT_KEY not in payload:
        raise ValueError(
            "checkpoint lacks sati_inference_contract; produce a new canonical SATI checkpoint"
        )
    return InferenceContract.model_validate(payload[CONTRACT_KEY])


#: Slicing fields the checkpoint pins because they are model-coupled. The emitted
#: slice size derives ``model.imgsz``, so a divergence would cut tiles at one size
#: while telling the backend another; ``image_format`` has no infer-time effect at
#: all (it is read only when *writing* slice files during prepare) but is pinned so
#: that two runs over identical pixels cannot carry different effective hashes.
PINNED_SLICING_FIELDS = ("nominal_slice_size", "image_format")

#: Slicing fields that are pure runtime geometry: the pyramid is planned per frame
#: from these, emitted slices stay ``nominal_slice_size`` px regardless, and the
#: backend call is unchanged in shape. Overridable at infer/eval time; pinned on
#: export, where the portable config must describe the trained pyramid.
GEOMETRY_SLICING_FIELDS = ("overlap_ratio", "level_policy", "max_levels")


def apply_inference_contract(
    cfg: RootConfig,
    checkpoint: Path | str,
    *,
    pin_geometry: bool = False,
) -> InferenceContract:
    """Reconstruct checkpoint-pinned model/preprocessing fields.

    Only the model-coupled subset is restored from the contract. Slice geometry
    (:data:`GEOMETRY_SLICING_FIELDS`) is left at its resolved value so infer/eval
    can re-slice an existing checkpoint with a different pyramid; a divergence in
    :data:`PINNED_SLICING_FIELDS` raises instead.

    ``pin_geometry=True`` restores geometry from the contract as well, for
    ``export``, whose portable inference config must describe the trained pyramid
    (D7). Because the override is then inert, it warns rather than dropping it
    silently (D8).
    """
    from sati.config_schemas import ModelArchitectureConfig, ModelConfig, SlicingConfig
    from sati.hashing import config_hash as calculate_config_hash, config_id

    contract = extract_inference_contract(checkpoint)

    # Rebuild the contract's own view of itself *before* touching ``cfg``, so the
    # verifications below compare two independently stored things — the contract's
    # fields against the contract's digests. Hashing values just assigned to ``cfg``
    # from those same fields is a tautology that can never fail (D5).
    contract_slicing = SlicingConfig.model_validate(contract.slicing)
    contract_architecture = ModelArchitectureConfig.model_validate(
        contract.model["architecture"]
    )
    contract_model = ModelConfig(
        architecture=contract_architecture,
        num_classes=int(contract.model["num_classes"]),
        imgsz=int(contract.model["imgsz"]),
    )
    if model_config_hash(contract_model) != contract.hashes["model_config_hash"]:
        raise ValueError("checkpoint model_config_hash failed verification")
    if preprocessing_hash(contract_slicing) != contract.hashes["preprocessing_hash"]:
        raise ValueError("checkpoint preprocessing_hash failed verification")

    # ``cfg.task`` is never restored from the contract, so this stays a genuine
    # compatibility check on the run's own task rather than a self-comparison.
    if target_schema_hash(cfg) != contract.hashes["target_schema_hash"]:
        raise ValueError("checkpoint target_schema_hash is incompatible with this SATI task")

    for name in PINNED_SLICING_FIELDS:
        resolved, pinned = getattr(cfg.slicing, name), getattr(contract_slicing, name)
        if resolved != pinned:
            raise PreprocessingOverrideRefused(name, resolved, pinned)

    cfg.model.architecture = contract_architecture
    cfg.model.num_classes = contract_model.num_classes
    cfg.model.imgsz = contract_model.imgsz
    if pin_geometry:
        ignored = [
            name
            for name in GEOMETRY_SLICING_FIELDS
            if getattr(cfg.slicing, name) != getattr(contract_slicing, name)
        ]
        if ignored:
            warnings.warn(
                "export describes the trained pyramid: ignoring slice-geometry override "
                f"of {', '.join('slicing.' + name for name in ignored)}. The exported "
                "inference config records the checkpoint's own geometry.",
                stacklevel=2,
            )
        cfg.slicing = contract_slicing

    digest = calculate_config_hash(cfg)
    cfg.runtime.config_hash = digest
    cfg.runtime.config_id = config_id(digest)
    return contract


def _copy_or_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def finalize_checkpoint(
    source: Path,
    *,
    role: str,
    output_dir: Path,
    contract: InferenceContract,
) -> tuple[Path, str]:
    staging = output_dir / f".{role}.staging.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_or_link(source, staging)
    embed_inference_contract(staging, contract)
    digest = checkpoint_hash(staging)
    destination = output_dir / f"{role}.{digest.removeprefix('sha256:')[:6]}.pt"
    os.replace(staging, destination)
    return destination, digest


def write_checkpoint_metadata(
    directory: Path,
    *,
    contract: InferenceContract,
    checkpoints: dict[str, tuple[Path, str]],
) -> Path:
    path = directory / CHECKPOINT_METADATA_NAME
    payload = {
        "schema_version": 2,
        "contract": contract.model_dump(mode="json"),
        "checkpoints": {
            role: {"path": artifact.name, "checkpoint_hash": digest}
            for role, (artifact, digest) in checkpoints.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return path


def read_checkpoint_metadata(path: Path | str) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / CHECKPOINT_METADATA_NAME
    return json.loads(candidate.read_text())
