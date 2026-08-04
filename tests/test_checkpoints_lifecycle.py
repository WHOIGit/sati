"""Portable checkpoint and collision-safe lifecycle integration contracts."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from sati.checkpoints import (
    build_inference_contract,
    extract_inference_contract,
    finalize_checkpoint,
    read_checkpoint_metadata,
    write_checkpoint_metadata,
)
from sati.config_loader import resolve_configuration
from sati.hashing import checkpoint_hash
from sati.lifecycle import (
    LifecycleError,
    RunLifecycle,
    prepare_target,
    validate_destructive_target,
    write_config_bundle,
)
from sati.train import TrainRefusedError, canonical_last_checkpoint


def _resolution(tmp_path, *, command="train"):
    return resolve_configuration(
        [
            "data.source.uri=unused.json",
            f"data.prepared.uri={tmp_path}/prepared",
            "model.num_classes=2",
            f"training_outputs.dir={tmp_path}/run",
        ],
        command=command,
        base_dir=tmp_path,
    )


def test_checkpoint_contract_embed_hash_name_and_torch_free_metadata(tmp_path):
    torch = pytest.importorskip("torch")
    resolution = _resolution(tmp_path)
    cfg = resolution.config
    source = tmp_path / "native.pt"
    torch.save({"model": {"weight": torch.tensor([1.0])}}, source)
    contract = build_inference_contract(cfg, class_mapping={0: "ring", 1: "spiral"})
    artifact, digest = finalize_checkpoint(
        source,
        role="best",
        output_dir=tmp_path / "checkpoints",
        contract=contract,
    )
    assert artifact.name == f"best.{digest.removeprefix('sha256:')[:6]}.pt"
    assert checkpoint_hash(artifact) == digest
    assert extract_inference_contract(artifact) == contract
    metadata = write_checkpoint_metadata(
        artifact.parent,
        contract=contract,
        checkpoints={"best": (artifact, digest)},
    )
    assert read_checkpoint_metadata(metadata)["checkpoints"]["best"]["checkpoint_hash"] == digest


def test_collision_clobber_and_failed_manifest(tmp_path):
    with pytest.raises(LifecycleError, match="unsafe"):
        validate_destructive_target(tmp_path, cwd=tmp_path)
    target = tmp_path / "runs" / "one"
    target.mkdir(parents=True)
    (target / "existing").write_text("x")
    with pytest.raises(LifecycleError, match="not empty"):
        prepare_target(target)
    prepare_target(target, clobber=True)
    assert list(target.iterdir()) == []

    resolution = _resolution(tmp_path)
    root = prepare_target(tmp_path / "run")
    write_config_bundle(root, resolution, argv=["sati", "train"])
    lifecycle = RunLifecycle(root)
    lifecycle.start(resolution.config)
    lifecycle.fail(RuntimeError("boom"))
    manifest = json.loads((root / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["failure"] == {"type": "RuntimeError", "message": "boom"}
    assert (root / "config" / "resolved.yaml").is_file()


def test_resume_refuses_completed_lifecycle(tmp_path):
    resolution = _resolution(tmp_path)
    root = prepare_target(tmp_path / "run")
    write_config_bundle(root, resolution, argv=["sati", "train"])
    lifecycle = RunLifecycle(root)
    lifecycle.start(resolution.config)
    lifecycle.complete()
    with pytest.raises(LifecycleError, match="cannot resume"):
        RunLifecycle(root).resume(resolution.config)


def test_resume_refuses_mismatched_resolved_configuration(tmp_path):
    resolution = _resolution(tmp_path)
    root = prepare_target(tmp_path / "run")
    write_config_bundle(root, resolution, argv=["sati", "train"])
    RunLifecycle(root).start(resolution.config)
    changed = resolution.config.model_copy(deep=True)
    changed.runtime.config_hash = "sha256:" + "0" * 64
    with pytest.raises(LifecycleError, match="config_hash"):
        RunLifecycle(root).resume(changed)


def test_resume_requires_existing_canonical_last_checkpoint(tmp_path):
    cfg = _resolution(tmp_path).config
    checkpoint_dir = cfg.training_outputs.checkpoints.dir
    checkpoint_dir.mkdir(parents=True)
    metadata = checkpoint_dir / "checkpoint_metadata.json"
    metadata.write_text(
        json.dumps({"checkpoints": {"last": {"path": "last.abc123.pt"}}})
    )
    with pytest.raises(TrainRefusedError, match="missing"):
        canonical_last_checkpoint(cfg)
    checkpoint = checkpoint_dir / "last.abc123.pt"
    checkpoint.write_bytes(b"last")
    assert canonical_last_checkpoint(cfg) == checkpoint


@pytest.mark.skipif(
    not os.environ.get("SATI_RUN_TRIPWIRE"),
    reason="set SATI_RUN_TRIPWIRE=1 for real Ultralytics checkpoint round trip",
)
def test_real_ultralytics_finalized_checkpoint_round_trip(tmp_path):
    from ultralytics import YOLO

    resolution = resolve_configuration(
        [
            "data.source.uri=unused.json",
            "data.prepared.uri=unused-prepared",
            "model.num_classes=80",
            "model.imgsz=640",
            f"training_outputs.dir={tmp_path}/run",
        ],
        command="train",
        base_dir=tmp_path,
    )
    model = YOLO("yolo26n.pt")
    source = __import__("pathlib").Path(model.ckpt_path)
    contract = build_inference_contract(
        resolution.config,
        class_mapping={index: str(index) for index in range(80)},
    )
    artifact, digest = finalize_checkpoint(
        source,
        role="best",
        output_dir=tmp_path / "checkpoints",
        contract=contract,
    )
    reloaded = YOLO(str(artifact))
    reloaded.predict(np.zeros((640, 640, 3), np.uint8), imgsz=640, verbose=False)
    assert checkpoint_hash(artifact) == digest
