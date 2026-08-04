"""Read-only canonical config/artifact inspection."""

from __future__ import annotations

import json

from sati.config_loader import resolve_configuration
from sati.inspect import (
    summarize_config,
    summarize_config_compare,
    summarize_dataset,
    summarize_model,
    summarize_results,
    summarize_slices,
)
from sati.lifecycle import prepare_target, write_config_bundle


def _resolution(tmp_path, overrides=()):
    return resolve_configuration(
        [
            "data.source.uri=unused.json",
            "model.num_classes=2",
            f"infer_outputs.dir={tmp_path}/run",
            *overrides,
        ],
        command="infer",
        base_dir=tmp_path,
    )


def test_config_summary_and_compare_are_read_only(tmp_path):
    first = _resolution(tmp_path)
    second = _resolution(tmp_path, ["recombination.cross_level_iou=0.7"])
    summary = summarize_config(first.config)
    assert summary["valid"] is True
    assert summary["config_id"] == first.config.runtime.config_id
    comparison = summarize_config_compare(first.config, second.config)
    assert "recombination.cross_level_iou" in comparison["config_hash"]
    assert comparison["preprocessing_hash"] == {}
    # Case A: a behavioral diff belongs to its hash group only, never `other`.
    assert "recombination.cross_level_iou" not in comparison["other"]
    assert not (tmp_path / "run").exists()


def test_config_compare_path_only_is_residual(tmp_path):
    first = _resolution(tmp_path)
    second = _resolution(tmp_path, [f"infer_outputs.dir={tmp_path}/elsewhere"])
    comparison = summarize_config_compare(first.config, second.config)
    # A portable-path change moves no behavioral hash: every hash group empty,
    # the difference is reported only under `other`.
    for group in ("config_hash", "target_schema_hash", "model_config_hash", "preprocessing_hash"):
        assert comparison[group] == {}
    assert "infer_outputs.dir" in comparison["other"]


def test_config_bundle_can_be_inspected_without_mutation(tmp_path):
    resolution = _resolution(tmp_path)
    root = prepare_target(tmp_path / "run")
    resolved = write_config_bundle(root, resolution, argv=["sati", "infer"])
    before = resolved.read_bytes()
    summary = summarize_config(resolution.config)
    assert summary["hashes"]["config_hash"] == resolution.config.runtime.config_hash
    assert resolved.read_bytes() == before


def test_dataset_slice_and_result_summaries(tmp_path):
    import cv2
    import numpy as np

    from sati.config_loader import compose_and_resolve
    from sati.io import IngestedAnnotation, IngestedDataset, IngestedFrame
    from sati.materialize import materialize_dataset

    image = tmp_path / "image.png"
    cv2.imwrite(str(image), np.zeros((80, 100, 3), np.uint8))
    dataset = IngestedDataset(
        frames=(
            IngestedFrame("f", str(image), 100, 80, "RGB", "training"),
        ),
        annotations=(
            IngestedAnnotation("f", 1, 0, "ring", (10, 10, 30, 30)),
        ),
        class_map={0: "ring"},
    )
    cfg = compose_and_resolve(
        [
            "data.source.uri=unused.json",
            "model.num_classes=1",
            "slicing.nominal_slice_size=64",
        ],
        command="infer",
        base_dir=tmp_path,
    )
    root = materialize_dataset(dataset, cfg, tmp_path / "prepared")
    dataset_summary = summarize_dataset(root)
    assert dataset_summary["n_frames"] == 1
    assert dataset_summary["hash_verification"]
    assert summarize_slices(root)["per_level"]

    # Result discovery and lifecycle/metrics sidebands are covered by infer/eval
    # tests; here the helper's missing-path failure remains explicit.
    try:
        summarize_results(tmp_path / "missing")
    except ValueError as error:
        assert "no canonical result table" in str(error)


def test_model_inspection_uses_torch_free_metadata(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "best.abcdef.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"bytes")
    metadata = {
        "schema_version": 1,
        "contract": {"schema_version": 1},
        "checkpoints": {
            "best": {
                "path": checkpoint.name,
                "checkpoint_hash": "sha256:" + "a" * 64,
            }
        },
    }
    (checkpoint.parent / "checkpoint_metadata.json").write_text(json.dumps(metadata))
    summary = summarize_model(checkpoint)
    assert summary["checkpoint_metadata"] == metadata
    assert summary["checkpoint_hash"].startswith("sha256:")
