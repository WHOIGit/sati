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
    second = _resolution(tmp_path, ["recombination.cross_level_threshold=0.7"])
    summary = summarize_config(first.config)
    assert summary["valid"] is True
    assert summary["config_id"] == first.config.runtime.config_id
    comparison = summarize_config_compare(first.config, second.config)
    assert "recombination.cross_level_threshold" in comparison["config_hash"]
    assert comparison["preprocessing_hash"] == {}
    # Case A: a behavioral diff belongs to its hash group only, never `other`.
    assert "recombination.cross_level_threshold" not in comparison["other"]
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
    assert dataset_summary["zero_annotation_frames"] == 0
    assert dataset_summary["image_width"] == {"count": 1, "min": 100, "max": 100, "median": 100}
    assert dataset_summary["box_long_side"]["max"] == 20.0

    slice_summary = summarize_slices(root)
    assert slice_summary["per_level"]
    assert set(slice_summary["stretched_axes"]) == {"x", "y", "both", "none"}
    assert set(slice_summary["dropped_targets"]) == {"size_gate", "min_area_ratio"}
    assert slice_summary["yolo_tree_bytes"] > 0

    # A small result table exercises the run-id / hash-family inventory (#12).
    from sati.manifests import (
        SCHEMA_VERSION,
        VALIDATION_RESULTS_SCHEMA,
        ManifestMetadata,
        build_table,
        write_manifest,
        write_metadata_sidecar,
    )

    results_dir = tmp_path / "run" / "results"
    write_manifest(
        build_table(
            VALIDATION_RESULTS_SCHEMA,
            [
                {
                    "record_type": "final_merge",
                    "fullframe_id": "f",
                    "dataset_split": "validation",
                    "run_id": "rid",
                    "schema_version": SCHEMA_VERSION,
                    "bbox_fullres_xyxy": [1.0, 2.0, 3.0, 4.0],
                    "score": 0.9,
                    "class_id": 0,
                    "config_hash": "sha256:" + "c" * 64,
                    "checkpoint_hash": "sha256:" + "k" * 64,
                }
            ],
        ),
        results_dir / "validation_results.parquet",
    )
    write_metadata_sidecar(
        ManifestMetadata(schema_version=SCHEMA_VERSION, class_mapping={"0": "ring"}, hashes={}),
        results_dir,
    )
    result_summary = summarize_results(results_dir)
    assert result_summary["run_ids"] == ["rid"]
    assert result_summary["hashes_present"]["config_hash"] == ["sha256:" + "c" * 64]
    assert "checkpoint_hash" in result_summary["hashes_present"]

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
