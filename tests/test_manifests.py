"""Manifest schemas, column-by-column writers, sidecar (DESIGN-DOC-v1 §4)."""

import json

import pyarrow as pa
import pytest
from pydantic import ValidationError

from sati.config_schemas import SlicingConfig
from sati.hashing import (
    class_mapping_hash,
    class_mapping_hash_full_source,
    config_hash,
    config_hash_full_source,
    dataset_hash,
    dataset_hash_full_source,
    preprocessing_hash,
    preprocessing_hash_full_source,
)
from sati.manifests import (
    ANNOTATIONS_SCHEMA,
    FULLFRAMES_SCHEMA,
    SCHEMA_VERSION,
    SLICE_ANNOTATIONS_SCHEMA,
    SLICES_SCHEMA,
    VALIDATION_RESULTS_SCHEMA,
    HashWithSource,
    ManifestMetadata,
    build_table,
    read_manifest,
    read_metadata_sidecar,
    verify_metadata_hashes,
    write_manifest,
    write_metadata_sidecar,
)

FRAME_ROW = {
    "fullframe_id": "img-001",
    "source_uri": "frames/img-001.png",
    "dataset_split": "training",
    "fullres_wh": [4000, 3000],
    "color_mode": "RGB",
}


def test_build_table_column_by_column():
    table = build_table(FULLFRAMES_SCHEMA, [FRAME_ROW])
    assert table.schema == FULLFRAMES_SCHEMA
    assert table.column("fullres_wh").to_pylist() == [[4000, 3000]]


def test_missing_key_becomes_typed_null():
    row = {
        "record_type": "frame_metadata",
        "fullframe_id": "img-001",
        "dataset_split": "training",
        "run_id": "r1",
        "schema_version": SCHEMA_VERSION,
        # every detection column omitted
    }
    table = build_table(VALIDATION_RESULTS_SCHEMA, [row])
    assert table.column("score").to_pylist() == [None]
    assert table.column("bbox_fullres_xyxy").to_pylist() == [None]
    assert table.column("source_levels").to_pylist() == [None]


def test_unknown_keys_dropped():
    table = build_table(FULLFRAMES_SCHEMA, [{**FRAME_ROW, "bogus": 1}])
    assert "bogus" not in table.schema.names


def test_null_leading_list_column_survives():
    # the exact case Arrow inference breaks on: null-leading list column
    rows = [
        {
            "record_type": "frame_metadata",
            "fullframe_id": "f",
            "dataset_split": "training",
            "run_id": "r",
            "schema_version": 1,
        },
        {
            "record_type": "final_merge",
            "fullframe_id": "f",
            "dataset_split": "training",
            "run_id": "r",
            "schema_version": 1,
            "bbox_fullres_xyxy": [1.0, 2.0, 3.0, 4.0],
            "score": 0.9,
            "class_id": 0,
            "source_levels": [0, 1],
        },
    ]
    table = build_table(VALIDATION_RESULTS_SCHEMA, rows)
    assert table.column("source_levels").to_pylist() == [None, [0, 1]]
    # and the null-bearing table must survive a parquet write (fixed-size lists
    # with null entries don't — hence variable lists in this schema)
    import pyarrow.parquet as pq
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        pq.write_table(table, f"{d}/t.parquet")
        assert pq.read_table(f"{d}/t.parquet").equals(table)


def test_write_read_roundtrip_and_schema_guard(tmp_path):
    table = build_table(FULLFRAMES_SCHEMA, [FRAME_ROW])
    path = write_manifest(table, tmp_path / "manifests" / "fullframes.parquet")
    again = read_manifest(path, FULLFRAMES_SCHEMA)
    assert again.equals(table)
    with pytest.raises(ValueError, match="schema mismatch"):
        read_manifest(path, ANNOTATIONS_SCHEMA)


def test_all_schemas_are_explicit():
    for schema in (
        FULLFRAMES_SCHEMA,
        ANNOTATIONS_SCHEMA,
        SLICES_SCHEMA,
        SLICE_ANNOTATIONS_SCHEMA,
        VALIDATION_RESULTS_SCHEMA,
    ):
        assert isinstance(schema, pa.Schema) and len(schema) > 0


def test_sidecar_roundtrip_preserves_class_mapping_order(tmp_path):
    class_mapping = {"0": "ring", "1": "spiral", "10": "gear"}
    meta = ManifestMetadata(
        schema_version=SCHEMA_VERSION,
        class_mapping=class_mapping,
        hashes={
            "class_mapping_hash": HashWithSource(
                value=class_mapping_hash({0: "ring", 1: "spiral", 10: "gear"}),
                source=class_mapping_hash_full_source(
                    {0: "ring", 1: "spiral", 10: "gear"}
                ),
            )
        },
    )
    write_metadata_sidecar(meta, tmp_path)
    raw = json.loads((tmp_path / "_metadata.json").read_text())
    assert list(raw["class_mapping"].keys()) == ["0", "1", "10"]
    again = read_metadata_sidecar(tmp_path)
    assert list(again.class_mapping.keys()) == ["0", "1", "10"]


def test_sidecar_strict_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        ManifestMetadata(schema_version=1, class_mapping={}, hashes={}, extra_field="no")


def test_hash_reverification(tmp_path):
    from sati.config_loader import compose_and_resolve

    cfg = compose_and_resolve(
        ["data.source.uri=x.json", "model.num_classes=2"],
        command="infer",
        base_dir=tmp_path,
    )
    meta = ManifestMetadata(
        schema_version=SCHEMA_VERSION,
        class_mapping={"0": "a"},
        hashes={
            "config_hash": HashWithSource(
                value=config_hash(cfg), source=config_hash_full_source(cfg)
            ),
            "tampered": HashWithSource(
                value=config_hash(cfg), source={"recipe": "other", "x": 1}
            ),
        },
    )
    # survive a disk round-trip, then re-verify
    write_metadata_sidecar(meta, tmp_path)
    results = verify_metadata_hashes(read_metadata_sidecar(tmp_path))
    assert results == {"config_hash": True, "tampered": False}


def test_dataset_hash_identity_semantics():
    frames = [
        {
            "fullframe_id": "f",
            "source_uri": "/a/image.png",
            "dataset_split": "training",
            "fullres_wh": [100, 80],
            "color_mode": "RGB",
        }
    ]
    annotations = [
        {"fullframe_id": "f", "ann_id": 1, "class_id": 0, "bbox_xyxy": [1, 2, 3, 4]}
    ]
    assert dataset_hash(frames, annotations, {0: "a"}) == dataset_hash(
        list(reversed(frames)),
        list(reversed(annotations)),
        {0: "renamed"},
    )
    changed = [{**annotations[0], "bbox_xyxy": [1, 2, 4, 4]}]
    assert dataset_hash(frames, annotations, {0: "a"}) != dataset_hash(
        frames,
        changed,
        {0: "a"},
    )
    source = dataset_hash_full_source(frames, annotations, {0: "a"})
    assert source["fullframes"][0]["source_identity"] == "image.png"


def test_class_mapping_hash_order_sensitive():
    assert class_mapping_hash({0: "a", 1: "b"}) != class_mapping_hash({1: "b", 0: "a"})
    assert class_mapping_hash({0: "a", 1: "b"}) == class_mapping_hash({0: "a", 1: "b"})


def test_preprocessing_hash_semantics():
    slicing = SlicingConfig()
    stack = [(4000, 3000), (2000, 1500), (1000, 750)]
    assert preprocessing_hash(slicing, stack) == preprocessing_hash(slicing, list(stack))
    assert preprocessing_hash(slicing, stack) != preprocessing_hash(slicing, stack[:2])
    other = SlicingConfig(overlap_ratio=0.25)
    assert preprocessing_hash(slicing, stack) != preprocessing_hash(other, stack)
    src = preprocessing_hash_full_source(slicing, stack)
    assert src["image_format"] == "png"
    start_zoom = SlicingConfig(
        level_policy={
            "kind": "start_then_geometric",
            "start_scale": 0.5,
            "subsequent_downsample_factor": 3.0,
        }
    )
    assert preprocessing_hash(start_zoom, stack) != preprocessing_hash(slicing, stack)
    assert preprocessing_hash_full_source(start_zoom)["level_policy"] == {
        "kind": "start_then_geometric",
        "start_scale": 0.5,
        "subsequent_downsample_factor": 3.0,
        "stop_when_long_side_lte_nominal": True,
    }
