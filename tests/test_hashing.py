"""Golden hash-source, canonical JSON, ID, and template contracts."""

from __future__ import annotations

import json
import math
from datetime import datetime

import pytest

from sati.config_loader import compose_and_resolve, render_template
from sati.hashing import (
    canonical_json,
    class_mapping_hash,
    class_mapping_hash_full_source,
    checkpoint_hash,
    config_hash,
    config_hash_source,
    config_id,
    dataset_hash,
    dataset_hash_full_source,
    deterministic_id,
    model_config_hash,
    model_config_hash_full_source,
    model_hash,
    prepared_dataset_hash,
    prepared_dataset_hash_full_source,
    prepared_dataset_id,
    preprocessing_hash,
    preprocessing_hash_full_source,
    recombination_hash,
    recombination_hash_full_source,
    target_schema_hash,
    target_schema_hash_full_source,
)

REQUIRED = [
    "data.source.uri=datasets/demo/annotations.json",
    "model.num_classes=3",
]


def _cfg(tmp_path, overrides=()):
    return compose_and_resolve(
        [*REQUIRED, *overrides],
        command="infer",
        base_dir=tmp_path,
    )


def test_canonical_json_shape_and_floats():
    assert canonical_json({"b": [1, 2.5, None], "a": {"y": True, "x": "s"}}) == (
        '{"a":{"x":"s","y":true},"b":[1,2.5,null]}'
    )
    assert canonical_json(1.0) == "1"
    assert canonical_json(0.1) == "0.1"
    assert canonical_json(float("nan")) == '"NaN"'
    assert canonical_json(math.inf) == '"Infinity"'
    assert canonical_json(-math.inf) == '"-Infinity"'


def test_config_hash_is_portable_command_aware_and_output_independent(tmp_path):
    a = _cfg(tmp_path / "a")
    b = _cfg(
        tmp_path / "b",
        ["infer_outputs.dir=elsewhere"],
    )
    assert config_hash(a) == config_hash(b)
    source = config_hash_source(a)
    assert "data" not in source["config"]
    prepare_source = config_hash_source(a, "prepare")
    assert prepare_source["config"]["data"]["source"]["uri"] == "<path>"
    assert "run_id" not in source["config"]["runtime"]
    assert "infer_outputs" not in source["config"]
    changed = _cfg(tmp_path, ["recombination.cross_level_iou=0.7"])
    assert config_hash(a) != config_hash(changed)
    assert config_hash(a, "prepare") == config_hash(changed, "prepare")


def test_model_hash_excludes_initialization_but_config_hash_includes_it(tmp_path):
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    a = _cfg(
        tmp_path,
        ["model.weights.source=checkpoint", f"model.weights.uri={first}"],
    )
    b = _cfg(
        tmp_path,
        ["model.weights.source=checkpoint", f"model.weights.uri={second}"],
    )
    assert model_config_hash(a.model) == model_config_hash(b.model)
    assert config_hash(a, "train") != config_hash(b, "train")


def test_identity_sources_match_golden_fixture(tmp_path):
    cfg = _cfg(tmp_path)
    fullframes = [
        {
            "fullframe_id": "f1",
            "source_uri": "/elsewhere/image.png",
            "dataset_split": "training",
            "fullres_wh": [100, 80],
            "color_mode": "RGB",
        }
    ]
    annotations = [
        {
            "fullframe_id": "f1",
            "ann_id": 7,
            "class_id": 0,
            "bbox_xyxy": [1.0, 2.0, 10.0, 20.0],
        }
    ]
    classes = {0: "ring", 1: "spiral"}
    logical = dataset_hash(fullframes, annotations, classes)
    sources = {
        "config": config_hash_source(cfg),
        "dataset": dataset_hash_full_source(fullframes, annotations, classes),
        "prepared_dataset": prepared_dataset_hash_full_source(logical, cfg.slicing),
        "target_schema": target_schema_hash_full_source(cfg),
        "class_mapping": class_mapping_hash_full_source(classes),
        "model": model_config_hash_full_source(cfg.model),
        "preprocessing": preprocessing_hash_full_source(cfg.slicing),
        "level_recombination": recombination_hash_full_source(
            cfg.recombination,
            "level_merge",
        ),
        "final_recombination": recombination_hash_full_source(
            cfg.recombination,
            "final_merge",
        ),
    }
    fixture = json.loads(
        (
            __import__("pathlib").Path(__file__).parent
            / "fixtures"
            / "hash_sources.json"
        ).read_text()
    )
    assert sources == fixture["sources"]
    assert {
        name: canonical_json(source) for name, source in sources.items()
    } == fixture["canonical_json"]
    assert {
        "config": config_hash(cfg),
        "dataset": logical,
        "prepared_dataset": prepared_dataset_hash(logical, cfg.slicing),
        "target_schema": target_schema_hash(cfg),
        "class_mapping": class_mapping_hash(classes),
        "model": model_config_hash(cfg.model),
        "preprocessing": preprocessing_hash(cfg.slicing),
        "level_recombination": recombination_hash(cfg.recombination, "level_merge"),
        "final_recombination": recombination_hash(cfg.recombination, "final_merge"),
    } == fixture["hashes"]


def test_checkpoint_and_exported_model_byte_hash_golden_values(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    exported = tmp_path / "model.onnx"
    checkpoint.write_bytes(b"checkpoint-bytes")
    exported.write_bytes(b"exported-model-bytes")
    assert checkpoint_hash(checkpoint) == (
        "sha256:68ff0e8c172a352fc955978d2357b69ca"
        "36d9836f5a251bdcafe408960449e8d"
    )
    assert model_hash(exported) == (
        "sha256:32e7460aaf83418c89763b51b717d3c"
        "96c174fa2741f5fcc7230df246581cb08"
    )


def test_human_ids_and_template_tokens_are_pinned(tmp_path):
    digest = "sha256:" + "a" * 64
    assert config_id(digest) == deterministic_id("config_hash", digest)
    assert prepared_dataset_id(digest) == deterministic_id(
        "prepared_dataset_hash",
        digest,
    )
    assert config_id(digest) != prepared_dataset_id(digest)
    cfg = _cfg(tmp_path)
    values = cfg.model_dump(mode="json", exclude_none=True)
    rendered = render_template(
        "{experiment.name}/{timestamp}_{runtime.config_id}",
        values,
        timestamp=datetime(2026, 7, 29, 12, 34, 56),
    )
    assert rendered == f"starter/2026-07-29_12-34-56_{cfg.runtime.config_id}"
    first = _cfg(tmp_path)
    second = _cfg(tmp_path)
    assert first.runtime.run_id != second.runtime.run_id
    with pytest.raises(ValueError, match="unresolved output template"):
        render_template("{not.closed", values)
