"""Canonical JSON and config-hash guarantees (DESIGN-DOC-v1 §4.7)."""

import math

from sati.config_loader import compose_and_resolve
from sati.hashing import canonical_json, config_hash, config_hash_source, model_config_hash

REQUIRED = [
    "data.source_path=datasets/demo/annotations.json",
    "model.num_classes=3",
]


def _resolve(overrides=(), **kwargs):
    return compose_and_resolve([*REQUIRED, *overrides], write=False, **kwargs)


def test_canonical_json_shape():
    obj = {"b": [1, 2.5, None], "a": {"y": True, "x": "s"}}
    assert canonical_json(obj) == '{"a":{"x":"s","y":true},"b":[1,2.5,null]}'


def test_canonical_json_floats():
    assert canonical_json(0.1) == "0.1"
    assert canonical_json(1.0) == "1"
    assert canonical_json(float("nan")) == '"NaN"'
    assert canonical_json(float("inf")) == '"Infinity"'
    assert canonical_json(-math.inf) == '"-Infinity"'


def test_config_hash_stable_across_roots(tmp_path):
    # Settled decision (v1 §4.7): same authored config resolved under two
    # different roots must hash identically — no absolute path reaches the hash.
    cfg_a = _resolve(base_dir=tmp_path / "rootA")
    cfg_b = _resolve(base_dir=tmp_path / "rootB")
    assert cfg_a.data.source_path != cfg_b.data.source_path
    assert config_hash(cfg_a) == config_hash(cfg_b)


def test_config_hash_ignores_run_id(tmp_path):
    a = _resolve(base_dir=tmp_path, run_id="run-one")
    b = _resolve(base_dir=tmp_path, run_id="run-two")
    assert config_hash(a) == config_hash(b)


def test_config_hash_sensitive_to_behavior(tmp_path):
    base = _resolve(base_dir=tmp_path)
    changed = _resolve(["slicing.overlap_ratio=0.3"], base_dir=tmp_path)
    assert config_hash(base) != config_hash(changed)


def test_hash_source_has_no_paths_or_run_id(tmp_path):
    source = config_hash_source(_resolve(base_dir=tmp_path))
    assert source["data"]["source_path"] == "<path>"
    assert source["output"]["run_dir"] == "<path>"
    assert source["data"]["split_file"] is None  # absent optional path stays None
    assert "run_id" not in source


def test_model_config_hash(tmp_path):
    cfg = _resolve(base_dir=tmp_path)
    assert model_config_hash(cfg.model).startswith("sha256:")
    other = _resolve(["model.num_classes=5"], base_dir=tmp_path)
    assert model_config_hash(cfg.model) != model_config_hash(other.model)


def test_file_backed_weights_hash_by_content(tmp_path):
    # Settled (2026-07-02): weights pointing at a real file contribute their
    # byte hash, so config_hash is location- and machine-independent.
    ckpt_a = tmp_path / "a" / "best.pt"
    ckpt_b = tmp_path / "b" / "best.pt"
    for ckpt in (ckpt_a, ckpt_b):
        ckpt.parent.mkdir()
        ckpt.write_bytes(b"identical fake weights")
    ha = config_hash(_resolve([f"model.weights={ckpt_a}"], base_dir=tmp_path))
    hb = config_hash(_resolve([f"model.weights={ckpt_b}"], base_dir=tmp_path))
    assert ha == hb

    ckpt_b.write_bytes(b"different fake weights")
    hb2 = config_hash(_resolve([f"model.weights={ckpt_b}"], base_dir=tmp_path))
    assert hb2 != ha


def test_hub_asset_weights_hash_verbatim(tmp_path, monkeypatch):
    # isolate CWD: a hub name that happens to match a local file hashes by
    # content (matching what ultralytics would actually load)
    monkeypatch.chdir(tmp_path)
    a = _resolve(["model.weights=yolo26n.pt"], base_dir=tmp_path)
    b = _resolve(["model.weights=yolo26s.pt"], base_dir=tmp_path)
    assert config_hash(a) != config_hash(b)
    source = config_hash_source(a)
    assert source["model"]["weights"] == "yolo26n.pt"
