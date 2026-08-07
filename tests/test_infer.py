"""Sliced inference engine and results rows (DESIGN-DOC-v1 §9, §4.5)."""

import cv2
import numpy as np
import pytest

from sati.backends import SliceDetection
from sati.config_loader import compose_and_resolve
from sati.hashing import recombination_hash
from sati.infer import execute_infer, infer_frame, rows_for_frame, run_hashes_for
from sati.io import IngestError
from sati.manifests import VALIDATION_RESULTS_SCHEMA, read_manifest
from sati.tiling import plan_frame


def _cfg(tmp_path, overrides=()):
    return compose_and_resolve(
        [
            f"data.source.uri={tmp_path}/unused.json",
            "model.num_classes=2",
            "slicing.nominal_slice_size=128",
            f"infer_outputs.dir={tmp_path}/run",
            *overrides,
        ],
        command="infer",
        base_dir=tmp_path,
    )


class EchoBackend:
    """Reports one fixed detection in every slice it is shown."""

    def __init__(self, box=(10.0, 10.0, 60.0, 60.0), score=0.9, class_id=1):
        self.det = SliceDetection(bbox_xyxy=box, score=score, class_id=class_id)
        self.loaded = None
        self.batches = []

    def train(self, data_yaml, cfg, *, resume_from=None):
        raise NotImplementedError

    def load(self, model_ref):
        self.loaded = model_ref

    def predict(self, slices, *, imgsz, confidence_threshold):
        self.batches.append(len(slices))
        return [[self.det] for _ in slices]


def test_infer_frame_full_pipeline(tmp_path):
    cfg = _cfg(tmp_path)
    pixels = np.zeros((200, 300, 3), np.uint8)
    backend = EchoBackend()
    result = infer_frame("f1", pixels, cfg, backend)
    plan = plan_frame((300, 200), cfg.slicing)
    assert len(result.slice_predictions) == len(plan.slices)  # one det per slice
    assert result.final_detections  # recombined output exists
    for det in result.final_detections:
        x1, y1, x2, y2 = det.bbox_fullres_xyxy
        assert 0 <= x1 < x2 <= 300 and 0 <= y1 < y2 <= 200  # full-res coords


def test_infer_frame_batching(tmp_path):
    cfg = _cfg(tmp_path, ["training.batch=2"])
    backend = EchoBackend()
    infer_frame("f1", np.zeros((200, 300, 3), np.uint8), cfg, backend)
    assert all(b <= 2 for b in backend.batches)
    assert sum(backend.batches) == len(plan_frame((300, 200), cfg.slicing).slices)


def test_rows_for_frame_record_types_and_provenance(tmp_path):
    cfg = _cfg(tmp_path)
    backend = EchoBackend()
    inference = infer_frame("f1", np.zeros((200, 300, 3), np.uint8), cfg, backend)
    hashes = run_hashes_for(cfg)
    rows = rows_for_frame(
        inference, cfg=cfg, hashes=hashes, dataset_split="validation", source_uri="x.png"
    )
    by_type = {}
    for row in rows:
        by_type.setdefault(row["record_type"], []).append(row)
    # slice_prediction off by default
    assert set(by_type) == {"frame_metadata", "level_merge", "final_merge"}
    (meta,) = by_type["frame_metadata"]
    assert meta["fullres_wh"] == [300, 200]
    for row in by_type["level_merge"]:
        assert row["recombination_hash"] == recombination_hash(cfg.recombination, "level_merge")
        # resolved from config now, not a module constant: `nms` happens to be the
        # default, so assert it tracks the config rather than passing by accident
        assert row["merge_strategy_name"] == cfg.recombination.method == "nms"
        assert row["merged_class_ids"] and row["merged_class_ids"][0] == row["class_id"]
    for row in by_type["final_merge"]:
        assert row["recombination_hash"] == recombination_hash(cfg.recombination, "final_merge")
        assert row["source_levels"]
        assert row["merged_class_ids"] and row["merged_class_ids"][0] == row["class_id"]
    # merge rows always carry the absorbed-class list; a frame_metadata row omits
    # the key entirely, which build_table turns into the typed null
    for row in by_type["frame_metadata"]:
        assert row.get("merged_class_ids") is None
    assert all(r["config_hash"] == hashes.config_hash for r in rows)
    assert all(r["run_id"] == cfg.runtime.run_id for r in rows)


def test_cache_slice_predictions_toggle(tmp_path):
    cfg = _cfg(tmp_path, ["results.cache_slice_predictions=true"])
    backend = EchoBackend()
    inference = infer_frame("f1", np.zeros((200, 300, 3), np.uint8), cfg, backend)
    rows = rows_for_frame(
        inference, cfg=cfg, hashes=run_hashes_for(cfg), dataset_split="holdout", source_uri="x"
    )
    kinds = {r["record_type"] for r in rows}
    assert "slice_prediction" in kinds
    n_sp = sum(1 for r in rows if r["record_type"] == "slice_prediction")
    assert n_sp == len(inference.slice_predictions)


def test_execute_infer_end_to_end(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for name in ("a", "b"):
        cv2.imwrite(str(imgs / f"{name}.png"), np.zeros((150, 250, 3), np.uint8))
    cfg = _cfg(tmp_path)
    backend = EchoBackend()
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"checkpoint")
    out = execute_infer(
        cfg,
        checkpoint=checkpoint,
        input_path=imgs,
        backend=backend,
        class_mapping={0: "a", 1: "b"},
    )
    assert backend.loaded == checkpoint
    table = read_manifest(out, VALIDATION_RESULTS_SCHEMA)
    frames = {r["fullframe_id"] for r in table.to_pylist()}
    assert frames == {"a", "b"}
    metas = [r for r in table.to_pylist() if r["record_type"] == "frame_metadata"]
    assert len(metas) == 2 and all(r["source_uri"].endswith(".png") for r in metas)
    # §4.5 sidecar exists and re-verifies
    from sati.manifests import read_metadata_sidecar, verify_metadata_hashes

    meta = read_metadata_sidecar(out.parent)
    assert all(verify_metadata_hashes(meta).values())


def test_execute_infer_requires_imagery(tmp_path):
    (tmp_path / "imgs").mkdir()
    cfg = _cfg(tmp_path)
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(IngestError, match="no input imagery"):
        execute_infer(
            cfg,
            checkpoint=checkpoint,
            input_path=tmp_path / "missing",
            backend=EchoBackend(),
        )
