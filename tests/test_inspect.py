"""Read-only inspect summaries (DESIGN-DOC-v1 §5.1)."""

import json

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from sati.cli import app
from sati.config_loader import compose_and_resolve, write_resolved
from sati.hashing import config_hash
from sati.infer import execute_infer
from sati.inspect import (
    summarize_config,
    summarize_dataset,
    summarize_model,
    summarize_results,
    summarize_slices,
)
from sati.io import ingest_coco
from sati.materialize import materialize_dataset


def _cfg(tmp_path, overrides=()):
    return compose_and_resolve(
        [
            "data.source_path=anns.json",
            "model.num_classes=2",
            "slicing.nominal_slice_size=128",
            "slicing.size_gate_px=4",
            f"output.run_dir={tmp_path}/run",
            *overrides,
        ],
        write=False,
        base_dir=tmp_path,
    )


@pytest.fixture
def prepared(tmp_path):
    img = np.zeros((200, 300, 3), np.uint8)
    img[40:90, 50:120] = 200
    cv2.imwrite(str(tmp_path / "img1.png"), img)
    cv2.imwrite(str(tmp_path / "img2.png"), img)
    (tmp_path / "anns.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "img1.png", "width": 300, "height": 200},
                    {"id": 2, "file_name": "img2.png", "width": 300, "height": 200},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [50, 40, 70, 50]}
                ],
                "categories": [{"id": 1, "name": "ring"}, {"id": 2, "name": "spiral"}],
            }
        )
    )
    cfg = _cfg(tmp_path)
    root = materialize_dataset(ingest_coco(tmp_path / "anns.json"), cfg, tmp_path / "run")
    return tmp_path, cfg, root


def test_summarize_config_hash_preview_matches_hashing(tmp_path):
    cfg = _cfg(tmp_path)
    write_resolved(cfg)
    summary = summarize_config(tmp_path / "run")  # run_dir form → resolved.yaml
    assert summary["valid"] is True
    assert summary["hashes"]["config_hash"] == config_hash(cfg)  # one code path
    assert summary["derived"]["model.imgsz"] == 128


def test_summarize_config_invalid_reports_not_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("data:\n  source_format: coco\n  bogus: 1\n")
    summary = summarize_config(bad)
    assert summary["valid"] is False and "bogus" in summary["errors"]


def test_summarize_config_diff(tmp_path):
    cfg_a = _cfg(tmp_path)
    cfg_b = _cfg(tmp_path, ["training.epochs=7"])
    write_resolved(cfg_a)
    b_path = tmp_path / "b.yaml"
    import yaml

    b_path.write_text(yaml.safe_dump(cfg_b.model_dump(mode="json")))
    summary = summarize_config(tmp_path / "run", diff=b_path)
    assert "training.epochs" in summary["diff"]
    assert summary["diff"]["training.epochs"]["other"] == 7
    assert "slicing.overlap_ratio" not in summary["diff"]


def test_summarize_dataset(prepared):
    _, _, root = prepared
    summary = summarize_dataset(root)
    assert summary["n_frames"] == 2 and summary["n_annotations"] == 1
    assert summary["per_split"] == {"training": 2}
    assert summary["per_class_counts"] == {"ring": 1}
    assert summary["zero_annotation_frames"] == 1
    assert summary["box_long_side"]["max"] == 70.0


def test_summarize_slices(prepared):
    _, _, root = prepared
    summary = summarize_slices(root)
    assert summary["per_level"]  # at least level 0
    assert summary["spot_check"]["missing"] == []
    assert summary["yolo_tree_bytes"] > 0
    assert summary["drop_tallies"]["dropped_by_min_area_ratio"] >= 0
    total_slices = sum(v["slices"] for v in summary["per_level"].values())
    assert summary["stretched_axis_frequency"]["none"] <= total_slices


def test_summarize_results_and_verification(prepared, tmp_path):
    _, cfg, _ = prepared
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    cv2.imwrite(str(imgs / "z.png"), np.zeros((150, 250, 3), np.uint8))

    from sati.backends import SliceDetection

    class EchoBackend:
        def train(self, data_yaml, cfg):
            raise NotImplementedError

        def load(self, model_ref):
            pass

        def predict(self, slices, *, imgsz, confidence_threshold):
            det = SliceDetection(bbox_xyxy=(10.0, 10.0, 60.0, 60.0), score=0.9, class_id=1)
            return [[det] for _ in slices]

    infer_cfg = _cfg(
        tmp_path, [f"data.source_path={imgs}", "model.weights=fake.pt",
                   f"output.run_dir={tmp_path}/infer_run"]
    )
    results = execute_infer(infer_cfg, backend=EchoBackend())
    summary = summarize_results(results)
    assert summary["record_types"]["frame_metadata"] == 1
    assert summary["run_ids"] == [infer_cfg.run_id]
    assert all(summary["sidecar"]["hash_verification"].values())
    assert summary["hash_families"]["config_hash"] == [config_hash(infer_cfg)]


def test_summarize_model_best_effort(tmp_path):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"not a real torch file")
    summary = summarize_model(ckpt)
    assert summary["checkpoint_hash"].startswith("sha256:")
    assert summary["ultralytics"] is None  # unreadable → note, not crash
    assert "metadata_note" in summary
    (tmp_path / "inference_config.yaml").write_text("slicing:\n  nominal_slice_size: 64\n")
    summary = summarize_model(ckpt)
    assert summary["inference_config"]["slicing"]["nominal_slice_size"] == 64


def test_inspect_cli_json(prepared):
    tmp_path, _, root = prepared
    result = CliRunner().invoke(app, ["inspect", "dataset", str(root), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["n_frames"] == 2
