"""Eval: scoring, per-level diagnostic, end-to-end oracle round trip (v1 §11)."""

import json

import cv2
import numpy as np
import pytest

from sati.backends import SliceDetection
from sati.config_loader import compose_and_resolve
from sati.evaluation import (
    coco_map_scores,
    execute_eval,
    gt_recall_by_size,
    per_level_diagnostic,
)
from sati.io import IngestError, ingest_coco
from sati.materialize import materialize_dataset
from sati.tiling import plan_frame, project_box


def _cfg(tmp_path, overrides=()):
    return compose_and_resolve(
        [
            "data.source.uri=unused.json",
            f"data.prepared.uri={tmp_path}/prepared",
            "model.num_classes=2",
            "slicing.nominal_slice_size=128",
            "slicing.size_gate_px=4",
            "slicing.min_area_ratio=0.2",
            f"eval_outputs.dir={tmp_path}/run",
            *overrides,
        ],
        command="eval",
        base_dir=tmp_path,
    )


def _prepare_tiny_dataset(tmp_path, cfg):
    """Two frames (one training, one validation), one GT box each."""
    for i, (w, h) in enumerate(((300, 200), (260, 180)), start=1):
        img = np.zeros((h, w, 3), np.uint8)
        img[40:90, 50:120] = 200
        cv2.imwrite(str(tmp_path / f"img{i}.png"), img)
    (tmp_path / "anns.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "img1.png", "width": 300, "height": 200},
                    {"id": 2, "file_name": "img2.png", "width": 260, "height": 180},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [50, 40, 70, 50]},
                    {"id": 2, "image_id": 2, "category_id": 1, "bbox": [50, 40, 70, 50]},
                ],
                "categories": [{"id": 1, "name": "ring"}, {"id": 2, "name": "spiral"}],
            }
        )
    )
    ds = ingest_coco(tmp_path / "anns.json")
    from sati.io import apply_split_override

    frames = apply_split_override(ds.frames, {"1": "training", "2": "validation"})
    ds = type(ds)(frames=frames, annotations=ds.annotations, class_map=ds.class_map)
    return materialize_dataset(ds, cfg, tmp_path / "prepared")


class OracleBackend:
    """Answers each slice with the true projection of the frame's GT boxes.

    Detections are queued per slice in plan order by ``prime()``; predict pops
    them chunk by chunk (matching infer_frame's traversal).
    """

    def __init__(self):
        self.queue = []

    def prime(self, fullres_wh, gt_boxes, cfg):
        plan = plan_frame(fullres_wh, cfg.slicing)
        for spec in plan.slices:
            level = plan.level(spec.level)
            dets = []
            for class_id, bbox in gt_boxes:
                proj = project_box(bbox, spec, level, fullres_wh, cfg.slicing)
                if proj is not None and proj.is_training_target:
                    dets.append(
                        SliceDetection(bbox_xyxy=proj.bbox_slice_xyxy, score=0.9, class_id=class_id)
                    )
            self.queue.append(dets)

    def train(self, data_yaml, cfg, *, resume_from=None):
        raise NotImplementedError

    def load(self, model_ref):
        self.loaded = model_ref

    def predict(self, slices, *, imgsz, confidence_threshold):
        out, self.queue = self.queue[: len(slices)], self.queue[len(slices) :]
        assert len(out) == len(slices)
        return out


class SilentBackend(OracleBackend):
    def predict(self, slices, *, imgsz, confidence_threshold):
        return [[] for _ in slices]


def test_execute_eval_oracle_scores_high(tmp_path):
    cfg = _cfg(tmp_path)
    _prepare_tiny_dataset(tmp_path, cfg)
    backend = OracleBackend()
    backend.prime((260, 180), [(0, (50.0, 40.0, 120.0, 90.0))], cfg)
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"checkpoint")
    results, metrics = execute_eval(cfg, checkpoint=checkpoint, backend=backend)
    assert results.is_file()
    assert metrics["n_frames"] == 1  # validation split only
    assert metrics["mAP50"] > 0.9  # oracle predictions round-trip to GT
    assert metrics["per_class_ap50_95"]["ring"] > 0.5
    assert metrics["per_class_ap50_95"]["spiral"] == 0.0  # no GT, no dets
    assert metrics["per_level"]  # diagnostic populated
    bin_32_64 = metrics["gt_recall_by_longside"]["64-128"]
    assert bin_32_64 == {"n_gt": 1, "recall50": 1.0}
    assert (tmp_path / "run" / "metrics" / "metrics.json").is_file()
    # results sidecar carries the prepared class map
    from sati.manifests import read_metadata_sidecar

    meta = read_metadata_sidecar(tmp_path / "run" / "results")
    assert meta.class_mapping == {"0": "ring", "1": "spiral"}


def test_execute_eval_no_detections(tmp_path):
    cfg = _cfg(tmp_path)
    _prepare_tiny_dataset(tmp_path, cfg)
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"checkpoint")
    _, metrics = execute_eval(cfg, checkpoint=checkpoint, backend=SilentBackend())
    assert metrics["mAP50"] == 0.0
    assert metrics["n_final_merge"] == 0
    assert metrics["gt_recall_by_longside"]["64-128"]["recall50"] == 0.0


def test_execute_eval_guards(tmp_path):
    cfg = _cfg(tmp_path)
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(IngestError, match="no prepared dataset"):
        execute_eval(cfg, checkpoint=checkpoint, backend=OracleBackend())


def test_per_level_diagnostic_shape():
    rows = [
        {"record_type": "final_merge", "source_levels": [0, 1], "score": 0.8},
        {"record_type": "final_merge", "source_levels": [0], "score": 0.6},
    ]
    diag = per_level_diagnostic(rows)
    assert diag["0"]["final_merge_count"] == 2
    assert diag["1"]["final_merge_count"] == 1
    assert diag["0"]["confidence_mean"] == pytest.approx(0.7)
    # a multi-level detection is counted once per contributing level, so the
    # per-level counts sum to more than the detection count. Intended, and more
    # pronounced under class-agnostic IoS matching, which forms larger clusters.
    assert sum(v["final_merge_count"] for v in diag.values()) == 3 > len(rows)


def test_gt_recall_matching_is_class_aware():
    gt = {"f": [{"class_id": 0, "bbox_xyxy": [0, 0, 100, 100]}]}
    wrong_class = {"f": [{"class_id": 1, "bbox_xyxy": [0, 0, 100, 100], "score": 0.9}]}
    right_class = {"f": [{"class_id": 0, "bbox_xyxy": [0, 0, 100, 100], "score": 0.9}]}
    assert gt_recall_by_size(gt, wrong_class)["64-128"]["recall50"] == 0.0
    assert gt_recall_by_size(gt, right_class)["64-128"]["recall50"] == 1.0


def test_coco_map_scores_empty_paths():
    class_map = {0: "a"}
    empty = coco_map_scores({"f": []}, {"f": []}, class_map)
    assert empty["mAP50"] == 0.0 and empty["per_class_ap50_95"] == {"a": 0.0}
