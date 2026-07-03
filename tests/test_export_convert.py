"""Export sidecar (§10) and convert interchange (§5)."""

import json

import pytest
import yaml

from sati.config_loader import compose_and_resolve
from sati.convert import execute_convert
from sati.export import execute_export, write_inference_config
from sati.hashing import checkpoint_hash
from sati.io import IngestError, ingest_coco


def _cfg(tmp_path, overrides=()):
    return compose_and_resolve(
        [
            "data.source_path=anns.json",
            "model.num_classes=2",
            f"output.run_dir={tmp_path}/run",
            *overrides,
        ],
        write=False,
        base_dir=tmp_path,
    )


def _write_coco(tmp_path):
    (tmp_path / "anns.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "img1.png", "width": 640, "height": 480},
                    {"id": 2, "file_name": "img2.png", "width": 320, "height": 240},
                ],
                "annotations": [
                    {"id": 10, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]},
                    {"id": 11, "image_id": 2, "category_id": 3, "bbox": [5, 5, 50, 60]},
                ],
                "categories": [{"id": 1, "name": "ring"}, {"id": 3, "name": "spiral"}],
            }
        )
    )
    for name in ("img1.png", "img2.png"):
        (tmp_path / name).write_bytes(b"fake png")  # never decoded by convert


# --- export -----------------------------------------------------------------


class FakeExportBackend:
    def train(self, data_yaml, cfg):
        raise NotImplementedError

    def load(self, model_ref):
        pass

    def predict(self, slices, *, imgsz, confidence_threshold):
        return []

    def export(self, checkpoint, *, fmt, imgsz):
        self.called = (checkpoint, fmt, imgsz)
        return checkpoint.with_suffix(f".{fmt}")


def test_execute_export_and_sidecar(tmp_path):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"weights")
    cfg = _cfg(tmp_path, [f"model.weights={ckpt}"])
    backend = FakeExportBackend()
    artifact, sidecar = execute_export(cfg, fmt="onnx", backend=backend)
    assert backend.called == (ckpt, "onnx", 1024)
    assert artifact == ckpt.with_suffix(".onnx")
    spec = yaml.safe_load(sidecar.read_text())
    assert spec["slicing"]["nominal_slice_size"] == 1024
    assert spec["slicing"]["stretch_policy"] == "stretch-to-nominal/v1"
    assert spec["recombination"]["within_level_iou"] == 0.5
    assert spec["hashes"]["checkpoint_hash"] == checkpoint_hash(ckpt)
    assert spec["class_map"] == {}  # no prepared root at data.source_path


def test_export_requires_local_checkpoint(tmp_path):
    cfg = _cfg(tmp_path, ["model.weights=hub-only.pt"])
    with pytest.raises(IngestError, match="local checkpoint file"):
        execute_export(cfg, backend=FakeExportBackend())


def test_inference_config_picks_up_prepared_class_map(tmp_path):
    _write_coco(tmp_path)
    cfg = _cfg(tmp_path)
    execute_convert(cfg, to="manifests")  # creates run/manifests with sidecar
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"w")
    cfg2 = _cfg(tmp_path, [f"data.source_path={tmp_path}/run", f"model.weights={ckpt}"])
    sidecar = write_inference_config(cfg2, ckpt, tmp_path / "out")
    spec = yaml.safe_load(sidecar.read_text())
    assert spec["class_map"] == {"0": "ring", "1": "spiral"}


# --- convert ----------------------------------------------------------------


def test_convert_coco_to_manifests_round_trip(tmp_path):
    _write_coco(tmp_path)
    cfg = _cfg(tmp_path)
    manifests_dir = execute_convert(cfg, to="manifests")
    assert (manifests_dir / "fullframes.parquet").is_file()
    assert not (manifests_dir / "slices.parquet").exists()  # convert never slices

    coco_dir = execute_convert(
        _cfg(tmp_path, [f"data.source_path={tmp_path}/run"]), to="coco"
    )
    data = json.loads((coco_dir / "training.json").read_text())
    assert {c["name"] for c in data["categories"]} == {"ring", "spiral"}
    ann = next(a for a in data["annotations"] if a["id"] == 10)
    assert ann["bbox"] == [10.0, 20.0, 30.0, 40.0]  # xyxy → xywh round trip

    # and the emitted COCO re-ingests cleanly
    again = ingest_coco(coco_dir / "training.json")
    assert {f.fullframe_id for f in again.frames} == {"1", "2"}
    box = next(a for a in again.annotations if a.ann_id == 10).bbox_xyxy
    assert box == (10.0, 20.0, 40.0, 60.0)


def test_convert_manifests_to_yolo(tmp_path):
    _write_coco(tmp_path)
    execute_convert(_cfg(tmp_path), to="manifests")
    out = execute_convert(_cfg(tmp_path, [f"data.source_path={tmp_path}/run"]), to="yolo")
    spec = yaml.safe_load((out / "data.yaml").read_text())
    assert spec["train"] == "images/training" and "val" not in spec
    assert spec["names"] == {0: "ring", 1: "spiral"}
    link = out / "images" / "training" / "1.png"
    assert link.is_symlink() and link.resolve() == (tmp_path / "img1.png").resolve()
    label = (out / "labels" / "training" / "1.txt").read_text().strip().split()
    assert label[0] == "0"  # remapped class id
    # cxcywh normalized: (10,20,40,60) on 640×480 → cx=25/640, w=30/640 ...
    # labels are %.6f-truncated text
    assert [float(v) for v in label[1:]] == pytest.approx(
        [25 / 640, 40 / 480, 30 / 640, 40 / 480], abs=1e-6
    )


def test_convert_unknown_target(tmp_path):
    _write_coco(tmp_path)
    with pytest.raises(IngestError, match="unknown convert target"):
        execute_convert(_cfg(tmp_path), to="tfrecord")
