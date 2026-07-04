"""Backend arg-passing, train gates, prepare/train CLI (v1 §7, §8, §13)."""

import json
import os

import cv2
import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from sati.backends import UltralyticsBackend
from sati.cli import app
from sati.config_loader import compose_and_resolve
from sati.hashing import preprocessing_hash
from sati.train import TrainRefusedError, check_preprocessing_gate, execute_train, locate_data_yaml


def _cfg(tmp_path, overrides=()):
    return compose_and_resolve(
        [
            "data.source_path=prepared",
            "model.num_classes=2",
            f"output.run_dir={tmp_path}/run",
            *overrides,
        ],
        write=False,
        base_dir=tmp_path,
    )


def _fake_prepared_tree(tmp_path, slicing):
    yolo = tmp_path / "prepared" / "yolo"
    yolo.mkdir(parents=True)
    (yolo / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(yolo),
                "train": "images/training",
                "val": "images/validation",
                "names": {0: "ring", 1: "spiral"},
                "sati_preprocessing_hash": preprocessing_hash(slicing),
            }
        )
    )
    return yolo / "data.yaml"


class FakeBackend:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.calls = []

    def train(self, data_yaml, cfg):
        self.calls.append(("train", data_yaml))
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint.write_bytes(b"fake weights")
        return self.checkpoint

    def load(self, model_ref):
        self.calls.append(("load", model_ref))

    def predict(self, slices, *, imgsz, confidence_threshold):
        return [[] for _ in slices]


def test_locate_data_yaml_forms(tmp_path):
    cfg = _cfg(tmp_path)
    data_yaml = _fake_prepared_tree(tmp_path, cfg.slicing)
    assert locate_data_yaml(cfg) == data_yaml  # root form
    direct = _cfg(tmp_path, [f"data.source_path={data_yaml}"])
    assert locate_data_yaml(direct) == data_yaml  # direct data.yaml form
    with pytest.raises(TrainRefusedError, match="run `sati prepare` first"):
        locate_data_yaml(_cfg(tmp_path, ["data.source_path=nowhere"]))


def test_preprocessing_gate(tmp_path):
    cfg = _cfg(tmp_path)
    data_yaml = _fake_prepared_tree(tmp_path, cfg.slicing)
    check_preprocessing_gate(data_yaml, cfg)  # matching: no raise
    mismatched = _cfg(tmp_path, ["slicing.overlap_ratio=0.4"])
    with pytest.raises(TrainRefusedError, match="preprocessing_hash mismatch"):
        check_preprocessing_gate(data_yaml, mismatched)
    spec = yaml.safe_load(data_yaml.read_text())
    del spec["sati_preprocessing_hash"]
    data_yaml.write_text(yaml.safe_dump(spec))
    with pytest.raises(TrainRefusedError, match="not a SATI-prepared tree"):
        check_preprocessing_gate(data_yaml, cfg)


def test_execute_train_returns_checkpoint_and_hash(tmp_path):
    cfg = _cfg(tmp_path)
    _fake_prepared_tree(tmp_path, cfg.slicing)
    backend = FakeBackend(cfg.output.run_dir / "train" / "weights" / "best.pt")
    checkpoint, ckpt_hash = execute_train(cfg, backend=backend)
    assert checkpoint.read_bytes() == b"fake weights"
    assert ckpt_hash.startswith("sha256:")


def test_ultralytics_backend_train_kwargs(tmp_path, monkeypatch):
    captured = {}

    class FakeYolo:
        def __init__(self, ref):
            captured["ref"] = ref

        def train(self, **kwargs):
            captured["kwargs"] = kwargs
            ckpt = tmp_path / "run" / "train" / "weights" / "best.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            ckpt.write_bytes(b"w")

    monkeypatch.setattr(UltralyticsBackend, "_yolo", staticmethod(FakeYolo))
    cfg = _cfg(tmp_path, ["training.epochs=3", "training.lr=0.01", "training.device=cpu"])
    checkpoint = UltralyticsBackend().train(tmp_path / "d.yaml", cfg)
    assert captured["ref"] == "yolo26n.pt"  # arch → hub-pretrained default
    kw = captured["kwargs"]
    assert kw["imgsz"] == 1024  # derived from nominal_slice_size
    assert kw["epochs"] == 3 and kw["lr0"] == 0.01 and kw["device"] == "cpu"
    assert kw["seed"] == 0 and kw["name"] == "train"
    assert checkpoint.is_file()

    cfg_weights = _cfg(tmp_path, ["model.weights=custom.pt"])
    UltralyticsBackend().train(tmp_path / "d.yaml", cfg_weights)
    assert captured["ref"] == "custom.pt"  # explicit weights win
    assert "lr0" not in captured["kwargs"]  # lr None → backend default


def test_ultralytics_backend_train_yolo_augment_kwargs(tmp_path, monkeypatch):
    captured = {}

    class FakeYolo:
        def train(self, **kwargs):
            captured["kwargs"] = kwargs
            ckpt = tmp_path / "run" / "train" / "weights" / "best.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            ckpt.write_bytes(b"w")

    monkeypatch.setattr(UltralyticsBackend, "_yolo", staticmethod(lambda ref: FakeYolo()))
    cfg = _cfg(
        tmp_path,
        [
            "+training.yolo_augment.hsv_h=0.0",
            "+training.yolo_augment.hsv_s=0.0",
            "+training.yolo_augment.hsv_v=0.0",
            "+training.yolo_augment.translate=0.0",
            "+training.yolo_augment.scale=0.0",
            "+training.yolo_augment.flipud=0.5",
            "+training.yolo_augment.fliplr=0.5",
            "+training.yolo_augment.mosaic=0.0",
            "+training.yolo_augment.close_mosaic=0",
            "+training.yolo_augment.auto_augment=null",
            "+training.yolo_augment.erasing=0.0",
        ],
    )
    UltralyticsBackend().train(tmp_path / "d.yaml", cfg)
    assert {
        key: captured["kwargs"][key]
        for key in (
            "hsv_h",
            "hsv_s",
            "hsv_v",
            "translate",
            "scale",
            "flipud",
            "fliplr",
            "mosaic",
            "close_mosaic",
            "auto_augment",
            "erasing",
        )
    } == {
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "flipud": 0.5,
        "fliplr": 0.5,
        "mosaic": 0.0,
        "close_mosaic": 0,
        "auto_augment": None,
        "erasing": 0.0,
    }


def test_backend_predict_requires_load():
    with pytest.raises(RuntimeError, match="call load"):
        UltralyticsBackend().predict([np.zeros((8, 8, 3), np.uint8)], imgsz=8, confidence_threshold=0.1)


def _tiny_coco(tmp_path):
    img = np.zeros((96, 160, 3), np.uint8)
    img[20:60, 30:90] = 255
    cv2.imwrite(str(tmp_path / "img1.png"), img)
    (tmp_path / "anns.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "img1.png", "width": 160, "height": 96}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [30, 20, 60, 40]}
                ],
                "categories": [{"id": 1, "name": "ring"}],
            }
        )
    )


def test_cli_prepare_end_to_end(tmp_path, monkeypatch):
    _tiny_coco(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "prepare",
            "data.source_path=anns.json",
            "model.num_classes=1",
            "slicing.nominal_slice_size=64",
            "slicing.size_gate_px=4",
            f"output.run_dir={tmp_path}/run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "run" / "resolved.yaml").is_file()
    assert (tmp_path / "run" / "manifests" / "slices.parquet").is_file()
    assert (tmp_path / "run" / "yolo" / "data.yaml").is_file()
    assert "level" in result.output  # per-level diagnostic table printed


@pytest.mark.skipif(
    not os.environ.get("SATI_RUN_TRIPWIRE"),
    reason="ultralytics coordinate-contract tripwire; set SATI_RUN_TRIPWIRE=1 "
    "(downloads hub weights on first run)",
)
def test_ultralytics_coordinate_tripwire():
    """§7 NOTE: assert returned boxes are in emitted-slice pixels with no letterbox.

    Uses hub-pretrained weights on a noise slice at conf≈0: whatever boxes come
    back must be in-bounds of the input's own pixel grid.
    """
    rng = np.random.default_rng(0)
    slice_px = rng.integers(0, 255, (640, 640, 3), dtype=np.uint8)
    backend = UltralyticsBackend()
    backend.load("yolo26n.pt")
    (detections,) = backend.predict([slice_px], imgsz=640, confidence_threshold=0.001)
    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        assert 0 <= x1 <= x2 <= 640 and 0 <= y1 <= y2 <= 640
        assert 0 < det.score <= 1
