"""Backend arguments, prepared gates, canonical training artifacts, and CLI."""

from __future__ import annotations

import csv
import json
import os
import shutil

import cv2
import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from sati.backends import BackendTrainingArtifacts, UltralyticsBackend
from sati.cli import app
from sati.config_loader import compose_and_resolve
from sati.hashing import (
    class_mapping_hash,
    class_mapping_hash_full_source,
    preprocessing_hash,
)
from sati.manifests import HashWithSource, ManifestMetadata, SCHEMA_VERSION, write_metadata_sidecar
from sati.train import (
    TrainRefusedError,
    check_prepared_gate,
    copy_native_figures,
    execute_train,
    locate_data_yaml,
    normalize_epoch_metrics,
)


def _cfg(tmp_path, overrides=()):
    cfg = compose_and_resolve(
        [
            "data.source.uri=unused.json",
            f"data.prepared.uri={tmp_path}/prepared",
            "model.num_classes=2",
            f"training_outputs.dir={tmp_path}/run",
            *overrides,
        ],
        command="train",
        base_dir=tmp_path,
    )
    cfg.data.prepared.dataset_hash = "sha256:" + "d" * 64
    cfg.data.prepared.prepared_dataset_hash = "sha256:" + "p" * 64
    cfg.data.prepared.prepared_dataset_id = "prepared-product"
    return cfg


def _fake_prepared_tree(tmp_path, cfg):
    root = tmp_path / "prepared"
    yolo = root / "yolo"
    yolo.mkdir(parents=True)
    (yolo / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(yolo),
                "train": "images/training",
                "val": "images/validation",
                "names": {0: "ring", 1: "spiral"},
                "sati_dataset_hash": cfg.data.prepared.dataset_hash,
                "sati_prepared_dataset_hash": cfg.data.prepared.prepared_dataset_hash,
                "sati_preprocessing_hash": preprocessing_hash(cfg.slicing),
            }
        )
    )
    write_metadata_sidecar(
        ManifestMetadata(
            schema_version=SCHEMA_VERSION,
            class_mapping={"0": "ring", "1": "spiral"},
            hashes={
                "class_mapping_hash": HashWithSource(
                    value=class_mapping_hash({0: "ring", 1: "spiral"}),
                    source=class_mapping_hash_full_source({0: "ring", 1: "spiral"}),
                )
            },
        ),
        root / "manifests",
    )
    return yolo / "data.yaml"


class FakeBackend:
    def __init__(self, root):
        self.root = root
        self.calls = []

    def train(self, data_yaml, cfg, *, resume_from=None):
        self.calls.append(("train", data_yaml, resume_from))
        weights = self.root / "weights"
        weights.mkdir(parents=True)
        best, last = weights / "best.pt", weights / "last.pt"
        best.write_bytes(b"best")
        last.write_bytes(b"last")
        with (self.root / "results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "metrics/mAP50(B)", "x"])
            writer.writeheader()
            writer.writerow({"epoch": 0, "metrics/mAP50(B)": 0.5, "x": 7})
        (self.root / "results.png").write_bytes(b"figure")
        return BackendTrainingArtifacts(root=self.root, best=best, last=last)

    def load(self, model_ref):
        self.calls.append(("load", model_ref))

    def predict(self, slices, *, imgsz, confidence_threshold):
        return [[] for _ in slices]


def test_locate_data_yaml_and_prepared_gate(tmp_path):
    cfg = _cfg(tmp_path)
    data_yaml = _fake_prepared_tree(tmp_path, cfg)
    assert locate_data_yaml(cfg) == data_yaml
    check_prepared_gate(data_yaml, cfg)
    cfg.data.prepared.prepared_dataset_hash = "sha256:" + "x" * 64
    with pytest.raises(TrainRefusedError, match="expected.*observed"):
        check_prepared_gate(data_yaml, cfg)


def test_training_normalization_helpers(tmp_path):
    native = tmp_path / "backend"
    native.mkdir()
    with (native / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "metrics/mAP50(B)", "extra"])
        writer.writeheader()
        writer.writerow({"epoch": 0, "metrics/mAP50(B)": 0.4, "extra": 9})
    (native / "results.png").write_bytes(b"png")
    metrics = normalize_epoch_metrics(native / "results.csv", tmp_path / "metrics/metrics.csv")
    assert metrics.read_text().splitlines()[0] == (
        "epoch,validation.map50,ultralytics.extra"
    )
    figures = copy_native_figures(native, tmp_path / "figures")
    assert figures == [tmp_path / "figures" / "ultralytics" / "results.png"]


@pytest.mark.parametrize("figures_enabled", [True, False])
def test_execute_train_canonicalizes_artifacts(tmp_path, monkeypatch, figures_enabled):
    cfg = _cfg(
        tmp_path,
        [f"training_outputs.figures.enabled={str(figures_enabled).lower()}"],
    )
    _fake_prepared_tree(tmp_path, cfg)
    native = cfg.training_outputs.backend.dir / "ultralytics"
    backend = FakeBackend(native)

    def fake_finalize(source, *, role, output_dir, contract):
        output_dir.mkdir(parents=True, exist_ok=True)
        digest = "sha256:" + (role[0] * 64)
        destination = output_dir / f"{role}.{digest[-6:]}.pt"
        shutil.copy2(source, destination)
        return destination, digest

    monkeypatch.setattr("sati.train.finalize_checkpoint", fake_finalize)
    checkpoint, digest = execute_train(cfg, backend=backend)
    assert checkpoint.parent == cfg.training_outputs.checkpoints.dir
    assert digest.startswith("sha256:")
    assert (cfg.training_outputs.checkpoints.dir / "checkpoint_metadata.json").is_file()
    assert (cfg.training_outputs.metrics.dir / "metrics.csv").is_file()
    figure = cfg.training_outputs.figures.dir / "ultralytics" / "results.png"
    assert figure.is_file() is figures_enabled
    assert native == cfg.training_outputs.dir / "backend" / "ultralytics"


def test_ultralytics_backend_train_kwargs(tmp_path, monkeypatch):
    captured = {}
    # Set exactly one augment; every other augment must stay out of the call so
    # stock Ultralytics defaults apply. Stock Ultralytics rejects ``knob=None``,
    # so the fake asserts what the real backend would enforce.
    cfg = _cfg(
        tmp_path,
        [
            "training.epochs=3",
            "training.lr=0.01",
            "training.device=cpu",
            "+training.yolo_augment.fliplr=0.5",
        ],
    )

    class FakeYolo:
        def __init__(self, ref):
            captured["ref"] = ref

        def train(self, **kwargs):
            captured["kwargs"] = kwargs
            # Stock Ultralytics raises TypeError on any ``knob=None``.
            none_kwargs = [name for name, value in kwargs.items() if value is None]
            assert none_kwargs == [], f"None-valued kwargs reached Ultralytics: {none_kwargs}"
            weights = cfg.training_outputs.backend.dir / "ultralytics" / "weights"
            weights.mkdir(parents=True)
            (weights / "best.pt").write_bytes(b"best")
            (weights / "last.pt").write_bytes(b"last")

    monkeypatch.setattr(UltralyticsBackend, "_yolo", staticmethod(FakeYolo))
    artifacts = UltralyticsBackend().train(tmp_path / "data.yaml", cfg)
    assert captured["ref"] == "yolo26n.pt"
    assert captured["kwargs"]["seed"] == cfg.runtime.seed
    assert captured["kwargs"]["project"] == str(cfg.training_outputs.backend.dir)
    assert captured["kwargs"]["name"] == "ultralytics"
    # SATI augmentation is off by default: every numeric knob is forwarded at
    # its off-value, the one authored knob carries its value, and auto_augment
    # (a detection no-op with no off-value) stays null and is never forwarded.
    assert captured["kwargs"]["fliplr"] == 0.5
    for off in ("hsv_h", "hsv_s", "hsv_v", "translate", "scale", "flipud", "mosaic", "erasing"):
        assert captured["kwargs"][off] == 0.0
    assert captured["kwargs"]["close_mosaic"] == 0
    assert "auto_augment" not in captured["kwargs"]
    assert artifacts.best.is_file() and artifacts.last.is_file()


def test_backend_predict_requires_load():
    with pytest.raises(RuntimeError, match="call load"):
        UltralyticsBackend().predict(
            [np.zeros((8, 8, 3), np.uint8)],
            imgsz=8,
            confidence_threshold=0.1,
            iou=0.7,
            agnostic_nms=False,
        )


def _tiny_coco(tmp_path):
    image = np.zeros((96, 160, 3), np.uint8)
    cv2.imwrite(str(tmp_path / "img1.png"), image)
    (tmp_path / "anns.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "img1.png", "width": 160, "height": 96}
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [30, 20, 60, 40],
                    }
                ],
                "categories": [{"id": 1, "name": "ring"}],
            }
        )
    )


def test_cli_prepare_end_to_end_and_reuse(tmp_path, monkeypatch):
    _tiny_coco(tmp_path)
    monkeypatch.chdir(tmp_path)
    args = [
        "prepare",
        "data.source.uri=anns.json",
        "model.num_classes=1",
        "slicing.nominal_slice_size=64",
        "slicing.size_gate_px=4",
        f"prepare_outputs.dir={tmp_path}/prepared",
    ]
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 0, first.output
    root = tmp_path / "prepared"
    assert (root / "config" / "resolved.yaml").is_file()
    assert json.loads((root / "prepare_manifest.json").read_text())["status"] == "complete"
    second = CliRunner().invoke(app, args)
    assert second.exit_code == 0
    assert "reused" in second.output


@pytest.mark.skipif(
    not os.environ.get("SATI_RUN_TRIPWIRE"),
    reason="set SATI_RUN_TRIPWIRE=1 to run the real Ultralytics coordinate tripwire",
)
def test_ultralytics_coordinate_tripwire():
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 255, (640, 640, 3), dtype=np.uint8)
    backend = UltralyticsBackend()
    backend.load("yolo26n.pt")
    (detections,) = backend.predict(
        [pixels], imgsz=640, confidence_threshold=0.001, iou=0.7, agnostic_nms=False
    )
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox_xyxy
        assert 0 <= x1 <= x2 <= 640 and 0 <= y1 <= y2 <= 640
